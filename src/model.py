"""构建一个在imageNet上经过预训练的Resnet34为encoder的Unet架构"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet34_Weights, resnet34
from src.prototype_memory import load_prototype_bank
from src.transformer_blocks import PrototypeCrossAttention, SkipAttentionGate, TransformerBottleneck

class ConvBlock(nn.Module):
    """
    Unet解码器，每个阶段执行两个3*3的卷积加激活
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
    
class DecoderBlock(nn.Module):
    """解码器的整体框架，双线性插值上采样 + skip connection + ConvBlock"""
    def __init__(self, in_channels,skip_channels,out_channels):
        super().__init__()
        #拼接的是通道
        self.conv_block = ConvBlock(in_channels+skip_channels, out_channels)

    def forward(self,x,skip):
        # 将上一阶段的x上采样到和skip过来的通道维一样的尺寸大小, x代表上一层的特征图，skip代表解码器过来的特征图
        x = F.interpolate(
            x,
            size= skip.shape[-2:],
            mode="bilinear" ,#双线性插值放大
            align_corners=False
        )

        x = torch.cat([x,skip], dim=1) #dim=0->batch size dim=1 channels
        x = self.conv_block(x)

        return x
    
def make_aux_head(in_channels, mid_channels=32):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, 1, kernel_size=1),
    )


class UNetResNet34(nn.Module):
    """ 编码器用RESNET34, 解码器用unet解码器的分割模型实现"""
    def __init__(
        self,
        encoder_weights=None,
        deep_supervision=False,
        boundary_aux=False,
        transformer_bottleneck_enable=False,
        transformer_bottleneck_layers=0,
        transformer_bottleneck_heads=8,
        transformer_bottleneck_dropout=0.1,
        prototype_attention_enable=False,
        prototype_bank_path="",
        prototype_attention_heads=8,
        prototype_attention_dropout=0.1,
        skip_attention_enable=False,
        skip_attention_levels=None,
    ):
        super().__init__()
        self._encoder_trainable = True
        self.deep_supervision = bool(deep_supervision)
        self.boundary_aux = bool(boundary_aux)
        self.transformer_bottleneck_enable = bool(transformer_bottleneck_enable)
        self.prototype_attention_enable = bool(prototype_attention_enable)
        self.skip_attention_enable = bool(skip_attention_enable)
        self.skip_attention_levels = set(skip_attention_levels or ["d4", "d3"])

        #======================================================================================
        #构建Resnet编码器
        encoder = resnet34(weights=encoder_weights)

        self.encoder_stem = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
        )
        self.encoder_pool = encoder.maxpool

        #resnet的四个主层
        self.encoder_layer1 = encoder.layer1
        self.encoder_layer2 = encoder.layer2
        self.encoder_layer3 = encoder.layer3
        self.encoder_layer4 = encoder.layer4



        ####################################################################################
        #bottleneck
        self.center = ConvBlock(512,512)
        if self.transformer_bottleneck_enable and int(transformer_bottleneck_layers) > 0:
            self.transformer_bottleneck = TransformerBottleneck(
                channels=512,
                num_layers=int(transformer_bottleneck_layers),
                num_heads=int(transformer_bottleneck_heads),
                dropout=float(transformer_bottleneck_dropout),
            )
        else:
            self.transformer_bottleneck = None

        if self.prototype_attention_enable:
            bank = load_prototype_bank(prototype_bank_path, map_location="cpu")
            self.prototype_attention = PrototypeCrossAttention(
                pos_prototypes=bank["pos_prototypes"],
                neg_prototypes=bank["neg_prototypes"],
                channels=512,
                num_heads=int(prototype_attention_heads),
                dropout=float(prototype_attention_dropout),
            )
        else:
            self.prototype_attention = None


        #=======================================================================
        #decoder
        self.decoder4 = DecoderBlock(in_channels=512,skip_channels=256,out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        self.skip_gate_d4 = (
            SkipAttentionGate(skip_channels=256, gate_channels=512)
            if self.skip_attention_enable and "d4" in self.skip_attention_levels
            else None
        )
        self.skip_gate_d3 = (
            SkipAttentionGate(skip_channels=128, gate_channels=256)
            if self.skip_attention_enable and "d3" in self.skip_attention_levels
            else None
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64,32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,1,kernel_size=1),
        )
        if self.deep_supervision:
            self.deep_supervision_heads = nn.ModuleList(
                [
                    make_aux_head(256),
                    make_aux_head(128),
                    make_aux_head(64),
                ]
            )
        else:
            self.deep_supervision_heads = nn.ModuleList()

        if self.boundary_aux:
            self.boundary_head = make_aux_head(64, mid_channels=16)
        else:
            self.boundary_head = None

    def encoder_modules(self):
        return [
            self.encoder_stem,
            self.encoder_layer1,
            self.encoder_layer2,
            self.encoder_layer3,
            self.encoder_layer4,
        ]

    def encoder_parameters(self):
        """
        返回“编码器部分”的参数。
        """

        for module in self.encoder_modules():
            yield from module.parameters()        

    def decoder_parameters(self):
            
        modules = [
            self.center,
            self.transformer_bottleneck,
            self.prototype_attention,
            self.decoder4,
            self.decoder3,
            self.decoder2,
            self.decoder1,
            self.skip_gate_d4,
            self.skip_gate_d3,
            self.segmentation_head,
            self.deep_supervision_heads,
            ]
        if self.boundary_head is not None:
            modules.append(self.boundary_head)
        for module in modules:
            if module is not None:
                yield from module.parameters()

    def set_encoder_trainable(self, trainable:bool):
        self._encoder_trainable = bool(trainable)
        for param in self.encoder_parameters():
            param.requires_grad = trainable

    def apply_encoder_freeze_mode(self):
        if self._encoder_trainable:
            return

        for module in self.encoder_modules():
            module.eval()


    def forward(self,x):
        input_size = x.shape[-2:]
        x0 = self.encoder_stem(x)
        x1 = self.encoder_layer1(self.encoder_pool(x0))
        x2=self.encoder_layer2(x1)
        x3=self.encoder_layer3(x2)
        x4 = self.encoder_layer4(x3)
        x4 = self.center(x4)
        if self.transformer_bottleneck is not None:
            x4 = self.transformer_bottleneck(x4)
        if self.prototype_attention is not None:
            x4 = self.prototype_attention(x4)

        skip3 = self.skip_gate_d4(x3, x4) if self.skip_gate_d4 is not None else x3
        d4 = self.decoder4(x4, skip3)
        skip2 = self.skip_gate_d3(x2, d4) if self.skip_gate_d3 is not None else x2
        d3 = self.decoder3(d4, skip2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)
        d1 = F.interpolate(
            d1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        logits = self.segmentation_head(d1)

        if not self.deep_supervision and not self.boundary_aux:
            return logits

        output = {"logits": logits}
        if self.deep_supervision:
            aux_features = [d4, d3, d2]
            aux_logits = []
            for head, features in zip(self.deep_supervision_heads, aux_features):
                aux_logits.append(
                    F.interpolate(
                        head(features),
                        size=input_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            output["aux_logits"] = aux_logits

        if self.boundary_aux and self.boundary_head is not None:
            output["boundary_logits"] = self.boundary_head(d1)

        return output
       


def build_model(
    pretrained=True,
    deep_supervision=False,
    boundary_aux=False,
    transformer_bottleneck_enable=False,
    transformer_bottleneck_layers=0,
    transformer_bottleneck_heads=8,
    transformer_bottleneck_dropout=0.1,
    prototype_attention_enable=False,
    prototype_bank_path="",
    prototype_attention_heads=8,
    prototype_attention_dropout=0.1,
    skip_attention_enable=False,
    skip_attention_levels=None,
):
    """
    构建模型实例。
    """

    # 如果不需要预训练，直接用随机初始化
    if not pretrained:
        model = UNetResNet34(
            encoder_weights=None,
            deep_supervision=deep_supervision,
            boundary_aux=boundary_aux,
            transformer_bottleneck_enable=transformer_bottleneck_enable,
            transformer_bottleneck_layers=transformer_bottleneck_layers,
            transformer_bottleneck_heads=transformer_bottleneck_heads,
            transformer_bottleneck_dropout=transformer_bottleneck_dropout,
            prototype_attention_enable=prototype_attention_enable,
            prototype_bank_path=prototype_bank_path,
            prototype_attention_heads=prototype_attention_heads,
            prototype_attention_dropout=prototype_attention_dropout,
            skip_attention_enable=skip_attention_enable,
            skip_attention_levels=skip_attention_levels,
        )
        return model

    # 如果需要预训练，就尝试加载官方默认权重
    try:
        weights = ResNet34_Weights.DEFAULT
        model = UNetResNet34(
            encoder_weights=weights,
            deep_supervision=deep_supervision,
            boundary_aux=boundary_aux,
            transformer_bottleneck_enable=transformer_bottleneck_enable,
            transformer_bottleneck_layers=transformer_bottleneck_layers,
            transformer_bottleneck_heads=transformer_bottleneck_heads,
            transformer_bottleneck_dropout=transformer_bottleneck_dropout,
            prototype_attention_enable=prototype_attention_enable,
            prototype_bank_path=prototype_bank_path,
            prototype_attention_heads=prototype_attention_heads,
            prototype_attention_dropout=prototype_attention_dropout,
            skip_attention_enable=skip_attention_enable,
            skip_attention_levels=skip_attention_levels,
        )
        return model

    except Exception as e:
        # 如果加载失败（比如没网、没缓存），
        # 就退回到随机初始化，避免整个程序直接崩掉
        print("Warning: 预训练权重加载失败，将改为随机初始化。")
        print(f"Detail: {e}")

        model = UNetResNet34(
            encoder_weights=None,
            deep_supervision=deep_supervision,
            boundary_aux=boundary_aux,
            transformer_bottleneck_enable=transformer_bottleneck_enable,
            transformer_bottleneck_layers=transformer_bottleneck_layers,
            transformer_bottleneck_heads=transformer_bottleneck_heads,
            transformer_bottleneck_dropout=transformer_bottleneck_dropout,
            prototype_attention_enable=prototype_attention_enable,
            prototype_bank_path=prototype_bank_path,
            prototype_attention_heads=prototype_attention_heads,
            prototype_attention_dropout=prototype_attention_dropout,
            skip_attention_enable=skip_attention_enable,
            skip_attention_levels=skip_attention_levels,
        )
        return model


def build_model_from_config(cfg):
    return build_model(
        pretrained=bool(cfg.get("pretrained", False)),
        deep_supervision=bool(cfg.get("deep_supervision_enable", False)),
        boundary_aux=bool(cfg.get("boundary_aux_enable", False)),
        transformer_bottleneck_enable=bool(cfg.get("transformer_bottleneck_enable", False)),
        transformer_bottleneck_layers=int(cfg.get("transformer_bottleneck_layers", 0)),
        transformer_bottleneck_heads=int(cfg.get("transformer_bottleneck_heads", 8)),
        transformer_bottleneck_dropout=float(cfg.get("transformer_bottleneck_dropout", 0.1)),
        prototype_attention_enable=bool(cfg.get("prototype_attention_enable", False)),
        prototype_bank_path=cfg.get("prototype_bank_path", ""),
        prototype_attention_heads=int(cfg.get("prototype_attention_heads", cfg.get("transformer_bottleneck_heads", 8))),
        prototype_attention_dropout=float(cfg.get("prototype_attention_dropout", cfg.get("transformer_bottleneck_dropout", 0.1))),
        skip_attention_enable=bool(cfg.get("skip_attention_enable", False)),
        skip_attention_levels=cfg.get("skip_attention_levels", ["d4", "d3"]),
    )
