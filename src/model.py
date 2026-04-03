"""构建一个在imageNet上经过预训练的Resnet34为encoder的Unet架构"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet34_Weights, resnet34

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
    
class UNetResNet34(nn.Module):
    """ 编码器用RESNET34, 解码器用unet解码器的分割模型实现"""
    def __init__(self, encoder_weights=None):
        super().__init__()

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


        #=======================================================================
        #decoder
        self.decoder4 = DecoderBlock(in_channels=512,skip_channels=256,out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64,32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,1,kernel_size=1),
        )        

    def encoder_parameters(self):
        """
        返回“编码器部分”的参数。
        """

        modules = [
            self.encoder_stem,
            self.encoder_layer1,
            self.encoder_layer2,
            self.encoder_layer3,
            self.encoder_layer4,
        ]

        for module in modules:
            yield from module.parameters()        

    def decoder_parameters(self):
            
        modules = [
            self.center,
            self.decoder4,
            self.decoder3,
            self.decoder2,
            self.decoder1,
            self.segmentation_head,
            ]
        for module in modules:
            yield from module.parameters()

    def set_encoder_trainable(self, trainable:bool):
        for param in self.encoder_parameters():
            param.requires_grad = trainable    


    def forward(self,x):
        input_size = x.shape[-2:]
        x0 = self.encoder_stem(x)
        x1 = self.encoder_layer1(self.encoder_pool(x0))
        x2=self.encoder_layer2(x1)
        x3=self.encoder_layer3(x2)
        x4 = self.encoder_layer4(x3)
        x4 = self.center(x4)
        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)
        d1 = F.interpolate(
            d1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        logits = self.segmentation_head(d1)
        return logits
       


def build_model(pretrained=True):
    """
    构建模型实例。
    """

    # 如果不需要预训练，直接用随机初始化
    if not pretrained:
        model = UNetResNet34(encoder_weights=None)
        return model

    # 如果需要预训练，就尝试加载官方默认权重
    try:
        weights = ResNet34_Weights.DEFAULT
        model = UNetResNet34(encoder_weights=weights)
        return model

    except Exception as e:
        # 如果加载失败（比如没网、没缓存），
        # 就退回到随机初始化，避免整个程序直接崩掉
        print("Warning: 预训练权重加载失败，将改为随机初始化。")
        print(f"Detail: {e}")

        model = UNetResNet34(encoder_weights=None)
        return model


