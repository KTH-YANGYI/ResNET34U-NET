import torch.nn.functional as F

from src.models.resnet34_unet_baseline import ResNet34UNetBaseline
from src.prototype_memory import load_prototype_bank
from src.transformer_blocks import PrototypeCrossAttention, SkipAttentionGate, TransformerBottleneck


class TransformerBottleneckUNet(ResNet34UNetBaseline):
    model_variant = "tbn_d1"

    def __init__(
        self,
        encoder_weights=None,
        transformer_bottleneck_layers=1,
        transformer_bottleneck_heads=8,
        transformer_bottleneck_dropout=0.1,
    ):
        super().__init__(encoder_weights=encoder_weights)
        self.transformer_bottleneck = TransformerBottleneck(
            channels=512,
            num_layers=int(transformer_bottleneck_layers),
            num_heads=int(transformer_bottleneck_heads),
            dropout=float(transformer_bottleneck_dropout),
        )

    def bottleneck_modules(self):
        return [self.center, self.transformer_bottleneck]

    def after_center(self, x4):
        return self.transformer_bottleneck(x4)


class TransformerPrototypeUNet(ResNet34UNetBaseline):
    model_variant = "tbn_d1_hnproto"

    def __init__(
        self,
        encoder_weights=None,
        transformer_bottleneck_layers=1,
        transformer_bottleneck_heads=8,
        transformer_bottleneck_dropout=0.1,
        prototype_bank_path="",
        prototype_attention_heads=8,
        prototype_attention_dropout=0.1,
    ):
        super().__init__(encoder_weights=encoder_weights)
        self.transformer_bottleneck = TransformerBottleneck(
            channels=512,
            num_layers=int(transformer_bottleneck_layers),
            num_heads=int(transformer_bottleneck_heads),
            dropout=float(transformer_bottleneck_dropout),
        )
        bank = load_prototype_bank(prototype_bank_path, map_location="cpu")
        self.prototype_attention = PrototypeCrossAttention(
            pos_prototypes=bank["pos_prototypes"],
            neg_prototypes=bank["neg_prototypes"],
            channels=512,
            num_heads=int(prototype_attention_heads),
            dropout=float(prototype_attention_dropout),
        )

    def bottleneck_modules(self):
        return [self.center, self.transformer_bottleneck, self.prototype_attention]

    def after_center(self, x4):
        x4 = self.transformer_bottleneck(x4)
        return self.prototype_attention(x4)


class SkipGateUNet(ResNet34UNetBaseline):
    model_variant = "skipgate_d4d3"

    def __init__(
        self,
        encoder_weights=None,
        skip_attention_levels=None,
        skip_attention_gamma_init=0.0,
    ):
        super().__init__(encoder_weights=encoder_weights)
        self.skip_attention_levels = set(skip_attention_levels or ["d4", "d3"])
        self.skip_gate_d4 = (
            SkipAttentionGate(skip_channels=256, gate_channels=512, gamma_init=skip_attention_gamma_init)
            if "d4" in self.skip_attention_levels
            else None
        )
        self.skip_gate_d3 = (
            SkipAttentionGate(skip_channels=128, gate_channels=256, gamma_init=skip_attention_gamma_init)
            if "d3" in self.skip_attention_levels
            else None
        )

    def decoder_path_modules(self):
        return [
            self.decoder4,
            self.decoder3,
            self.decoder2,
            self.decoder1,
            self.skip_gate_d4,
            self.skip_gate_d3,
            self.segmentation_head,
        ]

    def decode(self, x4, x3, x2, x1, x0, input_size):
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
        return d4, d3, d2, d1
