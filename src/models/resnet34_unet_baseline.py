import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

from src.models.blocks import ConvBlock, DecoderBlock


class ResNet34UNetBaseline(nn.Module):
    model_variant = "resnet34_unet_baseline"

    def __init__(self, encoder_weights=None):
        super().__init__()
        self._encoder_trainable = True

        encoder = resnet34(weights=encoder_weights)
        self.encoder_stem = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
        )
        self.encoder_pool = encoder.maxpool
        self.encoder_layer1 = encoder.layer1
        self.encoder_layer2 = encoder.layer2
        self.encoder_layer3 = encoder.layer3
        self.encoder_layer4 = encoder.layer4

        self.center = ConvBlock(512, 512)
        self.decoder4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def encoder_modules(self):
        return [
            self.encoder_stem,
            self.encoder_layer1,
            self.encoder_layer2,
            self.encoder_layer3,
            self.encoder_layer4,
        ]

    def bottleneck_modules(self):
        return [self.center]

    def decoder_path_modules(self):
        return [
            self.decoder4,
            self.decoder3,
            self.decoder2,
            self.decoder1,
            self.segmentation_head,
        ]

    def decoder_modules(self):
        return self.bottleneck_modules() + self.decoder_path_modules()

    def encoder_parameters(self):
        for module in self.encoder_modules():
            yield from module.parameters()

    def decoder_parameters(self):
        for module in self.decoder_modules():
            if module is not None:
                yield from module.parameters()

    def set_encoder_trainable(self, trainable: bool):
        self._encoder_trainable = bool(trainable)
        for param in self.encoder_parameters():
            param.requires_grad = trainable

    def apply_encoder_freeze_mode(self):
        if self._encoder_trainable:
            return
        for module in self.encoder_modules():
            module.eval()

    def encode(self, x):
        x0 = self.encoder_stem(x)
        x1 = self.encoder_layer1(self.encoder_pool(x0))
        x2 = self.encoder_layer2(x1)
        x3 = self.encoder_layer3(x2)
        x4 = self.encoder_layer4(x3)
        return x0, x1, x2, x3, x4

    def after_center(self, x4):
        return x4

    def decode(self, x4, x3, x2, x1, x0, input_size):
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
        return d4, d3, d2, d1

    def make_output(self, logits, d4, d3, d2, d1, input_size):
        return logits

    def forward(self, x):
        input_size = x.shape[-2:]
        x0, x1, x2, x3, x4 = self.encode(x)
        x4 = self.center(x4)
        x4 = self.after_center(x4)
        d4, d3, d2, d1 = self.decode(x4, x3, x2, x1, x0, input_size)
        logits = self.segmentation_head(d1)
        return self.make_output(logits, d4, d3, d2, d1, input_size)
