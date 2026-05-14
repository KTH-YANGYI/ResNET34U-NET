import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import make_aux_head
from src.models.resnet34_unet_baseline import ResNet34UNetBaseline


class AuxiliaryHeadUNet(ResNet34UNetBaseline):
    model_variant = "resnet34_unet_aux"

    def __init__(
        self,
        encoder_weights=None,
        deep_supervision=False,
        boundary_aux=False,
    ):
        super().__init__(encoder_weights=encoder_weights)
        self.deep_supervision = bool(deep_supervision)
        self.boundary_aux = bool(boundary_aux)

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

        self.boundary_head = make_aux_head(64, mid_channels=16) if self.boundary_aux else None

    def decoder_path_modules(self):
        modules = super().decoder_path_modules()
        modules.append(self.deep_supervision_heads)
        if self.boundary_head is not None:
            modules.append(self.boundary_head)
        return modules

    def make_output(self, logits, d4, d3, d2, d1, input_size):
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
