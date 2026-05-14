"""
Compatibility entrypoint for model construction.

New code should use `src.models.registry` directly. Existing training,
evaluation, and inference scripts can continue importing from `src.model`.
"""

from src.models import (
    MODEL_REGISTRY,
    ResNet34UNetBaseline,
    build_model,
    build_model_from_config,
    collect_model_diagnostics,
    infer_model_variant,
    normalize_model_variant,
)
from src.models.blocks import ConvBlock, DecoderBlock, make_aux_head


UNetResNet34 = ResNet34UNetBaseline


__all__ = [
    "MODEL_REGISTRY",
    "ConvBlock",
    "DecoderBlock",
    "ResNet34UNetBaseline",
    "UNetResNet34",
    "build_model",
    "build_model_from_config",
    "collect_model_diagnostics",
    "infer_model_variant",
    "make_aux_head",
    "normalize_model_variant",
]
