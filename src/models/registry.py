from torchvision.models import ResNet34_Weights

from src.parallel import unwrap_model
from src.models.resnet34_unet_baseline import ResNet34UNetBaseline
from src.models.variants.attention import (
    DecoderSelfAttentionUNet,
    SkipGateUNet,
    TransformerBottleneckUNet,
    TransformerPrototypeUNet,
)
from src.models.variants.auxiliary import AuxiliaryHeadUNet


MODEL_REGISTRY = {
    "resnet34_unet_baseline": ResNet34UNetBaseline,
    "tbn_d1": TransformerBottleneckUNet,
    "tbn_d1_hnproto": TransformerPrototypeUNet,
    "skipgate_d4d3": SkipGateUNet,
    "selfattn_d4d3": DecoderSelfAttentionUNet,
    "resnet34_unet_aux": AuxiliaryHeadUNet,
}

MODEL_ALIASES = {
    "canonical_resnet34_unet_pw12": "resnet34_unet_baseline",
    "resnet34_unet": "resnet34_unet_baseline",
    "resnet34_unet_pw12": "resnet34_unet_baseline",
    "baseline": "resnet34_unet_baseline",
    "resnet34_unet_tbn_d1": "tbn_d1",
    "resnet34_unet_tbn_d1_hnproto": "tbn_d1_hnproto",
    "resnet34_unet_skipgate_d4d3": "skipgate_d4d3",
    "resnet34_unet_selfattn_d4d3": "selfattn_d4d3",
    "decoder_selfattn_d4d3": "selfattn_d4d3",
}


def normalize_model_variant(name):
    key = str(name or "resnet34_unet_baseline").strip()
    return MODEL_ALIASES.get(key, key)


def infer_model_variant(options):
    variant = str(options.get("model_variant", "")).strip()
    if variant:
        return normalize_model_variant(variant)

    model_name = str(options.get("model_name", "")).strip()
    if model_name in MODEL_ALIASES:
        return normalize_model_variant(model_name)

    if bool(options.get("prototype_attention_enable", False)):
        return "tbn_d1_hnproto"
    if bool(options.get("transformer_bottleneck_enable", False)):
        return "tbn_d1"
    if bool(options.get("skip_attention_enable", False)):
        return "skipgate_d4d3"
    if bool(options.get("self_attention_enable", False)):
        return "selfattn_d4d3"
    if (
        bool(options.get("deep_supervision_enable", False))
        or bool(options.get("boundary_aux_enable", False))
        or bool(options.get("deep_supervision", False))
        or bool(options.get("boundary_aux", False))
    ):
        return "resnet34_unet_aux"
    return "resnet34_unet_baseline"


def variant_kwargs_from_config(cfg):
    variant = infer_model_variant(cfg)
    kwargs = {
        "transformer_bottleneck_layers": int(cfg.get("transformer_bottleneck_layers", 1)),
        "transformer_bottleneck_heads": int(cfg.get("transformer_bottleneck_heads", 8)),
        "transformer_bottleneck_dropout": float(cfg.get("transformer_bottleneck_dropout", 0.1)),
        "prototype_bank_path": cfg.get("prototype_bank_path", ""),
        "prototype_attention_heads": int(cfg.get("prototype_attention_heads", cfg.get("transformer_bottleneck_heads", 8))),
        "prototype_attention_dropout": float(cfg.get("prototype_attention_dropout", cfg.get("transformer_bottleneck_dropout", 0.1))),
        "skip_attention_levels": cfg.get("skip_attention_levels", ["d4", "d3"]),
        "skip_attention_gamma_init": float(cfg.get("skip_attention_gamma_init", 0.0)),
        "self_attention_levels": cfg.get("self_attention_levels", ["d4", "d3"]),
        "self_attention_heads": int(cfg.get("self_attention_heads", 4)),
        "self_attention_dropout": float(cfg.get("self_attention_dropout", 0.1)),
        "self_attention_sr_ratios": cfg.get("self_attention_sr_ratios", {"d4": 2, "d3": 4}),
        "self_attention_gamma_init": float(cfg.get("self_attention_gamma_init", 0.0)),
        "deep_supervision": bool(cfg.get("deep_supervision_enable", False)),
        "boundary_aux": bool(cfg.get("boundary_aux_enable", False)),
    }
    kwargs["model_variant"] = variant
    return kwargs


def prune_variant_kwargs(variant, kwargs):
    if variant == "resnet34_unet_baseline":
        return {}
    if variant == "tbn_d1":
        return {
            "transformer_bottleneck_layers": kwargs.get("transformer_bottleneck_layers", 1),
            "transformer_bottleneck_heads": kwargs.get("transformer_bottleneck_heads", 8),
            "transformer_bottleneck_dropout": kwargs.get("transformer_bottleneck_dropout", 0.1),
        }
    if variant == "tbn_d1_hnproto":
        return {
            "transformer_bottleneck_layers": kwargs.get("transformer_bottleneck_layers", 1),
            "transformer_bottleneck_heads": kwargs.get("transformer_bottleneck_heads", 8),
            "transformer_bottleneck_dropout": kwargs.get("transformer_bottleneck_dropout", 0.1),
            "prototype_bank_path": kwargs.get("prototype_bank_path", ""),
            "prototype_attention_heads": kwargs.get("prototype_attention_heads", 8),
            "prototype_attention_dropout": kwargs.get("prototype_attention_dropout", 0.1),
        }
    if variant == "skipgate_d4d3":
        return {
            "skip_attention_levels": kwargs.get("skip_attention_levels", ["d4", "d3"]),
            "skip_attention_gamma_init": kwargs.get("skip_attention_gamma_init", 0.0),
        }
    if variant == "selfattn_d4d3":
        return {
            "self_attention_levels": kwargs.get("self_attention_levels", ["d4", "d3"]),
            "self_attention_heads": kwargs.get("self_attention_heads", 4),
            "self_attention_dropout": kwargs.get("self_attention_dropout", 0.1),
            "self_attention_sr_ratios": kwargs.get("self_attention_sr_ratios", {"d4": 2, "d3": 4}),
            "self_attention_gamma_init": kwargs.get("self_attention_gamma_init", 0.0),
        }
    if variant == "resnet34_unet_aux":
        return {
            "deep_supervision": kwargs.get("deep_supervision", False),
            "boundary_aux": kwargs.get("boundary_aux", False),
        }
    return dict(kwargs)


def build_model(
    pretrained=True,
    allow_pretrained_fallback=True,
    model_variant=None,
    **variant_kwargs,
):
    options = dict(variant_kwargs)
    if model_variant is not None:
        options["model_variant"] = model_variant

    variant = infer_model_variant(options)
    if variant not in MODEL_REGISTRY:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_variant `{variant}`. Available variants: {choices}")

    model_cls = MODEL_REGISTRY[variant]
    model_kwargs = prune_variant_kwargs(variant, options)

    if not pretrained:
        model = model_cls(encoder_weights=None, **model_kwargs)
        model.model_variant = variant
        return model

    try:
        model = model_cls(encoder_weights=ResNet34_Weights.DEFAULT, **model_kwargs)
        model.model_variant = variant
        return model
    except Exception as exc:
        if not allow_pretrained_fallback:
            raise RuntimeError(
                "Pretrained encoder weights failed to load and "
                "allow_pretrained_fallback is disabled."
            ) from exc

        print("Warning: pretrained encoder weights failed; falling back to random initialization.")
        print(f"Detail: {exc}")
        model = model_cls(encoder_weights=None, **model_kwargs)
        model.model_variant = variant
        return model


def build_model_from_config(cfg):
    kwargs = variant_kwargs_from_config(cfg)
    return build_model(
        pretrained=bool(cfg.get("pretrained", False)),
        allow_pretrained_fallback=bool(cfg.get("allow_pretrained_fallback", True)),
        **kwargs,
    )


def collect_model_diagnostics(model):
    model = unwrap_model(model)
    diagnostics = {
        "model_variant": str(getattr(model, "model_variant", model.__class__.__name__)),
    }
    transformer = getattr(model, "transformer_bottleneck", None)
    if transformer is not None and hasattr(transformer, "gamma"):
        diagnostics["transformer_bottleneck_gamma"] = float(transformer.gamma.detach().cpu().item())

    prototype_attention = getattr(model, "prototype_attention", None)
    if prototype_attention is not None and hasattr(prototype_attention, "gamma"):
        diagnostics["prototype_attention_gamma"] = float(prototype_attention.gamma.detach().cpu().item())

    skip_gate_d4 = getattr(model, "skip_gate_d4", None)
    if skip_gate_d4 is not None and hasattr(skip_gate_d4, "gamma"):
        diagnostics["skip_gate_d4_gamma"] = float(skip_gate_d4.gamma.detach().cpu().item())

    skip_gate_d3 = getattr(model, "skip_gate_d3", None)
    if skip_gate_d3 is not None and hasattr(skip_gate_d3, "gamma"):
        diagnostics["skip_gate_d3_gamma"] = float(skip_gate_d3.gamma.detach().cpu().item())

    self_attention_d4 = getattr(model, "self_attention_d4", None)
    if self_attention_d4 is not None and hasattr(self_attention_d4, "gamma"):
        diagnostics["self_attention_d4_gamma"] = float(self_attention_d4.gamma.detach().cpu().item())

    self_attention_d3 = getattr(model, "self_attention_d3", None)
    if self_attention_d3 is not None and hasattr(self_attention_d3, "gamma"):
        diagnostics["self_attention_d3_gamma"] = float(self_attention_d3.gamma.detach().cpu().item())

    return diagnostics
