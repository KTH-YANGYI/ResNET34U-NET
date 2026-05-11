import json
from pathlib import Path


EXPERIMENTS = [
    "00_baseline_resnet34_unet_pw12",
    "01_tbn_d1",
    "02_tbn_d1_hnproto",
    "03_skipgate_d4d3",
]

FIXED_KEYS = [
    "seed",
    "image_size",
    "samples_path",
    "batch_size",
    "epochs",
    "encoder_lr",
    "decoder_lr",
    "weight_decay",
    "bce_weight",
    "dice_weight",
    "pos_weight",
    "normal_fp_loss_weight",
    "normal_fp_topk_ratio",
    "amp",
    "early_stop_patience",
    "early_stop_min_delta",
    "lr_factor",
    "lr_patience",
    "min_lr",
    "num_workers",
    "pretrained",
    "device",
    "auto_evaluate_after_train",
    "threshold",
    "train_eval_threshold",
    "train_eval_min_area",
    "use_imagenet_normalize",
    "augment_enable",
    "augment_hflip_p",
    "augment_vflip_p",
    "augment_rotate_deg",
    "augment_brightness",
    "augment_contrast",
    "augment_gamma",
    "augment_noise_std",
    "augment_blur_p",
    "target_normal_fpr",
    "lambda_fpr_penalty",
    "threshold_grid_start",
    "threshold_grid_end",
    "threshold_grid_step",
    "min_area_grid",
    "random_normal_k_factor",
    "use_hard_normal_replay",
    "stage2_hard_normal_ratio",
    "hard_normal_max_repeats_per_epoch",
    "hard_normal_warmup_epochs",
    "hard_normal_refresh_every",
    "hard_normal_pool_factor",
    "deep_supervision_enable",
    "boundary_aux_enable",
    "stage1_checkpoint_template",
]

EXPECTED_ARCHITECTURE = {
    "00_baseline_resnet34_unet_pw12": {
        "transformer_bottleneck_enable": False,
        "prototype_attention_enable": False,
        "skip_attention_enable": False,
        "stage1_load_strict": True,
    },
    "01_tbn_d1": {
        "transformer_bottleneck_enable": True,
        "transformer_bottleneck_layers": 1,
        "transformer_bottleneck_heads": 8,
        "transformer_bottleneck_dropout": 0.1,
        "prototype_attention_enable": False,
        "skip_attention_enable": False,
        "stage1_load_strict": False,
    },
    "02_tbn_d1_hnproto": {
        "transformer_bottleneck_enable": True,
        "transformer_bottleneck_layers": 1,
        "transformer_bottleneck_heads": 8,
        "transformer_bottleneck_dropout": 0.1,
        "prototype_attention_enable": True,
        "prototype_attention_heads": 8,
        "prototype_attention_dropout": 0.1,
        "prototype_pos_max": 128,
        "prototype_neg_max": 128,
        "prototype_l2_normalize": True,
        "prototype_batch_size": 64,
        "skip_attention_enable": False,
        "stage1_load_strict": False,
    },
    "03_skipgate_d4d3": {
        "transformer_bottleneck_enable": False,
        "prototype_attention_enable": False,
        "skip_attention_enable": True,
        "skip_attention_levels": ["d4", "d3"],
        "skip_attention_gamma_init": 0.0,
        "stage1_load_strict": False,
    },
}


def parse_scalar(raw):
    raw = raw.strip()
    lower = raw.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if raw.startswith("[") and raw.endswith("]"):
        return [parse_scalar(item.strip()) for item in raw[1:-1].split(",") if item.strip()]
    try:
        if any(char in raw for char in [".", "e", "E"]):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw.strip("\"'")


def load_simple_yaml(path):
    data = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or ":" not in text:
            continue
        key, raw = text.split(":", 1)
        key = key.strip()
        raw = raw.strip()
        data[key] = "" if raw == "" else parse_scalar(raw)
    return data


def fmt(value):
    return json.dumps(value, ensure_ascii=False)


def main():
    comparison_dir = Path(__file__).resolve().parent
    root = comparison_dir.parent
    configs = {
        name: load_simple_yaml(root / name / "config.yaml")
        for name in EXPERIMENTS
    }

    baseline = configs[EXPERIMENTS[0]]
    failures = []
    fixed_rows = []

    for key in FIXED_KEYS:
        baseline_value = baseline.get(key)
        row = {"key": key, "baseline": baseline_value, "matches": True, "values": {}}
        for name in EXPERIMENTS:
            value = configs[name].get(key)
            row["values"][name] = value
            if value != baseline_value:
                row["matches"] = False
                failures.append(
                    {
                        "type": "fixed_key_mismatch",
                        "key": key,
                        "baseline": baseline_value,
                        "experiment": name,
                        "value": value,
                    }
                )
        fixed_rows.append(row)

    architecture_rows = []
    for name, expected in EXPECTED_ARCHITECTURE.items():
        cfg = configs[name]
        for key, expected_value in expected.items():
            actual = cfg.get(key)
            ok = actual == expected_value
            architecture_rows.append(
                {
                    "experiment": name,
                    "key": key,
                    "expected": expected_value,
                    "actual": actual,
                    "ok": ok,
                }
            )
            if not ok:
                failures.append(
                    {
                        "type": "architecture_key_mismatch",
                        "experiment": name,
                        "key": key,
                        "expected": expected_value,
                        "actual": actual,
                    }
                )

    for name in EXPERIMENTS:
        cfg = configs[name]
        expected_fragment = f"experiments/attention_20260511/{name}/results"
        for key in ["save_dir_template", "global_postprocess_path", "prototype_bank_path_template"]:
            if key in cfg and expected_fragment not in str(cfg[key]):
                failures.append(
                    {
                        "type": "path_not_local_to_experiment",
                        "experiment": name,
                        "key": key,
                        "value": cfg[key],
                        "expected_fragment": expected_fragment,
                    }
                )

    result = {
        "ok": len(failures) == 0,
        "experiments": EXPERIMENTS,
        "fixed_keys": FIXED_KEYS,
        "fixed_rows": fixed_rows,
        "architecture_rows": architecture_rows,
        "failures": failures,
    }

    output_dir = comparison_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fairness_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Fairness Check",
        "",
        f"Status: {'PASS' if result['ok'] else 'FAIL'}",
        "",
        "## Fixed Settings",
        "",
        "| key | value |",
        "| --- | --- |",
    ]
    for row in fixed_rows:
        marker = "OK" if row["matches"] else "MISMATCH"
        lines.append(f"| {row['key']} | {marker}: {fmt(row['baseline'])} |")

    lines.extend(
        [
            "",
            "## Allowed Architecture Differences",
            "",
            "| experiment | key | expected | actual |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in architecture_rows:
        marker = "OK" if row["ok"] else "MISMATCH"
        lines.append(
            f"| {row['experiment']} | {row['key']} | {fmt(row['expected'])} | {marker}: {fmt(row['actual'])} |"
        )

    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure['type']}`: {fmt(failure)}")

    (output_dir / "FAIRNESS_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"ok": result["ok"], "failure_count": len(failures), "output_dir": str(output_dir)}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
