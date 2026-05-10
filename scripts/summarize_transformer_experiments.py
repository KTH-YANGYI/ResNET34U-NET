import argparse
import json
from pathlib import Path


EXPERIMENTS = [
    ("01_transformer_bottleneck", "Transformer bottleneck"),
    ("02_hard_negative_prototype_attention", "Hard-negative prototype attention"),
    ("03_decoder_skip_attention_gate", "Decoder skip attention gate"),
]


OOF_KEYS = [
    "defect_dice",
    "defect_iou",
    "defect_image_recall",
    "normal_fp_count",
    "normal_fpr",
    "pixel_precision_defect_macro",
    "pixel_recall_defect_macro",
    "pixel_f1_defect_macro",
    "pixel_auprc_all_labeled",
    "component_recall_3px",
    "boundary_f1_3px",
    "threshold",
    "min_area",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize the three transformer/attention experiments")
    parser.add_argument("--experiment-root", required=True)
    return parser.parse_args()


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def metric_table(rows, keys):
    header = ["Experiment"] + keys
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in keys]) + " |",
    ]
    for name, metrics in rows:
        lines.append("| " + " | ".join([name] + [fmt(metrics.get(key) if metrics else None) for key in keys]) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    root = Path(args.experiment_root)
    rows_oof = []
    rows_holdout = []
    path_lines = []

    for dirname, label in EXPERIMENTS:
        exp_dir = root / dirname
        oof_path = exp_dir / "results" / "stage2" / "oof_global_postprocess.json"
        holdout_path = exp_dir / "results" / "stage2" / "holdout_ensemble" / "holdout_metrics.json"
        rows_oof.append((label, read_json(oof_path)))
        rows_holdout.append((label, read_json(holdout_path)))
        path_lines.extend(
            [
                f"- {label} source: `{exp_dir / 'source'}`",
                f"- {label} config: `{exp_dir / 'source' / 'configs' / 'transformer_a40_20260510'}`",
                f"- {label} logs: `{exp_dir / 'logs'}`",
                f"- {label} results: `{exp_dir / 'results'}`",
            ]
        )

    text = [
        "# Transformer/Attention A40 Experiment Report",
        "",
        "This report is generated after the three 4-fold A40 experiments finish. OOF uses a pooled validation post-processing search. Holdout uses the frozen OOF threshold/min_area and a 4-fold mean ensemble.",
        "",
        "## OOF Metrics",
        "",
        metric_table(rows_oof, OOF_KEYS),
        "",
        "## Frozen Holdout Metrics",
        "",
        metric_table(rows_holdout, OOF_KEYS),
        "",
        "## File Layout",
        "",
        "\n".join(path_lines),
        "",
        "## Notes",
        "",
        "- `stage2_score` remains a checkpoint/post-processing selection score, not the thesis primary metric.",
        "- The primary comparison should use OOF defect Dice together with normal FP count/FPR, pixel PR metrics, component recall, boundary F1, and frozen holdout behavior.",
        "- Broken holdout samples are included only in inference outputs, not quantitative holdout metrics.",
        "",
    ]
    report_path = root / "FINAL_EXPERIMENT_REPORT.md"
    report_path.write_text("\n".join(text), encoding="utf-8")
    print({"report_path": str(report_path)})


if __name__ == "__main__":
    main()
