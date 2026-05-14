import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = [
    ("00_baseline_resnet34_unet_pw12", "Baseline ResNet34-U-Net pw12"),
    ("01_tbn_d1", "Transformer bottleneck d1"),
    ("02_tbn_d1_hnproto", "TBN d1 + hard-negative prototype attention"),
    ("03_skipgate_d4d3", "Decoder skip attention gate d4+d3"),
]

METRIC_KEYS = [
    "defect_dice",
    "defect_iou",
    "defect_image_recall",
    "normal_fp_count",
    "normal_fpr",
    "normal_fp_pixel_sum",
    "normal_largest_fp_area_max",
    "pixel_precision_defect_macro",
    "pixel_recall_defect_macro",
    "pixel_f1_defect_macro",
    "pixel_auprc_all_labeled",
    "component_recall_3px",
    "component_precision_3px",
    "component_f1_3px",
    "boundary_f1_3px",
    "threshold",
    "min_area",
]

DELTA_KEYS = [
    "defect_dice",
    "defect_iou",
    "defect_image_recall",
    "normal_fp_count",
    "normal_fpr",
    "pixel_f1_defect_macro",
    "pixel_auprc_all_labeled",
    "component_recall_3px",
    "boundary_f1_3px",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize the formal attention experiment set")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value(metrics, key):
    if not metrics:
        return None
    return metrics.get(key)


def numeric(metrics, key):
    raw = value(metrics, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def fmt(raw):
    if raw is None:
        return "NA"
    if isinstance(raw, int):
        return str(raw)
    try:
        return f"{float(raw):.6f}"
    except Exception:
        return str(raw)


def metric_table(rows):
    header = ["experiment"] + METRIC_KEYS
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in METRIC_KEYS]) + " |",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join([row["label"]] + [fmt(value(metrics, key)) for key in METRIC_KEYS])
            + " |"
        )
    return "\n".join(lines)


def delta_table(rows, baseline_metrics, tbn_metrics):
    header = ["experiment", "comparison"] + [f"delta_{key}" for key in DELTA_KEYS]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", "---"] + ["---:" for _ in DELTA_KEYS]) + " |",
    ]
    csv_rows = []

    for row in rows:
        name = row["name"]
        metrics = row["metrics"]
        if name == "00_baseline_resnet34_unet_pw12":
            continue

        comparisons = [("vs_baseline", baseline_metrics)]
        if name == "02_tbn_d1_hnproto":
            comparisons.append(("vs_01_tbn_d1", tbn_metrics))

        for comparison_name, reference in comparisons:
            out = {"experiment": name, "comparison": comparison_name}
            for key in DELTA_KEYS:
                current = numeric(metrics, key)
                base = numeric(reference, key)
                out[f"delta_{key}"] = None if current is None or base is None else current - base
            csv_rows.append(out)
            lines.append(
                "| "
                + " | ".join(
                    [name, comparison_name]
                    + [fmt(out[f"delta_{key}"]) for key in DELTA_KEYS]
                )
                + " |"
            )

    return "\n".join(lines), csv_rows


def load_rows(root, subpath):
    rows = []
    for name, label in EXPERIMENTS:
        path = root / name / subpath
        rows.append(
            {
                "name": name,
                "label": label,
                "path": str(path),
                "metrics": read_json(path),
            }
        )
    return rows


def rows_for_csv(rows, source):
    output = []
    for row in rows:
        metrics = row["metrics"] or {}
        csv_row = {
            "source": source,
            "experiment": row["name"],
            "label": row["label"],
            "path": row["path"],
            "available": bool(row["metrics"]),
        }
        for key in METRIC_KEYS:
            csv_row[key] = metrics.get(key)
        output.append(csv_row)
    return output


def main():
    args = parse_args()
    root = Path(args.experiment_root)
    report_dir = Path(args.output_dir) if args.output_dir else root / "comparison" / "results"
    report_dir.mkdir(parents=True, exist_ok=True)

    oof_rows = load_rows(root, Path("results/stage2/oof_global_postprocess.json"))
    holdout_rows = load_rows(root, Path("results/stage2/holdout_ensemble/holdout_metrics.json"))

    baseline_metrics = oof_rows[0]["metrics"] or {}
    tbn_metrics = oof_rows[1]["metrics"] or {}
    deltas_markdown, delta_rows = delta_table(oof_rows, baseline_metrics, tbn_metrics)

    metric_fieldnames = ["source", "experiment", "label", "path", "available"] + METRIC_KEYS
    write_csv(report_dir / "attention_oof_metrics.csv", rows_for_csv(oof_rows, "oof"), metric_fieldnames)
    write_csv(report_dir / "attention_holdout_metrics.csv", rows_for_csv(holdout_rows, "holdout"), metric_fieldnames)
    write_csv(
        report_dir / "attention_oof_deltas.csv",
        delta_rows,
        ["experiment", "comparison"] + [f"delta_{key}" for key in DELTA_KEYS],
    )

    report = [
        "# Attention Experiment Summary",
        "",
        "OOF results use the pooled validation threshold/min-area search. Holdout results, when available, must use the frozen OOF post-processing setting and must not tune on holdout.",
        "",
        "## OOF Metrics",
        "",
        metric_table(oof_rows),
        "",
        "## OOF Deltas",
        "",
        deltas_markdown,
        "",
        "## Frozen Holdout Metrics",
        "",
        metric_table(holdout_rows),
        "",
        "## Notes",
        "",
        "- `stage2_score` is intentionally excluded because it is a checkpoint/post-processing selection score, not a thesis endpoint.",
        "- The strictest claim should require improved or non-degraded Dice/IoU while keeping normal false positives at baseline level.",
        "- For `02_tbn_d1_hnproto`, the `vs_01_tbn_d1` row isolates the added value of hard-normal prototype attention beyond the transformer bottleneck.",
        "",
    ]
    (report_dir / "ATTENTION_EXPERIMENT_SUMMARY.md").write_text("\n".join(report), encoding="utf-8")
    (report_dir / "attention_summary_manifest.json").write_text(
        json.dumps(
            {
                "experiment_root": str(root),
                "oof_rows": oof_rows,
                "holdout_rows": holdout_rows,
                "delta_rows": delta_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print({"summary_dir": str(report_dir)})


if __name__ == "__main__":
    main()
