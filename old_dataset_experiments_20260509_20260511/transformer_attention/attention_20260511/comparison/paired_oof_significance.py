import argparse
import csv
import json
import random
from pathlib import Path


DEFAULT_METRICS = [
    "dice",
    "iou",
    "pixel_f1",
    "component_recall_3px",
    "boundary_f1_3px",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison for OOF per-image metrics")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    return parser.parse_args()


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def row_key(row):
    return str(row.get("sample_id") or row.get("image_path") or row.get("mask_path"))


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    baseline_rows = {row_key(row): row for row in read_rows(args.baseline)}
    variant_rows = {row_key(row): row for row in read_rows(args.variant)}
    keys = sorted(set(baseline_rows) & set(variant_rows))
    if not keys:
        raise SystemExit("No overlapping per-image rows found.")

    output = {
        "baseline": args.baseline,
        "variant": args.variant,
        "n_paired_images": len(keys),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "metrics": {},
    }

    for metric in args.metrics:
        diffs = []
        for key in keys:
            base = to_float(baseline_rows[key].get(metric))
            current = to_float(variant_rows[key].get(metric))
            if base is not None and current is not None:
                diffs.append(current - base)

        observed = mean(diffs)
        boot = []
        for _ in range(args.n_bootstrap):
            sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
            boot.append(mean(sample))

        output["metrics"][metric] = {
            "n": len(diffs),
            "mean_delta": observed,
            "ci95_low": percentile(boot, 0.025),
            "ci95_high": percentile(boot, 0.975),
            "p_delta_le_0": sum(1 for value in boot if value <= 0.0) / len(boot),
            "p_delta_ge_0": sum(1 for value in boot if value >= 0.0) / len(boot),
        }

    normal_fp_delta = []
    for key in keys:
        base = int(float(baseline_rows[key].get("has_prediction", 0) or 0))
        current = int(float(variant_rows[key].get("has_prediction", 0) or 0))
        is_normal = int(float(baseline_rows[key].get("is_defect", 0) or 0)) == 0
        if is_normal:
            normal_fp_delta.append(current - base)

    output["normal_fp_delta_sum"] = sum(normal_fp_delta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "significance_report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Paired OOF Significance",
        "",
        f"- Baseline: `{args.baseline}`",
        f"- Variant: `{args.variant}`",
        f"- Paired images: {len(keys)}",
        f"- Bootstrap samples: {args.n_bootstrap}",
        "",
        "| metric | mean_delta | 95% CI | p(delta<=0) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, stats in output["metrics"].items():
        ci = f"{stats['ci95_low']:.6f} to {stats['ci95_high']:.6f}"
        lines.append(
            f"| {metric} | {stats['mean_delta']:.6f} | {ci} | {stats['p_delta_le_0']:.4f} |"
        )
    lines.append("")
    lines.append(f"- Normal-image false-positive delta sum: {output['normal_fp_delta_sum']}")

    (output_dir / "significance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"output_dir": str(output_dir), "n_paired_images": len(keys)})


if __name__ == "__main__":
    main()
