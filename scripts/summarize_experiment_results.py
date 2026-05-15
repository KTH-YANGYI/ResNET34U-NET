#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


VARIANT_LABELS = {
    "resnet34_unet_baseline": "M0 baseline",
    "tbn_d1": "M1 bottleneck transformer",
    "skipgate_d4": "M2 skip attention gate d4",
    "skipgate_d3": "M3 skip attention gate d3",
    "skipgate_d4d3": "M4 skip attention gate d4+d3",
    "selfattn_d4d3": "M5 decoder self-attention d4+d3",
    "tbn_d1_hnproto": "M6 prototype cross-attention",
}


RUN_NAME_LABELS = {
    "m0_baseline": "M0 full baseline",
    "t1_no_stage1": "T1 no Stage1",
    "t2_no_hard_normal": "T2 no hard normal mining",
    "t3_normal_fp_loss": "T3 normal FP loss",
    "m1_tbn_d1": "M1 bottleneck transformer",
    "m2_skip_d4": "M2 skip attention gate d4",
    "m3_skip_d3": "M3 skip attention gate d3",
    "m4_skip_d4d3": "M4 skip attention gate d4+d3",
    "m5_selfattn_d4d3": "M5 decoder self-attention d4+d3",
    "m6_hnproto": "M6 prototype cross-attention",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize 811 experiment results")
    parser.add_argument("--run-root", required=True, help="Experiment run root, relative to project root or absolute")
    parser.add_argument("--scope", default="", help="Experiment scope label")
    parser.add_argument("--baseline-profile", default="", help="Baseline profile label")
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value):
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def safe_int(value):
    if value is None or value == "":
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def infer_status(stage2_dir):
    if (stage2_dir / "val_metrics.json").exists() and (stage2_dir / "holdout" / "holdout_metrics.json").exists():
        return "完成"
    if (stage2_dir / "best_stage2.pt").exists():
        return "已训练，待评估"
    if stage2_dir.exists():
        return "运行中或未完成"
    return "未开始"


def summarize_run(run_dir):
    stage2_dir = run_dir / "stage2"
    cfg = load_json(stage2_dir / "resolved_config.json")
    val = load_json(stage2_dir / "val_metrics.json")
    holdout = load_json(stage2_dir / "holdout" / "holdout_metrics.json")

    variant = cfg.get("model_variant", "")
    label = next((value for key, value in RUN_NAME_LABELS.items() if key in run_dir.name), "")
    if not label:
        label = VARIANT_LABELS.get(variant, variant)
    seed = cfg.get("seed", "")

    return {
        "run": run_dir.name,
        "status": infer_status(stage2_dir),
        "variant": variant,
        "label": label,
        "seed": seed,
        "val_stage2_score": val.get("stage2_score", ""),
        "val_defect_dice": val.get("defect_dice", ""),
        "val_defect_iou": val.get("defect_iou", ""),
        "val_normal_fpr": val.get("normal_fpr", ""),
        "val_threshold": val.get("threshold", ""),
        "val_min_area": val.get("min_area", ""),
        "holdout_labeled_count": holdout.get("labeled_count", ""),
        "holdout_defect_dice": holdout.get("defect_dice", ""),
        "holdout_defect_iou": holdout.get("defect_iou", ""),
        "holdout_normal_fpr": holdout.get("normal_fpr", ""),
        "holdout_threshold": holdout.get("threshold", ""),
        "holdout_min_area": holdout.get("min_area", ""),
    }


def sort_key(row):
    try:
        score = float(row["val_stage2_score"])
    except (TypeError, ValueError):
        score = float("-inf")
    return (-score, row["run"])


def write_csv(path, rows):
    fieldnames = [
        "run",
        "status",
        "label",
        "variant",
        "seed",
        "val_stage2_score",
        "val_defect_dice",
        "val_defect_iou",
        "val_normal_fpr",
        "val_threshold",
        "val_min_area",
        "holdout_labeled_count",
        "holdout_defect_dice",
        "holdout_defect_iou",
        "holdout_normal_fpr",
        "holdout_threshold",
        "holdout_min_area",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows):
    lines = [
        "| 实验 | 状态 | 变体 | seed | val score | val Dice | val IoU | val normal FPR | val 后处理 | holdout Dice | holdout IoU | holdout normal FPR |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        postprocess = ""
        if row["val_threshold"] != "" or row["val_min_area"] != "":
            postprocess = f"thr={safe_float(row['val_threshold'])}, area={safe_int(row['val_min_area'])}"
        lines.append(
            "| {run} | {status} | {label} | {seed} | {score} | {dice} | {iou} | {fpr} | {post} | {hdice} | {hiou} | {hfpr} |".format(
                run=row["run"],
                status=row["status"],
                label=row["label"],
                seed=row["seed"],
                score=safe_float(row["val_stage2_score"]),
                dice=safe_float(row["val_defect_dice"]),
                iou=safe_float(row["val_defect_iou"]),
                fpr=safe_float(row["val_normal_fpr"]),
                post=postprocess,
                hdice=safe_float(row["holdout_defect_dice"]),
                hiou=safe_float(row["holdout_defect_iou"]),
                hfpr=safe_float(row["holdout_normal_fpr"]),
            )
        )
    return "\n".join(lines)


def write_markdown(path, run_root, rows, scope, baseline_profile):
    completed = [row for row in rows if row["val_stage2_score"] != ""]
    best = sorted(completed, key=sort_key)[0] if completed else None

    lines = [
        "# 811 fixed split 实验结果汇总",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 实验目录：`{run_root.relative_to(PROJECT_ROOT) if run_root.is_relative_to(PROJECT_ROOT) else run_root}`",
        f"- scope：`{scope or 'unknown'}`",
        f"- baseline profile：`{baseline_profile or 'unknown'}`",
        "",
        "## 当前结论",
        "",
    ]

    if best is None:
        lines.append("目前还没有完整的 validation 指标，先看各实验目录和日志。")
    else:
        lines.append(
            "当前 validation 排名第一的是 `{run}`（{label}，seed={seed}），"
            "val stage2_score={score}，val Dice={dice}，val normal FPR={fpr}。".format(
                run=best["run"],
                label=best["label"],
                seed=best["seed"],
                score=safe_float(best["val_stage2_score"]),
                dice=safe_float(best["val_defect_dice"]),
                fpr=safe_float(best["val_normal_fpr"]),
            )
        )
        lines.append("正式结论仍以所有计划内变体完成后的同表比较为准。")

    lines.extend(
        [
            "",
            "## 指标表",
            "",
            markdown_table(rows),
            "",
            "## 目录说明",
            "",
            "每个实验都有独立目录：`<run>/stage2/` 保存 checkpoint、history、resolved config、validation 指标；`<run>/stage2/holdout/` 保存 holdout/test 指标和逐图结果。",
            "权重文件不建议推到 GitHub；用于论文和排查的 CSV/JSON/log 可以保留。",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    run_root = resolve_path(args.run_root)
    if not run_root.exists():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")

    rows = []
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("811_"):
            continue
        rows.append(summarize_run(run_dir))

    rows = sorted(rows, key=sort_key)
    write_csv(run_root / "results_summary.csv", rows)
    write_markdown(run_root / "summary_zh.md", run_root, rows, args.scope, args.baseline_profile)

    print(f"wrote {run_root / 'results_summary.csv'}")
    print(f"wrote {run_root / 'summary_zh.md'}")


if __name__ == "__main__":
    main()
