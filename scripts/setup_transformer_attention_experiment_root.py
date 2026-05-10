import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path


EXPERIMENTS = [
    {
        "dirname": "01_transformer_bottleneck",
        "title": "Transformer bottleneck",
        "config": "configs/transformer_a40_20260510/stage2_tbn_d1_a40.yaml",
        "build_prototypes": False,
        "purpose": "Add one low-resolution Transformer encoder block after the ResNet34-U-Net center block.",
    },
    {
        "dirname": "02_hard_negative_prototype_attention",
        "title": "Hard-negative prototype attention",
        "config": "configs/transformer_a40_20260510/stage2_hnproto_a40.yaml",
        "build_prototypes": True,
        "purpose": "Build fold-specific Stage1 positive/negative prototype banks and attend to them at the bottleneck.",
    },
    {
        "dirname": "03_decoder_skip_attention_gate",
        "title": "Decoder skip attention gate",
        "config": "configs/transformer_a40_20260510/stage2_skipgate_a40.yaml",
        "build_prototypes": False,
        "purpose": "Gate decoder d4/d3 skip features with additive Attention U-Net style gates.",
    },
]


SNAPSHOT_PATHS = [
    "src",
    "scripts",
    "configs",
    "manifests",
    "README.md",
    "requirements.txt",
    "DATASET_DESCRIPTION.md",
    "UNET_DATA_FLOW.md",
    "THESIS_METHODOLOGY.md",
    "EXPERIMENTS_20260509.md",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create source/result folders for the three A40 transformer experiments")
    parser.add_argument("--experiment-root", required=True)
    return parser.parse_args()


def git_output(args, cwd):
    return subprocess.check_output(["git", *args], cwd=cwd, universal_newlines=True).strip()


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def create_source_snapshot(repo_root, source_dir, commit):
    source_dir.mkdir(parents=True, exist_ok=True)
    if any(source_dir.iterdir()):
        raise RuntimeError(f"Source snapshot already exists and is not empty: {source_dir}")

    tar_path = source_dir.parent / "source_snapshot.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(tar_path), commit, *SNAPSHOT_PATHS],
        cwd=repo_root,
        check=True,
    )
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(source_dir)
    tar_path.unlink()
    write_text(source_dir / "SOURCE_COMMIT.txt", f"{commit}\n")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    experiment_root = Path(args.experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)
    commit = git_output(["rev-parse", "HEAD"], cwd=repo_root)

    root_runner = experiment_root / "run_all_3_experiments.sh"
    shutil.copy2(repo_root / "scripts" / "run_transformer_attention_experiment_set_4a40.sh", root_runner)
    root_runner.chmod(0o755)

    root_sbatch = experiment_root / "alvis_transformer_attention_4a40.sbatch"
    shutil.copy2(repo_root / "scripts" / "alvis_transformer_attention_4a40.sbatch", root_sbatch)

    for experiment in EXPERIMENTS:
        exp_dir = experiment_root / experiment["dirname"]
        source_dir = exp_dir / "source"
        (exp_dir / "results").mkdir(parents=True, exist_ok=True)
        (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
        create_source_snapshot(repo_root, source_dir, commit)
        write_text(
            exp_dir / "EXPERIMENT.md",
            "\n".join(
                [
                    f"# {experiment['title']}",
                    "",
                    f"Purpose: {experiment['purpose']}",
                    "",
                    f"- Source snapshot: `{source_dir}`",
                    f"- Source commit: `{commit}`",
                    f"- Config: `{source_dir / experiment['config']}`",
                    f"- Results: `{exp_dir / 'results'}`",
                    f"- Logs: `{exp_dir / 'logs'}`",
                    f"- Builds prototype banks: `{experiment['build_prototypes']}`",
                    "",
                ]
            ),
        )

    write_text(
        experiment_root / "RUNNING_STATUS.md",
        "\n".join(
            [
                "# Experiment Set Status",
                "",
                f"- Source commit: `{commit}`",
                "- Status: setup complete, Slurm job not submitted yet",
                "- GPU plan: 4 x A40, one fold per GPU",
                "",
            ]
        ),
    )
    print({"experiment_root": str(experiment_root), "source_commit": commit})


if __name__ == "__main__":
    main()
