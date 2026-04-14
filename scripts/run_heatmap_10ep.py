import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def find_latest_checkpoint(repo_root):
    pattern = str(repo_root / "outputs" / "checkpoints" / "resnet" / "*_best.pth")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No checkpoint found under outputs/checkpoints/resnet")
    files.sort(key=os.path.getmtime)
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description="Run heatmap debug at 0 epoch and after 10 epochs training.")
    parser.add_argument("--env", type=str, default="kaggle", choices=["local", "kaggle"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=12)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    # 1) Baseline heatmap with random/untrained weights (0 epoch).
    run_cmd(
        [
            py,
            "scripts/test_learned_landmark_kaggle.py",
            "--env",
            args.env,
            "--config",
            "resnet",
            "--split",
            args.split,
            "--num_samples",
            str(args.num_samples),
            "--save_per_keypoint",
            "--output_dir",
            "outputs/learned_landmark_test/debug_0ep",
        ],
        cwd=repo_root,
    )

    # 2) Train 10 epochs.
    run_cmd(
        [
            py,
            "scripts/train.py",
            "--config",
            "resnet_landmark_10ep",
            "--env",
            args.env,
        ],
        cwd=repo_root,
    )

    # 3) Heatmap after 10 epochs.
    ckpt = find_latest_checkpoint(repo_root)
    print(f"\nUsing checkpoint: {ckpt}")

    run_cmd(
        [
            py,
            "scripts/test_learned_landmark_kaggle.py",
            "--env",
            args.env,
            "--config",
            "resnet",
            "--split",
            args.split,
            "--num_samples",
            str(args.num_samples),
            "--save_per_keypoint",
            "--checkpoint",
            ckpt,
            "--output_dir",
            "outputs/learned_landmark_test/compare_10ep",
        ],
        cwd=repo_root,
    )

    print("\nDone.")
    print("- Baseline: outputs/learned_landmark_test/debug_0ep")
    print("- After 10 epochs: outputs/learned_landmark_test/compare_10ep")


if __name__ == "__main__":
    main()
