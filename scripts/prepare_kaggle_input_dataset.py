"""
Prepare a Kaggle dataset folder from generated masks/checkpoints.

Kaggle notebooks cannot write into /kaggle/input directly. The practical flow is:
1. Save artifacts under /kaggle/working.
2. Copy them into one dataset folder with dataset-metadata.json.
3. Create a Kaggle dataset from that folder, or save the notebook output and
   add it as an input in the next notebook.
"""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Package mask artifacts as a Kaggle dataset folder.")
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default="/kaggle/working/unet-mask-input-dataset")
    parser.add_argument("--dataset-id", type=str, required=True, help="Example: yourname/fer2013-unet-region-masks")
    parser.add_argument("--title", type=str, default="FER2013 U-Net Region Masks")
    return parser.parse_args()


def copytree_contents(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    dataset_dir = Path(args.dataset_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"source-dir not found: {source_dir}")

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    copytree_contents(source_dir, dataset_dir / "unet_region_masks")
    metadata = {
        "id": args.dataset_id,
        "title": args.title,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(dataset_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"--> Prepared Kaggle dataset folder: {dataset_dir}")
    print("--> If Kaggle API credentials are available, create it with:")
    print(f"    kaggle datasets create -p {dataset_dir} -r zip")
    print("--> Otherwise, save this notebook version and add the output dataset as input next run.")


if __name__ == "__main__":
    main()

