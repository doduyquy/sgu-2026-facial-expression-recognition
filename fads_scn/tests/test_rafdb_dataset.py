from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fads_scn.data.dataset import (
    RAFDBDataset,
    build_dataloaders,
    find_rafdb_root,
    resolve_rafdb_root,
)


def _write_rafdb_fixture(root: Path):
    train_dir = root / "RAF-DB DATASET" / "DATASET" / "train"
    test_dir = root / "RAF-DB DATASET" / "DATASET" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    dataset_root = train_dir.parent

    train_rows = []
    test_rows = []
    for raw_label in range(1, 8):
        train_class_dir = train_dir / str(raw_label)
        test_class_dir = test_dir / str(raw_label)
        train_class_dir.mkdir()
        test_class_dir.mkdir()
        for sample_index in range(5):
            stem = f"train_{raw_label:02d}_{sample_index:02d}"
            image = Image.new("RGB", (100, 100), color=(raw_label * 20, sample_index * 20, 100))
            image.save(train_class_dir / f"{stem}_aligned.jpg")
            train_rows.append({"image": f"{stem}.jpg", "label": raw_label})

        for sample_index in range(2):
            stem = f"test_{raw_label:02d}_{sample_index:02d}"
            image = Image.new("RGB", (100, 100), color=(raw_label * 20, sample_index * 20, 100))
            image.save(test_class_dir / f"{stem}_aligned.jpg")
            test_rows.append({"image": f"{stem}.jpg", "label": raw_label})

    pd.DataFrame(train_rows).to_csv(dataset_root / "train_labels.csv", index=False)
    pd.DataFrame(test_rows).to_csv(dataset_root / "test_labels.csv", index=False)
    return dataset_root


def _rafdb_config(data_path: Path):
    return {
        "seed": {"random_seed": 42},
        "model": {"in_channels": 3},
        "data": {
            "dataset": "rafdb",
            "data_path": str(data_path),
            "label_encoding": "auto",
            "val_ratio": 0.2,
            "split_seed": 42,
            "batch_size": 4,
            "num_workers": 0,
            "input_size": 96,
            "color_mode": "rgb",
            "normalization": "imagenet",
            "augmentation_profile": "rafdb",
            "use_random_erasing": True,
            "erasing_prob": 0.2,
            "use_mixup": False,
        },
    }


def test_rafdb_official_labels_are_mapped_to_model_class_order(tmp_path):
    dataset_root = _write_rafdb_fixture(tmp_path)
    dataset = RAFDBDataset(
        dataset_root,
        split="test",
        val_ratio=0.2,
        split_seed=42,
        label_encoding="auto",
    )

    # RAF raw: surprise, fear, disgust, happy, sad, angry, neutral.
    # Model:   angry, disgust, fear, happy, sad, surprise, neutral.
    assert dataset.labels.tolist() == [5, 5, 2, 2, 1, 1, 3, 3, 4, 4, 0, 0, 6, 6]


def test_rafdb_loader_uses_stratified_train_val_and_rgb_transforms(tmp_path):
    dataset_root = _write_rafdb_fixture(tmp_path)
    cfg = _rafdb_config(dataset_root)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    assert len(train_loader.dataset) == 28
    assert len(val_loader.dataset) == 7
    assert len(test_loader.dataset) == 14
    assert train_loader.dataset.get_class_counts().tolist() == [4] * 7
    assert val_loader.dataset.get_class_counts().tolist() == [1] * 7

    train_paths = set(train_loader.dataset.image_paths)
    val_paths = set(val_loader.dataset.image_paths)
    assert train_paths.isdisjoint(val_paths)

    images, labels, indices = next(iter(train_loader))
    assert images.shape == (4, 3, 96, 96)
    assert labels.shape == (4,)
    assert indices.shape == (4,)
    assert np.isfinite(images.numpy()).all()


def test_rafdb_root_resolves_screenshot_directory_layout(tmp_path):
    dataset_root = _write_rafdb_fixture(tmp_path)
    assert resolve_rafdb_root(tmp_path) == dataset_root
    assert find_rafdb_root(tmp_path) == dataset_root
