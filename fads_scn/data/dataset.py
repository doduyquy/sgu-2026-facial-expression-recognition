import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split


EMOTION_NAMES = [
    "angry",      # 0
    "disgust",    # 1
    "fear",       # 2
    "happy",      # 3
    "sad",        # 4
    "surprise",   # 5
    "neutral",    # 6
]

# RAF-DB's official Basic subset uses this one-based label order. The model and
# evaluator use EMOTION_NAMES above, so RAF labels must be remapped explicitly.
RAFDB_LABEL_ORDER = [
    "surprise",  # 1
    "fear",      # 2
    "disgust",   # 3
    "happy",     # 4
    "sad",       # 5
    "angry",     # 6
    "neutral",   # 7
]

EMOTION_ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
    "neutrality": "neutral",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def build_transforms(
    split: str = "train",
    input_size: int = 48,
    use_random_erasing: bool = True,
    erasing_prob: float = 0.3,
    in_channels: int = 1,
    normalization: str = "symmetric",
    color_mode: str = "grayscale",
    augmentation_profile: str = "fer2013",
):
    """
    Build data transformation pipeline for pure image FER.
    Train: Flip + Affine + ToTensor + Normalize + RandomErasing
    Val/Test: ToTensor + Normalize
    """
    if input_size < 16:
        raise ValueError("input_size must be at least 16")
    if in_channels not in (1, 3):
        raise ValueError("FER2013 supports in_channels=1 or 3")
    if normalization not in ("symmetric", "imagenet"):
        raise ValueError("normalization must be symmetric or imagenet")
    if normalization == "imagenet" and in_channels != 3:
        raise ValueError("imagenet normalization requires 3 channels")
    if color_mode not in ("grayscale", "rgb"):
        raise ValueError("color_mode must be grayscale or rgb")
    if color_mode == "rgb" and in_channels != 3:
        raise ValueError("rgb color_mode requires in_channels=3")
    if augmentation_profile not in ("fer2013", "rafdb"):
        raise ValueError("augmentation_profile must be fer2013 or rafdb")

    # Defaults intentionally preserve the prior 48x48 grayscale recipe exactly.
    transform_list = []
    if color_mode == "rgb":
        transform_list.append(T.Lambda(_convert_to_rgb))
    elif in_channels == 3:
        # FER2013 is grayscale: replicate its one channel; this does not invent colour.
        transform_list.append(T.Grayscale(num_output_channels=3))

    if split == "train" and augmentation_profile == "rafdb":
        # RAF-DB aligned faces are about 100x100. Use a restrained crop and
        # photometric jitter so expression geometry is not destroyed.
        resize_size = input_size + max(4, input_size // 12)
        transform_list.extend([
            T.Resize((resize_size, resize_size), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomResizedCrop(
                input_size,
                scale=(0.90, 1.0),
                ratio=(0.95, 1.05),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            ], p=0.30),
            T.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.96, 1.04)),
            T.ToTensor(),
        ])
    elif split == "train":
        if input_size != 48:
            transform_list.append(T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR))
        transform_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.92, 1.08)),
            T.ToTensor(),
        ])
    else:
        if input_size != 48:
            transform_list.append(T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR))
        transform_list.append(T.ToTensor())

    if normalization == "imagenet":
        transform_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        transform_list.append(T.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels))

    if split == "train":
        if use_random_erasing:
            transform_list.append(
                T.RandomErasing(p=erasing_prob, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0.0)
            )
        return T.Compose(transform_list)
    return T.Compose(transform_list)


def _normalized_column_name(column) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")


def _looks_like_image_reference(value) -> bool:
    value = str(value).strip().replace("\\", "/").lower()
    suffix = Path(value).suffix
    name = Path(value).name
    return suffix in IMAGE_EXTENSIONS or name.startswith(("train_", "test_"))


def _load_rafdb_label_table(csv_path: Path):
    """Read common RAF-DB CSV variants, including headerless files."""
    label_names = ("label", "labels", "emotion", "class", "target", "category")
    image_names = (
        "image", "images", "image_name", "imagename", "filename",
        "file_name", "file", "path", "image_path", "name",
    )

    dataframe = pd.read_csv(csv_path)
    normalized = {_normalized_column_name(col): col for col in dataframe.columns}
    has_known_label = any(name in normalized for name in label_names)
    if not has_known_label:
        # pandas treats the first sample as a header for a headerless CSV.
        dataframe = pd.read_csv(csv_path, header=None)

    normalized = {_normalized_column_name(col): col for col in dataframe.columns}
    label_column = next((normalized[name] for name in label_names if name in normalized), None)
    image_column = next((normalized[name] for name in image_names if name in normalized), None)

    if label_column is None:
        numeric_candidates = []
        emotion_candidates = []
        for column in dataframe.columns:
            values = dataframe[column]
            numeric_ratio = pd.to_numeric(values, errors="coerce").notna().mean()
            emotion_ratio = values.astype(str).str.strip().str.lower().isin(EMOTION_ALIASES).mean()
            numeric_candidates.append((numeric_ratio, column))
            emotion_candidates.append((emotion_ratio, column))
        emotion_score, emotion_column = max(emotion_candidates, default=(0.0, None))
        numeric_score, numeric_column = max(numeric_candidates, default=(0.0, None))
        label_column = emotion_column if emotion_score >= 0.8 else numeric_column
        if label_column is None or max(emotion_score, numeric_score) < 0.8:
            raise ValueError(f"Could not identify a label column in {csv_path}")

    if image_column is None:
        image_scores = []
        for column in dataframe.columns:
            if column == label_column:
                continue
            score = dataframe[column].map(_looks_like_image_reference).mean()
            image_scores.append((score, column))
        image_score, candidate = max(image_scores, default=(0.0, None))
        if image_score >= 0.8:
            image_column = candidate

    dataframe = dataframe.dropna(subset=[label_column]).reset_index(drop=True)
    return dataframe, image_column, label_column


def _map_rafdb_labels(raw_labels, label_encoding: str) -> np.ndarray:
    label_encoding = label_encoding.lower()
    values = list(raw_labels)
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce")
    all_numeric = numeric_values.notna().all()

    if not all_numeric:
        mapped = []
        for value in values:
            name = EMOTION_ALIASES.get(str(value).strip().lower())
            if name is None:
                raise ValueError(f"Unknown RAF-DB emotion label: {value!r}")
            mapped.append(EMOTION_NAMES.index(name))
        return np.asarray(mapped, dtype=np.int64)

    numeric = numeric_values.astype(np.int64).to_numpy()
    if label_encoding == "auto":
        if numeric.min() >= 1 and numeric.max() <= 7:
            label_encoding = "rafdb_1based"
        elif numeric.min() >= 0 and numeric.max() <= 6:
            label_encoding = "rafdb_0based"
        else:
            raise ValueError(f"Cannot infer RAF-DB label encoding from range {numeric.min()}..{numeric.max()}")

    if label_encoding in ("rafdb", "rafdb_1based"):
        raw_indices = numeric - 1
        order = RAFDB_LABEL_ORDER
    elif label_encoding == "rafdb_0based":
        raw_indices = numeric
        order = RAFDB_LABEL_ORDER
    elif label_encoding == "internal_0based":
        raw_indices = numeric
        order = EMOTION_NAMES
    else:
        raise ValueError(
            "label_encoding must be auto, rafdb_1based, rafdb_0based, or internal_0based"
        )

    if raw_indices.min() < 0 or raw_indices.max() >= len(order):
        raise ValueError(f"Labels are outside the valid range for {label_encoding}")
    return np.asarray([EMOTION_NAMES.index(order[index]) for index in raw_indices], dtype=np.int64)


def _resolve_rafdb_layout(data_path):
    """Return (label_root, image_root), which may differ on Kaggle mounts."""
    root = Path(data_path)
    image_candidates = [
        root,
        root / "DATASET",
        root / "RAF-DB DATASET" / "DATASET",
        root / "RAF-DB_DATASET" / "DATASET",
    ]
    seen = set()
    for image_root in image_candidates:
        image_root_key = str(image_root)
        if image_root_key in seen:
            continue
        seen.add(image_root_key)
        if not (image_root / "train").is_dir() or not (image_root / "test").is_dir():
            continue

        label_candidates = [image_root, root, image_root.parent]
        for label_root in label_candidates:
            if (
                (label_root / "train_labels.csv").is_file()
                and (label_root / "test_labels.csv").is_file()
            ):
                return label_root, image_root
    raise FileNotFoundError(
        "RAF-DB layout requires train/ and test/ image folders plus "
        "train_labels.csv and test_labels.csv (the CSV files may be one level above DATASET). "
        f"Checked under: {root}"
    )


def resolve_rafdb_root(data_path) -> Path:
    """Resolve the stable root for either compact or split RAF-DB layouts."""
    label_root, image_root = _resolve_rafdb_layout(data_path)
    return label_root if label_root != image_root else image_root


def find_rafdb_root(search_root) -> Path:
    """Find a RAF-DB root below a directory, following Kaggle input symlinks."""
    search_root = Path(search_root)
    if not search_root.exists():
        raise FileNotFoundError(f"Search root does not exist: {search_root}")

    # pathlib.rglob does not recurse through directory symlinks on all Python
    # versions. Kaggle inputs may be exposed that way, so use os.walk with
    # followlinks=True and identify the dataset from its structure, not slug.
    visited = set()
    for current_dir, dir_names, file_names in os.walk(search_root, followlinks=True):
        real_dir = os.path.realpath(current_dir)
        if real_dir in visited:
            dir_names[:] = []
            continue
        visited.add(real_dir)

        lower_dirs = {name.lower() for name in dir_names}
        lower_files = {name.lower() for name in file_names}
        has_image_dirs = {"train", "test"}.issubset(lower_dirs)
        has_label_csvs = {"train_labels.csv", "test_labels.csv"}.issubset(lower_files)
        if has_image_dirs or has_label_csvs:
            try:
                return resolve_rafdb_root(current_dir)
            except FileNotFoundError:
                continue

    mounted_inputs = []
    try:
        mounted_inputs = sorted(path.name for path in search_root.iterdir())
    except OSError:
        pass
    mounted_text = ", ".join(mounted_inputs[:20]) if mounted_inputs else "<none>"
    raise FileNotFoundError(
        f"Could not find RAF-DB below {search_root}. Mounted inputs: {mounted_text}. "
        "Expected a folder containing train/, test/, train_labels.csv and test_labels.csv."
    )


class PureImageFER2013(Dataset):
    """
    Pure Image-Based FER2013 Dataset.
    Only takes raw 48x48 pixel values from CSV.
    Zero dependency on bounding boxes, landmarks, or .npz files.
    """

    def __init__(self, data_path: str, split: str = "train", transform=None):
        super().__init__()
        self.split = split
        self.transform = transform

        # Resolve CSV path
        csv_candidates = [
            Path(data_path) / f"{split}.csv",
            Path(data_path) / f"fer13-split/{split}.csv",
            Path("dataset/fer13-split") / f"{split}.csv",
        ]
        csv_file = None
        for candidate in csv_candidates:
            if candidate.exists():
                csv_file = candidate
                break

        if csv_file is None:
            # Fallback to direct path
            csv_file = Path(data_path) / f"{split}.csv"

        df = pd.read_csv(csv_file, usecols=[0, 1])
        # Vectorized parsing into numpy array of uint8 images
        self.labels = df.iloc[:, 0].to_numpy(dtype=np.int64)
        raw_pixels = df.iloc[:, 1].tolist()
        
        # Pre-parse pixels to (N, 48, 48) uint8 array for ultra-fast loading
        parsed_imgs = []
        for p_str in raw_pixels:
            arr = np.fromstring(p_str, sep=' ', dtype=np.uint8).reshape(48, 48)
            parsed_imgs.append(arr)
        self.images = np.stack(parsed_imgs, axis=0)  # [N, 48, 48]

        # Mutable labels for SCN dynamic relabeling
        self.relabelled_count = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        img_arr = self.images[index]
        label = int(self.labels[index])

        img = Image.fromarray(img_arr)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, index

    def update_label(self, index: int, new_label: int):
        """Update label dynamically for SCN relabeling."""
        if 0 <= index < len(self.labels) and self.labels[index] != new_label:
            self.labels[index] = new_label
            self.relabelled_count += 1

    def get_class_counts(self):
        """Return counts per class for computing class weights."""
        counts = np.bincount(self.labels, minlength=7)
        return counts


class RAFDBDataset(Dataset):
    """RAF-DB Basic dataset backed by image folders and split label CSV files."""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform=None,
        val_ratio: float = 0.1,
        split_seed: int = 42,
        label_encoding: str = "rafdb_1based",
    ):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError("RAF-DB split must be train, val, or test")
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("RAF-DB val_ratio must be in (0, 1)")

        self.split = split
        self.transform = transform
        self.root = resolve_rafdb_root(data_path)
        label_root, image_root = _resolve_rafdb_layout(self.root)
        source_split = "test" if split == "test" else "train"
        image_dir = image_root / source_split
        csv_path = label_root / f"{source_split}_labels.csv"

        dataframe, image_column, label_column = _load_rafdb_label_table(csv_path)
        labels = _map_rafdb_labels(dataframe[label_column].tolist(), label_encoding)
        image_paths = sorted(
            path for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No image files found in {image_dir}")

        if image_column is None:
            if len(image_paths) != len(labels):
                raise ValueError(
                    f"{csv_path.name} has {len(labels)} labels but {image_dir} has "
                    f"{len(image_paths)} images; include a filename column to match them safely"
                )
            matched_paths = image_paths
        else:
            lookup = self._build_image_lookup(image_paths, image_dir)
            matched_paths = []
            missing = []
            for reference in dataframe[image_column].tolist():
                path = self._resolve_image_reference(reference, image_dir, lookup)
                if path is None:
                    missing.append(str(reference))
                else:
                    matched_paths.append(path)
            if missing:
                examples = ", ".join(missing[:5])
                raise FileNotFoundError(
                    f"Could not match {len(missing)} CSV image names in {image_dir}. Examples: {examples}"
                )

        # The Kaggle layout also stores images under train/1..7 and test/1..7.
        # When present, use those folders as an independent label-integrity check.
        folder_labels = []
        for path in matched_paths:
            try:
                folder_label = int(path.parent.name)
            except ValueError:
                folder_labels = []
                break
            if not 1 <= folder_label <= 7:
                folder_labels = []
                break
            folder_labels.append(folder_label)
        if folder_labels:
            folder_mapped = _map_rafdb_labels(folder_labels, "rafdb_1based")
            if not np.array_equal(folder_mapped, labels):
                raise ValueError(
                    "CSV labels disagree with RAF-DB class folders 1..7. "
                    "Check data.label_encoding before training."
                )

        matched_paths = np.asarray(matched_paths, dtype=object)
        if source_split == "train":
            indices = np.arange(len(labels))
            train_indices, val_indices = train_test_split(
                indices,
                test_size=val_ratio,
                random_state=int(split_seed),
                shuffle=True,
                stratify=labels,
            )
            selected_indices = train_indices if split == "train" else val_indices
            selected_indices = np.sort(selected_indices)
            matched_paths = matched_paths[selected_indices]
            labels = labels[selected_indices]

        self.image_paths = [Path(path) for path in matched_paths.tolist()]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.relabelled_count = 0

    @staticmethod
    def _build_image_lookup(image_paths, image_dir):
        lookup = {}
        for path in image_paths:
            relative = path.relative_to(image_dir).as_posix().lower()
            keys = {relative, path.name.lower(), path.stem.lower()}
            if path.stem.lower().endswith("_aligned"):
                unaligned_stem = path.stem[:-8].lower()
                keys.add(unaligned_stem)
                keys.add(f"{unaligned_stem}{path.suffix.lower()}")
            for key in keys:
                lookup.setdefault(key, path)
        return lookup

    @staticmethod
    def _resolve_image_reference(reference, image_dir, lookup):
        value = str(reference).strip().strip('"').replace("\\", "/")
        direct = image_dir / value
        if direct.is_file():
            return direct

        value_path = Path(value)
        stem = value_path.stem.lower()
        suffix = value_path.suffix.lower()
        keys = [value.lower(), value_path.name.lower(), stem]
        if not stem.endswith("_aligned"):
            keys.extend([f"{stem}_aligned", f"{stem}_aligned{suffix or '.jpg'}"])
        for key in keys:
            if key in lookup:
                return lookup[key]
        return None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        with Image.open(self.image_paths[index]) as image:
            image = image.copy()
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index]), index

    def update_label(self, index: int, new_label: int):
        if 0 <= index < len(self.labels) and self.labels[index] != new_label:
            self.labels[index] = new_label
            self.relabelled_count += 1

    def get_class_counts(self):
        return np.bincount(self.labels, minlength=len(EMOTION_NAMES))


def build_dataset(cfg: dict, split: str, transform=None):
    """Build the configured dataset while keeping a shared seven-class order."""
    data_cfg = cfg.get("data", {})
    dataset_name = str(data_cfg.get("dataset", "fer2013")).lower().replace("-", "")
    data_path = data_cfg.get("data_path", "dataset/fer13-split")

    if transform is None:
        transform = build_transforms(
            split,
            input_size=data_cfg.get("input_size", 48),
            use_random_erasing=data_cfg.get("use_random_erasing", True),
            erasing_prob=data_cfg.get("erasing_prob", 0.3),
            in_channels=cfg.get("model", {}).get("in_channels", 1),
            normalization=data_cfg.get("normalization", "symmetric"),
            color_mode=data_cfg.get("color_mode", "grayscale"),
            augmentation_profile=data_cfg.get("augmentation_profile", "fer2013"),
        )

    if dataset_name in ("rafdb", "rafdbbasic"):
        return RAFDBDataset(
            data_path=data_path,
            split=split,
            transform=transform,
            val_ratio=float(data_cfg.get("val_ratio", 0.1)),
            split_seed=int(data_cfg.get("split_seed", cfg.get("seed", {}).get("random_seed", 42))),
            label_encoding=data_cfg.get("label_encoding", "rafdb_1based"),
        )
    if dataset_name in ("fer2013", "fer13"):
        return PureImageFER2013(data_path=data_path, split=split, transform=transform)
    raise ValueError(f"Unsupported dataset: {data_cfg.get('dataset')}")


def build_dataloaders(cfg: dict):
    """Factory to build train, val, and test dataloaders."""
    data_cfg = cfg.get("data", {})
    data_path = data_cfg.get("data_path", "dataset/fer13-split")
    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 2)
    use_random_erasing = data_cfg.get("use_random_erasing", True)
    erasing_prob = data_cfg.get("erasing_prob", 0.3)
    seed = cfg.get("seed", {}).get("random_seed", None)
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        worker_init_fn = seed_worker

    input_size = data_cfg.get("input_size", 48)
    in_channels = cfg.get("model", {}).get("in_channels", 1)
    normalization = data_cfg.get("normalization", "symmetric")
    color_mode = data_cfg.get("color_mode", "grayscale")
    augmentation_profile = data_cfg.get("augmentation_profile", "fer2013")
    train_tf = build_transforms(
        "train", input_size, use_random_erasing, erasing_prob, in_channels,
        normalization, color_mode, augmentation_profile,
    )
    val_tf = build_transforms(
        "val", input_size, in_channels=in_channels, normalization=normalization,
        color_mode=color_mode, augmentation_profile=augmentation_profile,
    )
    test_tf = build_transforms(
        "test", input_size, in_channels=in_channels, normalization=normalization,
        color_mode=color_mode, augmentation_profile=augmentation_profile,
    )

    train_ds = build_dataset(cfg, split="train", transform=train_tf)
    val_ds = build_dataset(cfg, split="val", transform=val_tf)
    test_ds = build_dataset(cfg, split="test", transform=test_tf)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    return train_loader, val_loader, test_loader
