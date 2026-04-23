# 📐 Khảo sát Template Thiết Kế — `sgu-2026-facial-expression-recognition`

> Mục đích: Phân tích kiến trúc & thiết kế để copy template cho dự án CV tương tự.

---

## 1. Tổng quan cấu trúc thư mục

```
sgu-2026-facial-expression-recognition/
│
├── configs/                ← YAML configs (base + per-model + env)
│   ├── base.yaml           ← Default config chung cho toàn project
│   ├── simple_cnn.yaml     ← Override config cho SimpleCNN
│   ├── resnet.yaml         ← Override config cho ResNet
│   ├── vgg19.yaml          ← Override config cho VGG19
│   ├── env.yaml            ← Platform config (local / kaggle)
│   └── resnet_kaggle_test.yaml
│
├── data/                   ← Dataset thực (gitignore)
│   └── fer13-split/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/                    ← SOURCE CODE chính
│   ├── data/               ← Data pipeline
│   │   ├── dataset.py      ← Custom Dataset class
│   │   ├── transforms.py   ← Augmentation + preprocessing
│   │   └── dataloader.py   ← build_dataloader() factory
│   │
│   ├── models/             ← Tất cả model architectures
│   │   ├── __init__.py     ← MODEL_REGISTRY + get_model() factory
│   │   ├── simple_cnn.py
│   │   ├── resnet.py
│   │   ├── vgg.py
│   │   └── ...
│   │
│   ├── training/           ← Training pipeline
│   │   ├── trainer.py      ← Trainer class (fit/train_one_epoch/validate)
│   │   ├── optimizer.py    ← build_optimizer() + build_scheduler()
│   │   └── losses.py       ← build_loss() factory
│   │
│   ├── evaluation/         ← Đánh giá & phân tích
│   │   ├── evaluator.py    ← evaluate_and_show()
│   │   ├── metrics.py      ← compute_metrics(), plot_confusion_matrix()
│   │   └── error_analysis.py
│   │
│   └── utils/              ← Utilities chung
│       ├── config.py       ← load_config() (YAML merge)
│       ├── checkpoint.py   ← save/load .pth
│       ├── seed.py         ← set_seed()
│       ├── logger_wandb.py ← WandB integration
│       ├── visualization.py← plot helper
│       └── data_stats.py   ← class distribution stats
│
├── scripts/                ← Entry points chạy nhanh
│   ├── train.py            ← main() entry point
│   ├── evaluate.py
│   ├── prepare_data.py
│   └── analyze_errors.py
│
├── notebooks/              ← EDA + demo Jupyter
├── outputs/                ← Checkpoints, logs, figures (gitignore)
├── tests/                  ← Unit tests
├── requirements.txt
└── README.md
```

---

## 2. Các Design Pattern được sử dụng

### 2.1 Registry-based Model System

**File:** `src/models/__init__.py`

```python
MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "resnet": lambda config, **kw: ResNet50(...),
}

def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(...)
    return MODEL_REGISTRY[name](**kwargs)
```

> ✅ **Cách dùng:** Thêm model mới chỉ cần:
> 1. Tạo file `src/models/new_model.py`
> 2. Đăng ký vào `MODEL_REGISTRY`
> 3. Tạo `configs/new_model.yaml`

---

### 2.2 Config-driven với YAML Inheritance

**Hệ thống 3 tầng config:**

```
base.yaml           ← default toàn project
  └── model.yaml    ← override riêng cho model (ghi đè lên base)
        └── env.yaml ← override platform (local/kaggle)
```

**Merge logic** (`src/utils/config.py`):
```python
config = _deep_update(base_config, model_config)  # deep merge
config = {**config, **env_config}                  # env override
```

**base.yaml** định nghĩa toàn bộ schema:
```yaml
data:
  name, num_classes, image_size, batch_size, num_workers
seed:
  random_seed
model:
  name, pretrained
training:
  epochs, patience, weight_decay, loss, optimizer, lr, scheduler
logging:
  use_wandb, wandb_entity, wandb_project
env:
  platform  # local | kaggle
kaggle:
  data_path, root_path
local:
  data_path, root_path
```

---

### 2.3 Builder / Factory Pattern cho Components

Mỗi component chính đều có một hàm `build_*()`:

| Hàm | File | Mô tả |
|-----|------|--------|
| `build_dataloader(config, data_path)` | `src/data/dataloader.py` | Tạo train/val/test DataLoader |
| `get_model(name, config)` | `src/models/__init__.py` | Tạo model theo registry |
| `build_loss(config, class_weights)` | `src/training/losses.py` | CrossEntropy / FocalLoss |
| `build_optimizer(model, config)` | `src/training/optimizer.py` | Adam / SGD |
| `build_scheduler(optimizer, config)` | `src/training/optimizer.py` | ReduceLROnPlateau / Step / Cosine |
| `load_config(model, env)` | `src/utils/config.py` | Load & merge YAML |

---

### 2.4 Trainer Class Pattern

**File:** `src/training/trainer.py`

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 criterion, optimizer, scheduler,
                 config, device, run_name, save_dir): ...

    def train_one_epoch(self) -> (loss, acc): ...
    def validate(self) -> (loss, acc): ...
    def fit(self) -> (all_train_loss, all_val_loss): ...
```

**Tính năng nổi bật của Trainer:**
- ✅ Early stopping (patience-based)
- ✅ Best checkpoint auto-save theo `val_loss`
- ✅ LR scheduler step (ReduceLROnPlateau / others)
- ✅ WandB logging tích hợp
- ✅ 3-phase staged training (0-30%: clean, 30-70%: SCN, 70-100%: refinement)
- ✅ `_extract_logits(outputs)` — hỗ trợ model output dạng `dict` hoặc `tensor`

---

### 2.5 Standardized Model Output Interface

Model output phải có chuẩn:
```python
# Kiểu dict (khuyến nghị cho multi-output model)
return {
    "logits": ...,          # tensor (B, num_classes)
    "aux_losses": {         # dict các auxiliary losses
        "landmark_diversity": ...,
        ...
    }
}

# Hoặc đơn giản: tensor thẳng (B, num_classes) - cho simple model
return logits
```

`Trainer._extract_logits()` tự xử lý cả 2 trường hợp.

---

### 2.6 Platform-aware Entry Point

**File:** `scripts/train.py`

```python
# Chạy local:
python scripts/train.py --config resnet --env local

# Chạy Kaggle:
python scripts/train.py --config resnet --env kaggle
```

Một lệnh chạy được cả 2 môi trường, chỉ đổi `--env`.

---

## 3. Luồng dữ liệu tổng thể

```
main() in scripts/train.py
    │
    ├── load_config(args.config, args.env)
    │       └── base.yaml ← model.yaml ← env.yaml  (deep merge)
    │
    ├── set_seed(random_seed)
    │
    ├── build_dataloader(config, data_path)
    │       └── Dataset (CSV-based) + Transforms → DataLoader x3
    │
    ├── get_model(name=config['model']['name'], config=config)
    │       └── MODEL_REGISTRY lookup → instantiate
    │
    ├── build_loss(config, class_weights)
    ├── build_optimizer(model, config)
    ├── build_scheduler(optimizer, config)
    │
    ├── Trainer.fit()
    │       ├── train_one_epoch() [N epochs]
    │       │       └── forward → loss → backward → step
    │       ├── validate()
    │       ├── early_stop check
    │       ├── checkpoint save (best val_loss)
    │       └── WandB log
    │
    └── evaluate_and_show(model, test_loader, ...)
            ├── compute_metrics() → accuracy, F1, classification report
            ├── plot_confusion_matrix()
            └── plot_prediction_grid() (10 đúng / 10 sai)
```

---

## 4. Hướng dẫn Copy Template sang dự án mới

### Bước 1: Clone cấu trúc thư mục

```
new-project/
├── configs/
│   ├── base.yaml       ← copy & chỉnh schema
│   ├── env.yaml        ← copy & chỉnh paths
│   └── model_a.yaml    ← tạo mới
├── data/
├── src/
│   ├── data/           ← copy dataset.py, transforms.py, dataloader.py → chỉnh
│   ├── models/         ← copy __init__.py (registry) → thêm model mới
│   ├── training/       ← copy trainer.py, losses.py, optimizer.py → giữ nguyên phần lớn
│   ├── evaluation/     ← copy evaluator.py, metrics.py → giữ nguyên
│   └── utils/          ← copy nguyên cả folder (config, seed, checkpoint, wandb)
├── scripts/
│   └── train.py        ← copy → chỉnh imports nếu cần
├── outputs/
├── notebooks/
└── requirements.txt
```

### Bước 2: Các file cần chỉnh nhiều nhất

| File | Việc cần làm |
|------|--------------|
| `configs/base.yaml` | Đổi `num_classes`, `image_size`, dataset name, WandB project |
| `configs/env.yaml` | Đổi data_path và root_path theo dự án mới |
| `src/data/dataset.py` | Viết lại Dataset class theo format dữ liệu mới |
| `src/data/transforms.py` | Chỉnh augmentation phù hợp task mới |
| `src/models/*.py` | Thêm model architecture mới |
| `src/models/__init__.py` | Đăng ký model mới vào `MODEL_REGISTRY` |

### Bước 3: Các file có thể copy gần như nguyên

| File | Ghi chú |
|------|---------|
| `src/utils/config.py` | ✅ Giữ nguyên |
| `src/utils/seed.py` | ✅ Giữ nguyên |
| `src/utils/checkpoint.py` | ✅ Giữ nguyên |
| `src/utils/logger_wandb.py` | ✅ Giữ nguyên (chỉ đổi project name trong yaml) |
| `src/training/optimizer.py` | ✅ Giữ nguyên |
| `src/training/losses.py` | ✅ Giữ nguyên (thêm loss nếu cần) |
| `src/training/trainer.py` | ⚠️ Giữ phần lớn, xóa code SCN/landmark nếu không cần |
| `src/evaluation/metrics.py` | ✅ Giữ nguyên cho classification tasks |
| `scripts/train.py` | ⚠️ Chỉnh class_weight logic nếu khác |

---

## 5. Tóm tắt Template Patterns

| Pattern | Mô tả |
|---------|-------|
| **Registry Pattern** | `MODEL_REGISTRY` dict → `get_model(name)` factory |
| **Config Inheritance** | `base.yaml` ← `model.yaml` ← `env.yaml` (deep merge) |
| **Builder Pattern** | `build_*()` functions cho mọi component lớn |
| **Trainer Class** | Encapsulate toàn bộ training loop vào 1 class |
| **Standardized Output** | Model output là `dict {"logits": ..., "aux_losses": {...}}` |
| **Platform Awareness** | `--env local/kaggle` chuyển đổi paths tự động |
| **WandB Integration** | Log metrics + images + model artifacts |
| **Early Stopping** | Patience-based, save best checkpoint theo val_loss |

