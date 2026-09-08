"""Shared config, dataset resolution and checkpoint construction for every entrypoint."""
import copy
import hashlib
from pathlib import Path

import torch
import yaml

from .models.attentive_scn_model import AttentiveSCNFER

ROOT = Path(__file__).resolve().parent.parent


def merge_config(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, _seen=()):
    path = Path(path)
    if not path.is_file():
        path = ROOT / path
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular _base_ config: {path}")
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    parent = cfg.pop("_base_", None)
    if parent:
        base, _ = load_config(path.parent / parent, (*_seen, path))
        cfg = merge_config(base, cfg)
    return cfg, path


def apply_overrides(cfg, overrides):
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Expected --set section.key=value, received {item}")
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown config section: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown config option: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    return cfg


def resolve_data_path(cfg, env="local", override_path=None, required_splits=("train", "val")):
    requested = Path(override_path or cfg.get("data", {}).get("data_path", "dataset/fer13-split"))
    candidates = [requested, requested / "fer13-split"]
    if override_path is None and env == "kaggle":
        for root in (
            "/kaggle/input/datasets/doduyquynii/fer13-split", "/kaggle/input/fer13-split",
            "/kaggle/input/sgu-2026-facial-expression-recognition/dataset",
            "/kaggle/input/sgu-2026-facial-expression-recognition", "/kaggle/input/fer2013/dataset",
            "/kaggle/input/fer2013",
        ):
            candidates.extend([Path(root), Path(root) / "fer13-split"])
    for candidate in candidates:
        if all((candidate / f"{split}.csv").is_file() for split in required_splits):
            return str(candidate.resolve())
    raise FileNotFoundError(f"Missing {required_splits} CSV files under {requested}; pass --data_path explicitly")


def build_model(cfg, pretrained=None):
    m = cfg.get("model", {})
    use_pretrained = m.get("use_pretrained", True) if pretrained is None else pretrained
    names = ("num_classes", "in_channels", "embed_dim", "num_attn_heads", "use_latent_graph",
             "dropout", "stem_init", "use_spatial_attention", "attention_type", "attention_norm",
             "backbone_mode", "preserve_pretrained_norm", "fusion_gate_init")
    return AttentiveSCNFER(
        backbone_name=m.get("backbone", "resnet50"), use_pretrained=use_pretrained,
        pretrained_weights_path=m.get("pretrained_weights_path", "") if pretrained is not False else "",
        **{name: m[name] for name in names if name in m},
    )


def load_checkpoint_model(weights_path, device="cpu", fallback_cfg=None):
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    cfg = copy.deepcopy(checkpoint.get("config") or fallback_cfg)
    if cfg is None:
        raise ValueError("Checkpoint has no config; provide --config for these legacy weights")
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    return model.to(device).eval(), cfg, checkpoint


def tta_mode(value):
    if value in (False, None, "none"):
        return False
    if value in (True, "flip"):
        return "flip"
    if value in ("multiscale", "multi_scale"):
        return "multiscale"
    raise ValueError(f"Unsupported TTA: {value}")


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
