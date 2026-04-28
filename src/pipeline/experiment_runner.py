"""Config-driven experiment runner."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.artifact_builder import (
    build_hierarchical_cache,
    ensure_pixel_motif_artifacts,
    load_artifacts_from_input,
    normalize_data_config,
    resolve_artifact_paths,
    run_command,
    validate_manifest,
    write_manifest,
    zip_artifacts,
)


ROOT_DIR = Path(__file__).resolve().parents[2]

# Modes supported by run_experiment
# build_and_train   : build artifacts from CSV then train (lần đầu chạy version đó)
# train_from_artifact: load artifact từ /kaggle/input rồi train (các lần sau)
# build_only        : chỉ build artifact, không train
# train_only        : skip build, dùng artifact đã có trong working, train
# debug_only        : chỉ chạy debug batch
VALID_MODES = {"build_and_train", "train_from_artifact", "build_only", "train_only", "debug_only"}


def load_experiment_config(config_name_or_path: str) -> tuple[dict[str, Any], Path]:
    """Load an experiment config by name or YAML path."""
    raw = Path(config_name_or_path)
    candidates: list[Path]
    if raw.suffix in {".yaml", ".yml"}:
        candidates = [raw, ROOT_DIR / raw]
    else:
        candidates = [
            ROOT_DIR / "configs" / "experiments" / f"{config_name_or_path}.yaml",
            ROOT_DIR / "configs" / f"{config_name_or_path}.yaml",
            raw,
        ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if not isinstance(cfg, dict):
                raise ValueError(f"Experiment config must be a mapping: {path}")
            return cfg, path.resolve()
    raise FileNotFoundError(f"Cannot find experiment config {config_name_or_path!r}. Tried: {candidates}")


def debug_hierarchical_batch(train_cfg: dict[str, Any], paths: dict[str, Path]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.debug_hierarchical_batch",
        "--config",
        str(train_cfg.get("config", "hierarchical_motif_gnn")),
        "--env",
        str(train_cfg.get("env", "kaggle")),
        "--pixel_motif_dataset_path",
        str(paths["pixel_motif_dir"]),
        "--graph_repo_path",
        str(paths["graph_repo"]),
        "--batch_size",
        str(train_cfg.get("debug_batch_size", 2)),
        "--num_workers",
        "0",
    ]
    run_command(cmd)


def debug_candidate_attention_batch(train_cfg: dict[str, Any], paths: dict[str, Path]) -> None:
    cmd = [
        sys.executable,
        "scripts/debug_candidate_attention_batch.py",
        "--config",
        str(train_cfg.get("config", "learnable_slot_candidate_motif_gnn")),
        "--candidate_attention_dir",
        str(paths.get("candidate_attention_dir", paths["pixel_motif_dir"])),
        "--graph_repo_path",
        str(paths["graph_repo"]),
        "--batch_size",
        str(train_cfg.get("debug_batch_size", 2)),
    ]
    run_command(cmd)


def debug_full_graph_d4_batch(train_cfg: dict[str, Any], paths: dict[str, Path]) -> None:
    cmd = [
        sys.executable,
        "scripts/debug_full_graph_d4_batch.py",
        "--config",
        str(train_cfg.get("config", "full_graph_adaptive_motif_slot_gnn_d4a")),
        "--env",
        str(train_cfg.get("env", "kaggle")),
        "--graph_repo_path",
        str(paths["graph_repo"]),
        "--batch_size",
        str(train_cfg.get("debug_batch_size", 2)),
        "--num_workers",
        "0",
    ]
    run_command(cmd)


def debug_batch_for_experiment(train_cfg: dict[str, Any], paths: dict[str, Path]) -> None:
    config_name = str(train_cfg.get("config", ""))
    if "full_graph_adaptive_motif_slot_gnn" in config_name:
        debug_full_graph_d4_batch(train_cfg, paths)
    elif "learnable_slot_candidate_motif_gnn" in config_name:
        debug_candidate_attention_batch(train_cfg, paths)
    else:
        debug_hierarchical_batch(train_cfg, paths)


def train_model(
    train_cfg: dict[str, Any],
    paths: dict[str, Path],
    *,
    epochs: int | None,
    no_wandb: bool,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.train",
        "--config",
        str(train_cfg["config"]),
        "--env",
        str(train_cfg.get("env", "kaggle")),
        "--pixel_motif_dataset_path",
        str(paths["pixel_motif_dir"]),
        "--candidate_attention_dataset_path",
        str(paths.get("candidate_attention_dir", paths["pixel_motif_dir"])),
        "--graph_repo_path",
        str(paths["graph_repo"]),
        "--epochs",
        str(epochs if epochs is not None else train_cfg.get("epochs", 80)),
        "--experiment_name",
        str(train_cfg.get("experiment_name", train_cfg.get("name", "")) or train_cfg.get("config", "")),
    ]
    if no_wandb or bool(train_cfg.get("no_wandb", False)):
        cmd.append("--no_wandb")
    for flag, value in [
        ("--max_train_batches", max_train_batches if max_train_batches is not None else train_cfg.get("max_train_batches")),
        ("--max_val_batches", max_val_batches if max_val_batches is not None else train_cfg.get("max_val_batches")),
        ("--max_test_batches", max_test_batches if max_test_batches is not None else train_cfg.get("max_test_batches")),
    ]:
        if value is not None:
            cmd.extend([flag, str(value)])
    run_command(cmd)


def zip_outputs(outputs_cfg: dict[str, Any], experiment_name: str) -> None:
    if not bool(outputs_cfg.get("zip_outputs", True)):
        return
    outputs_dir = ROOT_DIR / "outputs"
    if not outputs_dir.exists():
        print(f"[zip] Skip because outputs directory is missing: {outputs_dir}", flush=True)
        return
    zip_name = str(outputs_cfg.get("zip_name", f"{experiment_name}_outputs.zip"))
    zip_path = Path(outputs_cfg.get("zip_path", Path("/kaggle/working") / zip_name))
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=outputs_dir)
    print(f"[zip] Created {zip_path} ({zip_path.stat().st_size / 1024**2:.2f} MB)", flush=True)


def _resolve_mode(
    mode: str | None,
    *,
    build_only: bool,
    train_only: bool,
    debug_only: bool,
) -> str:
    """Resolve mode string, supporting both new --mode flag and legacy boolean flags."""
    # Legacy flags take precedence if mode is default
    legacy_flags = sum(bool(v) for v in [build_only, train_only, debug_only])
    if legacy_flags > 1:
        raise ValueError("--build_only, --train_only, and --debug_only are mutually exclusive")

    if mode is not None and mode not in VALID_MODES:
        raise ValueError(f"--mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    if build_only:
        return "build_only"
    if train_only:
        return "train_only"
    if debug_only:
        return "debug_only"
    if mode is not None:
        return mode
    return "build_and_train"


def run_experiment(
    config_name_or_path: str,
    *,
    csv_root: str | Path | None = None,
    out_root: str | Path | None = None,
    pixel_motif_dir: str | Path | None = None,
    epochs: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
    smoke: bool = False,
    # New hybrid mode
    mode: str | None = None,                    # build_and_train | train_from_artifact | build_only | train_only | debug_only
    artifact_input_path: str | Path | None = None,  # path khi mode=train_from_artifact
    zip_artifacts_after_build: bool = False,     # có zip toàn bộ artifacts sau khi build không
    # Legacy flags (vẫn giữ để backward compatible)
    build_only: bool = False,
    train_only: bool = False,
    debug_only: bool = False,
    no_wandb: bool = False,
    no_skip_existing: bool = False,
) -> None:
    """Run one experiment from config.

    Modes:
        build_and_train       Build artifacts from CSV then train. (default)
        train_from_artifact   Load artifact from artifact_input_path, validate manifest, then train.
        build_only            Build artifacts only, no train.
        train_only            Skip build, use existing artifacts in out_root, train.
        debug_only            Debug batch forward only.
    """
    cfg, cfg_path = load_experiment_config(config_name_or_path)
    experiment_cfg = dict(cfg.get("experiment", {}) or {})
    data_cfg = dict(cfg.get("data", {}) or {})
    train_cfg = dict(cfg.get("training", {}) or {})
    outputs_cfg = dict(cfg.get("outputs", {}) or {})
    experiment_name = str(experiment_cfg.get("name", cfg_path.stem))
    train_cfg.setdefault("experiment_name", experiment_name)

    resolved_mode = _resolve_mode(mode, build_only=build_only, train_only=train_only, debug_only=debug_only)

    if smoke:
        data_cfg["smoke"] = True
    if no_skip_existing:
        data_cfg["skip_existing"] = False
    if out_root is not None:
        out_root_text = str(out_root).replace("\\", "/")
        if not out_root_text.startswith("/kaggle/") and str(train_cfg.get("env", "kaggle")) == "kaggle":
            train_cfg["env"] = "local"
            print(
                f"[local] out_root={out_root} is not a Kaggle path; using training.env=local for subprocesses.",
                flush=True,
            )
    if pixel_motif_dir is not None:
        data_cfg["pixel_motif_dir"] = str(pixel_motif_dir)
    elif out_root is not None:
        data_cfg.pop("pixel_motif_dir", None)
        data_cfg.pop("motif_bank_dir", None)
        data_cfg.pop("candidate_dir", None)
        data_cfg.pop("candidate_attention_dir", None)

    data_cfg_normalized = normalize_data_config(data_cfg)
    default_out_root = Path(out_root or data_cfg_normalized.get("artifact_root", "/kaggle/working/artifacts"))

    print("=" * 100, flush=True)
    print(f"Experiment : {experiment_name}", flush=True)
    print(f"Config     : {cfg_path}", flush=True)
    print(f"Mode       : {resolved_mode}", flush=True)
    print(f"Model cfg  : {train_cfg.get('config')}", flush=True)
    print(f"out_root   : {default_out_root}", flush=True)

    # ------------------------------------------------------------------
    # MODE: train_from_artifact
    # ------------------------------------------------------------------
    if resolved_mode == "train_from_artifact":
        if artifact_input_path is None:
            raise ValueError(
                "--artifact_input_path is required when mode=train_from_artifact.\n"
                "Example: --artifact_input_path /kaggle/input/fer2013-pixel-motif-v2-hierarchial/artifacts"
            )

        print(f"artifact   : {artifact_input_path}", flush=True)
        print("=" * 100, flush=True)

        # Copy artifact từ input -> working
        paths = load_artifacts_from_input(artifact_input_path, default_out_root)

        # Nếu config yêu cầu V3 cache và chưa có -> build on top of V2
        if bool(data_cfg.get("build_hierarchical_cache", False)):
            v2_dir = paths["pixel_motif_dir"]  # V2 vừa copy từ input
            v3_dir = paths["out_root"] / "pixel_motif_dataset_v3_hierarchical"
            build_hierarchical_cache(
                data_cfg_normalized, v2_dir, paths["graph_repo"], v3_dir, skip_existing=True
            )
            paths["pixel_motif_dir"] = v3_dir
            print(f"[pipeline] Using V3 hierarchical cache: {v3_dir}", flush=True)

        # Validate manifest (optional — warn if manifest.json missing from older artifact uploads)
        require_node_indices = bool(train_cfg.get("debug_batch", False))  # C cần node_indices
        manifest_path = paths["out_root"] / "manifest.json"
        if str(data_cfg_normalized.get("recipe", "")) in {"candidate_attention_v1", "full_graph_d4"}:
            print("[manifest] Non pixel-motif artifact detected; skipping pixel-motif manifest checks.", flush=True)
        elif manifest_path.exists():
            validate_manifest(
                paths["out_root"],
                data_cfg_normalized,
                require_node_indices=require_node_indices,
                require_node_mask=require_node_indices,
            )
        else:
            print(
                "[warn] manifest.json not found in artifact root. "
                "Skipping validation — make sure the artifact was built with the correct config. "
                "Re-run with mode=build_and_train to regenerate a manifest.",
                flush=True,
            )

        print(f"graph_repo : {paths['graph_repo']}", flush=True)
        print(f"dataset    : {paths['pixel_motif_dir']}", flush=True)

        # Debug batch nếu cần
        if bool(train_cfg.get("debug_batch", train_cfg.get("debug_hierarchical_batch", False))):
            debug_batch_for_experiment(train_cfg, paths)

        if bool(train_cfg.get("enabled", True)):
            train_model(
                train_cfg,
                paths,
                epochs=epochs,
                no_wandb=no_wandb,
                max_train_batches=max_train_batches,
                max_val_batches=max_val_batches,
                max_test_batches=max_test_batches,
            )
            zip_outputs(outputs_cfg, experiment_name)
        else:
            print("Training disabled by experiment config.", flush=True)
        return

    # ------------------------------------------------------------------
    # All other modes: resolve paths from working/out_root
    # ------------------------------------------------------------------
    paths = resolve_artifact_paths(data_cfg, out_root_override=out_root)
    recipe = str(data_cfg_normalized.get("recipe", ""))
    if recipe == "candidate_attention_v1":
        dataset_log_path = paths["candidate_attention_dir"]
    elif recipe == "full_graph_d4":
        dataset_log_path = paths["graph_repo"]
    else:
        dataset_log_path = paths["pixel_motif_dir"]
    print(f"graph_repo : {paths['graph_repo']}", flush=True)
    print(f"dataset    : {dataset_log_path}", flush=True)
    print("=" * 100, flush=True)

    # ------------------------------------------------------------------
    # MODE: build_and_train | build_only
    # ------------------------------------------------------------------
    if resolved_mode in {"build_and_train", "build_only"}:
        paths = ensure_pixel_motif_artifacts(data_cfg, csv_root=csv_root, out_root=out_root)

        # Write manifest after successful build
        write_manifest(
            paths["out_root"],
            data_cfg_normalized,
            experiment_name,
            paths["pixel_motif_dir"],
        )

        # Optionally zip artifacts for download/publishing to Kaggle Dataset
        if zip_artifacts_after_build:
            zip_name = f"{experiment_name}_artifacts.zip"
            zip_path = Path("/kaggle/working") / zip_name
            zip_artifacts(paths["out_root"], zip_path)

        if resolved_mode == "build_only":
            print("Build-only mode complete.", flush=True)
            return

    # ------------------------------------------------------------------
    # MODE: train_only — skip build, use existing artifacts in working
    # ------------------------------------------------------------------
    if resolved_mode == "train_only":
        # Validate manifest if it exists; warn if missing
        manifest_path = paths["out_root"] / "manifest.json"
        if str(data_cfg_normalized.get("recipe", "")) in {"candidate_attention_v1", "full_graph_d4"}:
            print("[manifest] Non pixel-motif artifact detected; skipping pixel-motif manifest checks.", flush=True)
        elif manifest_path.exists():
            require_node_indices = bool(train_cfg.get("debug_batch", False))
            validate_manifest(
                paths["out_root"],
                data_cfg_normalized,
                require_node_indices=require_node_indices,
                require_node_mask=require_node_indices,
            )
        else:
            print(
                "[warn] manifest.json not found in artifact root. "
                "Skipping validation — make sure artifacts are from the correct build.",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Debug batch (build_and_train, train_only)
    # ------------------------------------------------------------------
    if bool(train_cfg.get("debug_batch", train_cfg.get("debug_hierarchical_batch", False))):
        debug_batch_for_experiment(train_cfg, paths)

    if resolved_mode == "debug_only":
        print("Debug-only mode complete.", flush=True)
        return

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    if bool(train_cfg.get("enabled", True)):
        train_model(
            train_cfg,
            paths,
            epochs=epochs,
            no_wandb=no_wandb,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
            max_test_batches=max_test_batches,
        )
        zip_outputs(outputs_cfg, experiment_name)
    else:
        print("Training disabled by experiment config.", flush=True)
