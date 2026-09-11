"""Apply validation-selected original/horizontal-flip TTA to an existing checkpoint."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.data.dataset import EMOTION_NAMES, PureImageFER2013, build_transforms
from fads_scn.evaluation.evaluator import plot_confusion_matrix
from fads_scn.evaluation.weighted_flip_tta import WeightedHorizontalFlipTTASweep
from fads_scn.models.attentive_scn_model import AttentiveSCNFER


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select original/horizontal-flip TTA ratio on validation, then apply it once to test."
    )
    parser.add_argument("--weights", required=True, help="Path to attentive_scn_best.pth")
    parser.add_argument("--config", default="fads_scn/configs/scn_convnext.yaml", help="Matching model config")
    parser.add_argument("--env", choices=["local", "kaggle"], default="local")
    parser.add_argument("--data_path", default=None, help="Override FER split directory")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", default=None, help="Defaults to the checkpoint directory")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    return parser.parse_args()


def resolve_data_path(cfg, environment, override):
    if override:
        return override
    configured = cfg.get("data", {}).get("data_path", "dataset/fer13-split")
    if environment != "kaggle":
        return configured
    candidates = [
        "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split",
        "/kaggle/input/datasets/doduyquynii/fer13-split",
        "/kaggle/input/fer13-split/fer13-split",
        "/kaggle/input/fer13-split",
        "/kaggle/input/sgu-2026-facial-expression-recognition/dataset/fer13-split",
        "/kaggle/input/sgu-2026-facial-expression-recognition/fer13-split",
        "/kaggle/input/fer2013/dataset/fer13-split",
        "/kaggle/input/fer2013",
    ]
    return next((path for path in candidates if (Path(path) / "val.csv").exists()), configured)


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = repo_root / args.config
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    checkpoint_path = Path(args.weights)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    m_cfg = cfg["model"]
    model = AttentiveSCNFER(
        backbone_name=m_cfg.get("backbone", "resnet50"),
        num_classes=m_cfg.get("num_classes", 7),
        in_channels=m_cfg.get("in_channels", 1),
        embed_dim=m_cfg.get("embed_dim", 256),
        num_attn_heads=m_cfg.get("num_attn_heads", 8),
        use_latent_graph=m_cfg.get("use_latent_graph", True),
        use_spatial_attention=m_cfg.get("use_spatial_attention", True),
        graph_mode=m_cfg.get("graph_mode", "dense"),
        graph_topk=m_cfg.get("graph_topk", 3),
        graph_self_loop_bias=m_cfg.get("graph_self_loop_bias", 1.0),
        dropout=0.0,
        classifier_type=m_cfg.get("classifier_type", "linear"),
        cosface_scale=m_cfg.get("cosface_scale", 30.0),
        cosface_margin=m_cfg.get("cosface_margin", 0.20),
        use_pretrained=False,
        pretrained_weights_path="",
        stem_init=m_cfg.get("stem_init", "mean"),
    )
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.to(device).eval()

    data_cfg = cfg.get("data", {})
    data_path = resolve_data_path(cfg, args.env, args.data_path)
    transform_args = {
        "input_size": data_cfg.get("input_size", 48),
        "in_channels": m_cfg.get("in_channels", 1),
        "normalization": data_cfg.get("normalization", "symmetric"),
    }
    batch_size = args.batch_size or data_cfg.get("batch_size", 64)
    loader_args = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": data_cfg.get("num_workers", 2),
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(
        PureImageFER2013(data_path, split="val", transform=build_transforms("val", **transform_args)),
        **loader_args,
    )
    test_loader = DataLoader(
        PureImageFER2013(data_path, split="test", transform=build_transforms("test", **transform_args)),
        **loader_args,
    )

    eval_cfg = cfg.get("evaluation", {})
    print("\n[WEIGHTED FLIP TTA] Validation selects; test receives the frozen ratio.")
    print(f"Checkpoint: {checkpoint_path}\nData: {data_path}\nDevice: {device}")
    result = WeightedHorizontalFlipTTASweep.sweep_and_apply(
        model,
        val_loader,
        test_loader,
        device=device,
        flip_weights=eval_cfg.get("weighted_flip_weights"),
        selection_metric=eval_cfg.get("tta_selection_metric", "accuracy"),
    )
    for row in result["validation_results"]:
        print(
            f"[VAL] original={row['original_weight']:.1f} flip={row['flip_weight']:.1f} | "
            f"Acc={row['accuracy']*100:.2f}% F1={row['macro_f1']*100:.2f}%"
        )

    test_metrics = result["test_metrics"]
    print(
        f"\n[SELECTED] original={result['selected_original_weight']:.1f} "
        f"flip={result['selected_flip_weight']:.1f} by val {result['selection_metric']}"
    )
    print(f"[TEST] Acc={test_metrics['accuracy']*100:.2f}% | F1={test_metrics['macro_f1']*100:.2f}%")

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "weighted_flip_tta_selection_existing_checkpoint.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "selection_split": "val",
                "selection_metric": result["selection_metric"],
                "selected_original_weight": result["selected_original_weight"],
                "selected_flip_weight": result["selected_flip_weight"],
                "validation_results": [
                    {
                        key: row[key]
                        for key in ("original_weight", "flip_weight", "loss", "accuracy", "macro_f1", "hybrid_score")
                    }
                    for row in result["validation_results"]
                ],
                "test_metrics": {
                    key: test_metrics[key]
                    for key in ("loss", "accuracy", "macro_f1", "hybrid_score", "per_class_acc")
                },
            },
            handle,
            indent=2,
        )
    cm_path = output_dir / "confusion_matrix_test_weighted_flip_tta.png"
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        EMOTION_NAMES,
        cm_path,
        title=(
            "FER2013 Test | Weighted Flip TTA "
            f"(original={result['selected_original_weight']:.1f}, flip={result['selected_flip_weight']:.1f}) | "
            f"Acc={test_metrics['accuracy']*100:.2f}%"
        ),
    )
    print(f"[SAVED] {result_path}\n[SAVED] {cm_path}")


if __name__ == "__main__":
    main()
