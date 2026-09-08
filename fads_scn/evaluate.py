"""Evaluate one checkpoint using a fixed TTA mode, with optional explicit logit bias."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.runtime import load_config, load_checkpoint_model, resolve_data_path, tta_mode, checkpoint_hash
from fads_scn.data.dataset import PureImageFER2013, transforms_from_config, EMOTION_NAMES, seed_worker
from fads_scn.evaluation.evaluator import evaluate_model, plot_confusion_matrix
from fads_scn.training.trainer import serializable_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", help="Fallback only for legacy checkpoints without embedded config")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--env", choices=["local", "kaggle"], default="local")
    parser.add_argument("--data_path")
    parser.add_argument("--device")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--tta", "--use_tta", choices=["none", "flip", "multiscale"], default=None)
    parser.add_argument("--bias", help="Calibration JSON; applied explicitly, never auto-selected using test")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    fallback = load_config(args.config)[0] if args.config else None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, checkpoint = load_checkpoint_model(args.weights, device, fallback)
    mode = tta_mode(args.tta if args.tta is not None else checkpoint.get("tta", cfg.get("training", {}).get("val_tta", "flip")))
    data_path = resolve_data_path(cfg, args.env, args.data_path, (args.split,))
    # Even --split train must use deterministic evaluation transforms.
    dataset = PureImageFER2013(data_path, args.split, transforms_from_config(cfg, "val"))
    loader = DataLoader(dataset, batch_size=args.batch_size or cfg.get("data", {}).get("batch_size", 64),
                        shuffle=False, num_workers=cfg.get("data", {}).get("num_workers", 2),
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42))
    bias = None
    if args.bias:
        with open(args.bias, encoding="utf-8") as stream:
            result = json.load(stream)
        if result.get("search_split") != "val" or result.get("class_names") != EMOTION_NAMES:
            raise ValueError("Bias artifact must record validation search and matching class names")
        if tta_mode(result.get("tta")) != mode:
            raise ValueError("Bias artifact TTA does not match evaluation TTA")
        if result.get("checkpoint_sha256") != checkpoint_hash(args.weights):
            raise ValueError("Bias artifact belongs to a different checkpoint")
        bias = result["best_bias"]
    metrics = evaluate_model(model, loader, device, use_tta=mode, class_bias=bias)
    print(f"{args.split.upper()} | {cfg['model']['backbone']} | TTA={mode} | Acc={metrics['accuracy']:.2%} | F1={metrics['macro_f1']:.2%} | NLL={metrics['nll']:.4f}")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.weights).resolve().parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_{mode or 'none'}{'_bias' if bias is not None else ''}"
    with (output_dir / f"metrics_{tag}.json").open("w", encoding="utf-8") as stream:
        json.dump(serializable_metrics(metrics), stream, indent=2, allow_nan=False)
    plot_confusion_matrix(metrics["confusion_matrix"], EMOTION_NAMES, output_dir / f"confusion_matrix_{tag}.png",
                          title=f"FER2013 {args.split.upper()} | TTA={mode} | Acc={metrics['accuracy']:.2%}")


if __name__ == "__main__":
    main()
