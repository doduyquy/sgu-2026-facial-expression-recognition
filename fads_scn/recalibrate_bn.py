"""Recompute train-only BN statistics on an existing checkpoint; select by validation."""
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
from fads_scn.data.dataset import PureImageFER2013, transforms_from_config, build_clean_train_loader
from fads_scn.evaluation.evaluator import evaluate_model
from fads_scn.training.ema import recalibrate_bn
from fads_scn.training.trainer import selection_key, serializable_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", help="Fallback for legacy weights without embedded config")
    parser.add_argument("--data_path")
    parser.add_argument("--env", choices=["local", "kaggle"], default="local")
    parser.add_argument("--device")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--tta", choices=["none", "flip", "multiscale"])
    parser.add_argument("--output", help="New checkpoint path; refuses to overwrite existing files")
    args = parser.parse_args()
    weights = Path(args.weights).resolve()
    target = Path(args.output).resolve() if args.output else weights.with_name(weights.stem + "_bn.pth")
    if target == weights or target.exists() or target.with_suffix(".json").exists():
        raise FileExistsError(f"Choose a new output path: {target}")
    fallback = load_config(args.config)[0] if args.config else None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, checkpoint = load_checkpoint_model(weights, device, fallback)
    cfg["data"]["data_path"] = resolve_data_path(cfg, args.env, args.data_path)
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size
    mode = tta_mode(args.tta if args.tta is not None else checkpoint.get("tta", cfg.get("training", {}).get("val_tta", "flip")))
    val_ds = PureImageFER2013(cfg["data"]["data_path"], "val", transforms_from_config(cfg, "val"))
    val = DataLoader(val_ds, batch_size=cfg["data"].get("batch_size", 64), shuffle=False,
                     num_workers=cfg["data"].get("num_workers", 2), generator=torch.Generator().manual_seed(43))
    before = evaluate_model(model, val, device, use_tta=mode)
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    batches = recalibrate_bn(model, build_clean_train_loader(cfg), device, args.max_batches)
    after = evaluate_model(model, val, device, use_tta=mode)
    accepted = batches > 0 and selection_key(after) > selection_key(before)
    selected = after if accepted else before
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()} if accepted else original
    result = {"epoch": checkpoint.get("epoch"), "state_dict": state, "config": cfg,
              "tta": mode, "selection_metric": "accuracy", "weights_source": checkpoint.get("weights_source", "unknown"),
              "bn_recalibrated": accepted or checkpoint.get("bn_recalibrated", False),
              "val_acc": selected["accuracy"], "macro_f1": selected["macro_f1"],
              "val_loss": selected["nll"], "hybrid_score": selected["hybrid_score"],
              "parent_checkpoint_sha256": checkpoint_hash(weights)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        torch.save(result, stream)
    report = {"before": serializable_metrics(before), "after": serializable_metrics(after),
              "accepted": accepted, "batches": batches, "output": str(target)}
    with target.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"VAL before={before['accuracy']:.2%} after={after['accuracy']:.2%}; selected={'recalibrated' if accepted else 'original'}")
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
