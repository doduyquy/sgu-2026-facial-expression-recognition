import argparse
from pathlib import Path
import torch
import os
import sys

# ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Upgrade an old checkpoint to include epochs_total and optional metadata")
    parser.add_argument("--ckpt", required=True, help="Path to existing checkpoint file")
    parser.add_argument("--out", default=None, help="Output path for upgraded checkpoint (default: same dir with _upgraded.pth)")
    parser.add_argument("--total-epochs", type=int, default=None, help="Total epochs to set in upgraded checkpoint")
    parser.add_argument("--wandb-run-id", default=None, help="Optional wandb run id to attach")
    parser.add_argument("--non-interactive", action='store_true', help="Don't prompt; abort if required fields missing")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location='cpu')

    epoch = int(ckpt.get('epoch', 0))
    has_epochs_total = 'epochs_total' in ckpt
    total_epochs = args.total_epochs or ckpt.get('epochs_total', None)

    if total_epochs is None:
        if args.non_interactive:
            print("No total epochs provided and checkpoint missing epochs_total. Aborting (non-interactive).")
            return
        else:
            v = input("Enter total epochs to set in upgraded checkpoint (required): ")
            if v.strip() == '':
                print("No value entered. Aborting.")
                return
            try:
                total_epochs = int(v.strip())
            except Exception:
                print("Invalid integer. Aborting.")
                return

    # Copy existing ckpt but ensure keys exist
    new_ckpt = dict(ckpt)
    new_ckpt['epoch'] = epoch
    new_ckpt['epochs_total'] = int(total_epochs)

    # attach wandb run id if provided
    if args.wandb_run_id is not None:
        new_ckpt['wandb_run_id'] = args.wandb_run_id

    # if optimizer present but missing param_groups lr, leave as-is

    out_path = args.out
    if out_path is None:
        out_path = ckpt_path.with_name(ckpt_path.stem + "_upgraded.pth")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(new_ckpt, str(out_path))
    print(f"Saved upgraded checkpoint to: {out_path}")


if __name__ == '__main__':
    main()
