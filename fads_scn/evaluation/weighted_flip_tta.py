"""Validation-only selection of the original/horizontal-flip TTA ratio."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from ..data.dataset import EMOTION_NAMES


class WeightedHorizontalFlipTTASweep:
    """Sweep original-vs-horizontal-flip logit weights without touching test selection."""

    @staticmethod
    @torch.no_grad()
    def sweep_and_apply(model, val_loader, test_loader, device, flip_weights=None, selection_metric="accuracy"):
        """
        Select ``(1-w) * logits_original + w * logits_horizontal_flip`` on validation,
        then evaluate test once with the selected ``w``. Vertical flips are deliberately
        excluded because an upside-down face is not label-preserving for FER.
        """
        if selection_metric not in {"accuracy", "macro_f1", "hybrid_score"}:
            raise ValueError("selection_metric must be accuracy, macro_f1, or hybrid_score")

        candidate_weights = np.linspace(0.0, 1.0, 11) if flip_weights is None else flip_weights
        weights = [float(weight) for weight in candidate_weights]
        if not weights or any(weight < 0.0 or weight > 1.0 for weight in weights):
            raise ValueError("flip_weights must contain one or more values in [0, 1]")
        weights = list(dict.fromkeys(weights))
        device = torch.device(device)
        model.eval()

        def collect_views(loader):
            original_logits, flipped_logits, all_targets = [], [], []
            for batch in loader:
                if len(batch) == 3:
                    images, targets, _ = batch
                else:
                    images, targets = batch
                images = images.to(device, non_blocking=True)
                original_logits.append(model(images, use_tta=False)["logits"].cpu())
                flipped_logits.append(model(torch.flip(images, dims=[-1]), use_tta=False)["logits"].cpu())
                all_targets.append(targets.cpu())
            return torch.cat(original_logits), torch.cat(flipped_logits), torch.cat(all_targets)

        def metrics_for(logits, targets):
            predictions = torch.argmax(logits, dim=-1).numpy()
            labels = targets.numpy()
            cm = confusion_matrix(labels, predictions, labels=list(range(len(EMOTION_NAMES))))
            with np.errstate(divide="ignore", invalid="ignore"):
                per_class = np.nan_to_num(np.diag(cm) / cm.sum(axis=1))
            accuracy = float(accuracy_score(labels, predictions))
            macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
            return {
                "loss": float(F.cross_entropy(logits, targets).item()),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "hybrid_score": float(accuracy * macro_f1),
                "per_class_acc": {
                    name: round(float(value) * 100, 2)
                    for name, value in zip(EMOTION_NAMES, per_class)
                },
                "confusion_matrix": cm,
                "report": classification_report(
                    labels,
                    predictions,
                    labels=list(range(len(EMOTION_NAMES))),
                    target_names=EMOTION_NAMES,
                    digits=4,
                    zero_division=0,
                    output_dict=True,
                ),
            }

        val_original, val_flipped, val_targets = collect_views(val_loader)
        validation_results = []
        for weight in weights:
            metrics = metrics_for((1.0 - weight) * val_original + weight * val_flipped, val_targets)
            validation_results.append({"flip_weight": weight, "original_weight": 1.0 - weight, **metrics})

        # If metrics tie, prefer the conventional 50/50 average to avoid an arbitrary
        # one-view choice caused by a tie on a small validation split.
        selected = max(
            validation_results,
            key=lambda item: (
                item[selection_metric],
                item["macro_f1"],
                item["accuracy"],
                -abs(item["flip_weight"] - 0.5),
            ),
        )

        test_metrics = None
        if test_loader is not None:
            test_original, test_flipped, test_targets = collect_views(test_loader)
            weight = selected["flip_weight"]
            test_metrics = metrics_for((1.0 - weight) * test_original + weight * test_flipped, test_targets)

        return {
            "selection_metric": selection_metric,
            "selected_flip_weight": selected["flip_weight"],
            "selected_original_weight": selected["original_weight"],
            "validation_results": validation_results,
            "validation_selected": selected,
            "test_metrics": test_metrics,
        }
