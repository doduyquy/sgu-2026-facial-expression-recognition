import math

import torch


class RegionOcclusionGenerator:
    """Create masked training views for face-region robustness."""

    _FACE_ANCHORS = {
        "upper_face": (0.50, 0.34),
        "center_face": (0.50, 0.50),
        "mouth": (0.50, 0.70),
    }

    def __init__(self, config=None):
        cfg = config or {}
        self.apply_prob = float(cfg.get("apply_prob", 0.5))
        self.min_area = float(cfg.get("min_area", 0.08))
        self.max_area = float(cfg.get("max_area", 0.22))
        self.min_aspect = float(cfg.get("min_aspect", 0.75))
        self.max_aspect = float(cfg.get("max_aspect", 1.35))
        self.fill_value = float(cfg.get("fill_value", 0.5))
        self.policy = str(cfg.get("policy", "mixed_face_regions")).lower()
        self.anchor_jitter = float(cfg.get("anchor_jitter", 0.08))

        if not 0.0 <= self.apply_prob <= 1.0:
            raise ValueError("training.occlusion_consistency.apply_prob must be in [0, 1].")
        if not 0.0 < self.min_area <= self.max_area <= 1.0:
            raise ValueError(
                "training.occlusion_consistency min_area/max_area must satisfy "
                "0 < min_area <= max_area <= 1."
            )
        if not 0.0 < self.min_aspect <= self.max_aspect:
            raise ValueError(
                "training.occlusion_consistency min_aspect/max_aspect must satisfy "
                "0 < min_aspect <= max_aspect."
            )
        if self.policy not in ("random", "mixed_face_regions"):
            raise ValueError(
                "training.occlusion_consistency.policy must be either "
                "'random' or 'mixed_face_regions'."
            )
        if self.anchor_jitter < 0.0:
            raise ValueError("training.occlusion_consistency.anchor_jitter must be >= 0.")

    @staticmethod
    def _uniform(low, high, device):
        if low == high:
            return low
        return torch.empty((), device=device).uniform_(low, high).item()

    def _choose_mode(self, device):
        if self.policy == "random":
            return "random"

        choice = torch.rand((), device=device).item()
        if choice < 0.40:
            return "random"
        if choice < 0.60:
            return "upper_face"
        if choice < 0.80:
            return "mouth"
        return "center_face"

    def _sample_box(self, height, width, mode, device):
        total_area = max(height * width, 1)
        target_area = self._uniform(self.min_area, self.max_area, device) * total_area
        log_aspect = self._uniform(
            math.log(self.min_aspect),
            math.log(self.max_aspect),
            device,
        )
        aspect = math.exp(log_aspect)

        box_w = max(1, min(width, int(round(math.sqrt(target_area * aspect)))))
        box_h = max(1, min(height, int(round(math.sqrt(target_area / aspect)))))

        if mode == "random":
            max_left = max(width - box_w, 0)
            max_top = max(height - box_h, 0)
            left = (
                int(torch.randint(0, max_left + 1, (1,), device=device).item())
                if max_left > 0
                else 0
            )
            top = (
                int(torch.randint(0, max_top + 1, (1,), device=device).item())
                if max_top > 0
                else 0
            )
            return top, left, box_h, box_w

        anchor_x, anchor_y = self._FACE_ANCHORS[mode]
        jitter_x = self._uniform(-self.anchor_jitter, self.anchor_jitter, device)
        jitter_y = self._uniform(-self.anchor_jitter, self.anchor_jitter, device)
        center_x = min(max(anchor_x + jitter_x, 0.0), 1.0) * width
        center_y = min(max(anchor_y + jitter_y, 0.0), 1.0) * height

        left = int(round(center_x - box_w / 2.0))
        top = int(round(center_y - box_h / 2.0))
        left = max(0, min(left, max(width - box_w, 0)))
        top = max(0, min(top, max(height - box_h, 0)))
        return top, left, box_h, box_w

    def __call__(self, images):
        if images.dim() != 4:
            raise ValueError("RegionOcclusionGenerator expects images shaped [B, C, H, W].")

        masked = images.clone()
        applied = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)
        if self.apply_prob <= 0.0:
            return masked, applied

        _, _, height, width = images.shape
        for index in range(images.size(0)):
            if torch.rand((), device=images.device).item() > self.apply_prob:
                continue

            mode = self._choose_mode(images.device)
            top, left, box_h, box_w = self._sample_box(height, width, mode, images.device)
            masked[index, :, top: top + box_h, left: left + box_w] = self.fill_value
            applied[index] = True

        # Avoid DDP rank skew: with small per-GPU batches, pure Bernoulli
        # sampling can occasionally mask zero local samples on one rank while
        # another rank runs the masked-view branch. Keeping at least one local
        # occlusion when the feature is enabled makes the training path steadier.
        if not applied.any().item() and images.size(0) > 0:
            index = int(torch.randint(0, images.size(0), (1,), device=images.device).item())
            mode = self._choose_mode(images.device)
            top, left, box_h, box_w = self._sample_box(height, width, mode, images.device)
            masked[index, :, top: top + box_h, left: left + box_w] = self.fill_value
            applied[index] = True

        return masked, applied
