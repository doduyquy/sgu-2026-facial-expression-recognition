import importlib
import numpy as np


DEFAULT_LANDMARK_INDEXES = [33, 263, 1, 61, 291, 199, 4, 48, 278, 57, 287, 152]


class LandmarkExtractor:
    """Optional landmark extractor with safe fallback if backend is unavailable."""

    def __init__(self, enabled=False, backend="mediapipe", landmark_indexes=None, min_detection_confidence=0.2):
        self.enabled = enabled
        self.backend = backend
        self.landmark_indexes = landmark_indexes or DEFAULT_LANDMARK_INDEXES
        self.min_detection_confidence = min_detection_confidence
        self._face_mesh = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_face_mesh"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _zeros_points(self):
        return np.zeros((len(self.landmark_indexes), 2), dtype=np.float32)

    def _zeros_mask(self):
        return np.zeros((len(self.landmark_indexes),), dtype=np.float32)

    def _ensure_backend(self):
        if not self.enabled or self.backend != "mediapipe":
            return
        if self._face_mesh is not None:
            return

        try:
            mp = importlib.import_module("mediapipe")
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=self.min_detection_confidence,
            )
        except Exception:
            self._face_mesh = None

    @staticmethod
    def normalize_points_relative(points, mask, eps=1e-6):
        pts = points.copy()
        valid = mask > 0.5
        if valid.sum() < 2:
            return pts

        xs = pts[valid, 0]
        ys = pts[valid, 1]
        cx = xs.mean()
        cy = ys.mean()
        scale = np.std(np.concatenate([xs, ys])) + eps

        pts[valid, 0] = (xs - cx) / scale
        pts[valid, 1] = (ys - cy) / scale
        return pts

    @staticmethod
    def points_to_heatmaps(points, mask=None, heatmap_size=64, sigma=1.0):
        if isinstance(heatmap_size, int):
            h, w = heatmap_size, heatmap_size
        else:
            h, w = heatmap_size

        n = points.shape[0]
        heatmaps = np.zeros((n, h, w), dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w]

        if mask is None:
            mask = np.ones((n,), dtype=np.float32)

        for i in range(n):
            if mask[i] <= 0.5:
                continue

            px = float(points[i, 0]) * (w - 1)
            py = float(points[i, 1]) * (h - 1)
            dist2 = (xx - px) ** 2 + (yy - py) ** 2
            heatmaps[i] = np.exp(-dist2 / (2.0 * sigma ** 2)).astype(np.float32)

        return heatmaps

    @staticmethod
    def _upsample_gray(gray_image_np):
        """FaceMesh struggles on 48x48 FER images, so upscale before inference."""
        h, w = gray_image_np.shape
        target = 192
        if min(h, w) >= target:
            return gray_image_np

        scale = max(1, int(np.ceil(target / float(min(h, w)))))
        up = np.repeat(np.repeat(gray_image_np, scale, axis=0), scale, axis=1)
        return up.astype(np.uint8)

    def extract_points_with_mask(self, gray_image_np):
        if not self.enabled:
            return self._zeros_points(), self._zeros_mask()

        self._ensure_backend()
        if self.backend != "mediapipe" or self._face_mesh is None:
            return self._zeros_points(), self._zeros_mask()
        if gray_image_np.ndim != 2:
            return self._zeros_points(), self._zeros_mask()

        # Retry strategy for tiny FER images: original -> upsampled.
        candidates = [gray_image_np]
        upsampled = self._upsample_gray(gray_image_np)
        if upsampled.shape != gray_image_np.shape:
            candidates.append(upsampled)

        result = None
        for cand in candidates:
            rgb = np.stack([cand, cand, cand], axis=-1)
            result = self._face_mesh.process(rgb)
            if result.multi_face_landmarks:
                break

        if result is None or (not result.multi_face_landmarks):
            return self._zeros_points(), self._zeros_mask()

        landmarks = result.multi_face_landmarks[0].landmark
        points = self._zeros_points()
        mask = self._zeros_mask()

        for i, idx in enumerate(self.landmark_indexes):
            if idx >= len(landmarks):
                continue
            x = min(max(float(landmarks[idx].x), 0.0), 1.0)
            y = min(max(float(landmarks[idx].y), 0.0), 1.0)
            points[i, 0] = x
            points[i, 1] = y
            mask[i] = 1.0

        return points, mask
