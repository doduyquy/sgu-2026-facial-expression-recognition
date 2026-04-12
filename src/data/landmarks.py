import numpy as np
import importlib


DEFAULT_LANDMARK_INDEXES = [
    33, 263, 1, 61, 291, 199, 4, 48, 278, 57, 287, 152
]


class LandmarkExtractor:
    """Extract sparse facial landmarks from a grayscale FER image.

    The extractor is optional and designed to fail safely:
    - If MediaPipe is unavailable, it returns zeros.
    - If no face is detected, it returns zeros.
    """

    def __init__(self, enabled=False, backend="mediapipe", landmark_indexes=None):
        self.enabled = enabled
        self.backend = backend
        self.landmark_indexes = landmark_indexes or DEFAULT_LANDMARK_INDEXES
        self.output_dim = len(self.landmark_indexes) * 2

        self._mp_face_mesh = None
        self._face_mesh = None

    def __getstate__(self):
        """Drop non-picklable runtime objects for DataLoader worker spawn."""
        state = self.__dict__.copy()
        state["_mp_face_mesh"] = None
        state["_face_mesh"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_backend(self):
        if not self.enabled or self.backend != "mediapipe":
            return
        if self._face_mesh is not None:
            return

        try:
            mp = importlib.import_module("mediapipe")
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
            )
        except Exception:
            self._face_mesh = None

    def _zeros_points(self):
        return np.zeros((len(self.landmark_indexes), 2), dtype=np.float32)

    def _zeros_mask(self):
        return np.zeros((len(self.landmark_indexes),), dtype=np.float32)

    @staticmethod
    def normalize_points_relative(points, mask, eps=1e-6):
        """Normalize points relative to face geometry using valid landmarks only."""
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
    def points_to_heatmaps(points, mask=None, heatmap_size=64, sigma=1.0, semantic_weighting=True):
        """Convert absolute normalized points (N, 2) into Gaussian heatmaps (N, H, W)."""
        if isinstance(heatmap_size, int):
            h, w = heatmap_size, heatmap_size
        else:
            h, w = heatmap_size

        n_points = points.shape[0]
        heatmaps = np.zeros((n_points, h, w), dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        if mask is None:
            mask = np.ones((n_points,), dtype=np.float32)

        for i in range(n_points):
            if mask[i] <= 0.5:
                continue
            x, y = float(points[i, 0]), float(points[i, 1])

            px = x * (w - 1)
            py = y * (h - 1)
            dist2 = (xx - px) ** 2 + (yy - py) ** 2
            heatmaps[i] = np.exp(-dist2 / (2.0 * (sigma ** 2))).astype(np.float32)
            if semantic_weighting:
                heatmaps[i] *= float(i + 1) / float(max(n_points, 1))

        return heatmaps

    @staticmethod
    def heatmaps_to_argmax_points(heatmaps):
        """Decode heatmaps (N, H, W) to normalized points (N, 2) with argmax."""
        if heatmaps.ndim != 3:
            raise ValueError("heatmaps must have shape (N, H, W)")

        n, h, w = heatmaps.shape
        points = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            flat_idx = int(np.argmax(heatmaps[i]))
            py, px = divmod(flat_idx, w)
            points[i, 0] = px / max(w - 1, 1)
            points[i, 1] = py / max(h - 1, 1)
        return points

    def extract_points_with_mask(self, gray_image_np):
        """Return absolute normalized points and visibility mask.

        Returns:
            points: (num_points, 2), absolute normalized in [0, 1]
            mask: (num_points,), 1 for detected point, 0 for missing point
        """
        if not self.enabled:
            return self._zeros_points(), self._zeros_mask()

        self._ensure_backend()

        if self.backend != "mediapipe" or self._face_mesh is None:
            return self._zeros_points(), self._zeros_mask()

        if gray_image_np.ndim != 2:
            return self._zeros_points(), self._zeros_mask()

        # MediaPipe FaceMesh expects 3-channel RGB image.
        rgb = np.stack([gray_image_np, gray_image_np, gray_image_np], axis=-1)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return self._zeros_points(), self._zeros_mask()

        face_lm = result.multi_face_landmarks[0].landmark
        h, w = gray_image_np.shape
        if h <= 0 or w <= 0:
            return self._zeros_points(), self._zeros_mask()

        points = np.zeros((len(self.landmark_indexes), 2), dtype=np.float32)
        mask = np.zeros((len(self.landmark_indexes),), dtype=np.float32)
        for i, idx in enumerate(self.landmark_indexes):
            if idx >= len(face_lm):
                continue
            x = float(face_lm[idx].x)
            y = float(face_lm[idx].y)
            # Clamp to [0, 1] for training stability.
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            points[i, 0] = x
            points[i, 1] = y
            mask[i] = 1.0

        return points, mask

    def extract_points(self, gray_image_np):
        """Backward-compatible helper returning points only."""
        points, _ = self.extract_points_with_mask(gray_image_np)
        return points

    def extract(self, gray_image_np):
        """Return flattened normalized landmarks of shape (2 * num_points,)."""
        points, _ = self.extract_points_with_mask(gray_image_np)
        return points.reshape(-1).astype(np.float32)
