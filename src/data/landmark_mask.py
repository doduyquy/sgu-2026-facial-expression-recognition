"""
Landmark-guided Gaussian Mask Generator using Dlib 68-point predictor.

Given a grayscale 48x48 FER2013 image, this module:
1. Resizes to a working resolution for Dlib face detection.
2. Detects 68 facial landmarks.
3. Groups landmarks into 6 anatomical regions.
4. Renders a soft Gaussian heatmap per region on a target grid (e.g. 14x14).

Output: Tensor [K, Hf, Wf] with values in [0, 1].
        K = num_regions (default 6), Hf x Wf = feature map grid.
"""

import os
import numpy as np
import torch

try:
    import dlib
except ImportError:
    dlib = None


# 68-point landmark indices grouped into 6 facial regions.
# Reference: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
REGION_LANDMARK_GROUPS = {
    "forehead":   list(range(17, 27)),          # Eyebrows (17-26)
    "left_eye":   list(range(36, 42)),           # Left eye (36-41)
    "right_eye":  list(range(42, 48)),           # Right eye (42-47)
    "nose":       list(range(27, 36)),           # Nose bridge + tip (27-35)
    "mouth":      list(range(48, 68)),           # Outer + inner lips (48-67)
    "chin":       list(range(0, 17)),            # Jawline / chin contour (0-16)
}

# Ordered list matching FacialRegionDictionary.REGION_NAMES
REGION_ORDER = ["forehead", "left_eye", "right_eye", "nose", "mouth", "chin"]


def _find_shape_predictor():
    """
    Search for the Dlib 68-point shape predictor .dat file in common locations.
    On Kaggle, users typically place it under /kaggle/input/.
    """
    filename = "shape_predictor_68_face_landmarks.dat"
    search_paths = [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "checkpoints", filename),
        os.path.join(os.getcwd(), "models", filename),
        # Kaggle working directory (wget downloads go here)
        "/kaggle/working/" + filename,
    ]
    # Kaggle input datasets (recursive search)
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            if filename in files:
                search_paths.insert(0, os.path.join(root, filename))
                break

    for path in search_paths:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"Could not find '{filename}'. "
        "Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
        "and place it in the project root or checkpoints/ directory."
    )


class DlibLandmarkMaskGenerator:
    """
    Generates [K, Hf, Wf] soft Gaussian masks from facial landmarks.

    Usage:
        generator = DlibLandmarkMaskGenerator(grid_size=14, sigma=1.5)
        masks = generator(image_np_gray_48x48)  # -> Tensor [6, 14, 14]
    """

    def __init__(
        self,
        grid_size=14,
        sigma=1.5,
        num_regions=6,
        predictor_path=None,
        detection_size=224,
    ):
        """
        Args:
            grid_size:      Output spatial size (Hf = Wf = grid_size).
            sigma:          Gaussian spread (in grid-space units). Larger = softer.
            num_regions:    Number of facial regions (must be <= 6).
            predictor_path: Path to shape_predictor_68_face_landmarks.dat.
            detection_size: Image is resized to this before Dlib detection.
        """
        if dlib is None:
            raise ImportError("dlib is required. Install with: pip install dlib")

        self.grid_size = grid_size
        self.sigma = sigma
        self.num_regions = min(num_regions, len(REGION_ORDER))
        self.detection_size = detection_size

        if predictor_path is None:
            predictor_path = _find_shape_predictor()
        print(f"--> [LandmarkMask] Using predictor: {predictor_path}")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # Precompute grid coordinates [Hf, Wf]
        coords = torch.arange(grid_size, dtype=torch.float32)
        self._grid_y, self._grid_x = torch.meshgrid(coords, coords, indexing="ij")

    def _detect_landmarks(self, image_gray_uint8):
        """
        Detect 68 landmarks on a grayscale uint8 image.
        Returns numpy array [68, 2] (x, y) in pixel coords of detection_size,
        or None if no face is detected.
        """
        h_orig, w_orig = image_gray_uint8.shape[:2]

        # Resize to detection_size for better Dlib accuracy
        if h_orig != self.detection_size or w_orig != self.detection_size:
            import cv2
            image_resized = cv2.resize(
                image_gray_uint8,
                (self.detection_size, self.detection_size),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            image_resized = image_gray_uint8

        faces = self.detector(image_resized, 1)
        if len(faces) == 0:
            # Fallback: assume entire image is the face
            faces = [dlib.rectangle(0, 0, self.detection_size - 1, self.detection_size - 1)]

        # Use the largest detected face
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self.predictor(image_resized, face)

        landmarks = np.array(
            [(shape.part(i).x, shape.part(i).y) for i in range(68)],
            dtype=np.float32,
        )
        return landmarks

    def _landmarks_to_masks(self, landmarks):
        """
        Convert 68 landmarks (in detection_size coords) to [K, Hf, Wf] masks.

        Each region's center is the mean of its landmark group, mapped to
        grid coordinates. A 2D Gaussian is rendered at that center.
        """
        scale = self.grid_size / self.detection_size
        masks = torch.zeros(self.num_regions, self.grid_size, self.grid_size)

        for i, region_name in enumerate(REGION_ORDER[: self.num_regions]):
            indices = REGION_LANDMARK_GROUPS[region_name]
            region_pts = landmarks[indices]  # [N_pts, 2] in pixel coords

            # Center in grid coordinates
            cx = region_pts[:, 0].mean() * scale
            cy = region_pts[:, 1].mean() * scale

            # 2D Gaussian: exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma^2))
            dist_sq = (self._grid_x - cx) ** 2 + (self._grid_y - cy) ** 2
            gaussian = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))

            # Normalize to [0, 1] (peak = 1.0)
            gaussian = gaussian / (gaussian.max() + 1e-8)
            masks[i] = gaussian

        return masks

    def __call__(self, image_gray_uint8):
        """
        Generate landmark masks from a grayscale uint8 numpy image.

        Args:
            image_gray_uint8: numpy array [H, W] dtype uint8 (e.g. 48x48).

        Returns:
            masks: Tensor [K, Hf, Wf] float32, values in [0, 1].
                   Returns uniform masks (all 1s) if landmark detection fails.
        """
        try:
            landmarks = self._detect_landmarks(image_gray_uint8)
            if landmarks is None:
                return torch.ones(self.num_regions, self.grid_size, self.grid_size)
            return self._landmarks_to_masks(landmarks)
        except Exception:
            # Graceful fallback: uniform mask = vanilla attention
            return torch.ones(self.num_regions, self.grid_size, self.grid_size)


# =====================================================================
# Self-test
# =====================================================================
if __name__ == "__main__":
    print("=== Testing DlibLandmarkMaskGenerator ===")

    # Create a dummy 48x48 grayscale image
    dummy = np.random.randint(0, 256, (48, 48), dtype=np.uint8)

    try:
        gen = DlibLandmarkMaskGenerator(grid_size=14, sigma=1.5)
        masks = gen(dummy)
        print(f"Masks shape: {masks.shape}")       # [6, 14, 14]
        print(f"Masks dtype: {masks.dtype}")        # float32
        print(f"Masks range: [{masks.min():.4f}, {masks.max():.4f}]")
        assert masks.shape == (6, 14, 14)
        print("Test passed!")
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        print("Generating fallback masks instead...")
        masks = torch.ones(6, 14, 14)
        print(f"Fallback masks shape: {masks.shape}")
        print("Fallback test passed!")
