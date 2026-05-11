import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.emotions_dict import EMOTION_DICT, EMOTION_NAMES
from src.data.transforms import build_transform
from src.models import get_model
from src.utils.config import load_config


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "region_aligned_fer_17042026_1656_best.pth"
)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise ValueError("Checkpoint does not contain a valid state dict.")


def strip_prefix_if_all_keys(state_dict, prefix):
    keys = list(state_dict.keys())
    if keys and all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def configure_for_checkpoint(config, state_dict):
    legacy_standard_transformer = (
        any(k.startswith("transformer_encoder.layers.0.self_attn.") for k in state_dict)
        and "vgg_type_embed" not in state_dict
        and "visual_pos_embed" not in state_dict
    )

    if legacy_standard_transformer:
        config["model"]["legacy_checkpoint_compat"] = True
        config["model"]["fusion_type"] = "transformer"
        config["model"]["cross_attention_direction"] = "region_query"
        config["model"]["use_clip_dictionary"] = False
        print("[INFO] Detected legacy RegionAlignedFER checkpoint.")
    else:
        config["model"]["legacy_checkpoint_compat"] = False
        print("[INFO] Detected current RegionAlignedFER checkpoint/config.")

    return config


def load_fer_model(args, device):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    state_dict = strip_prefix_if_all_keys(state_dict, "module.")
    state_dict = strip_prefix_if_all_keys(state_dict, "_orig_mod.")

    config = load_config(args.config, env="local")
    config["logging"]["use_wandb"] = False
    config = configure_for_checkpoint(config, state_dict)

    model = get_model(name=config["model"]["name"], config=config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    transform = build_transform(config, split="test")
    print(f"[INFO] Loaded model: {config['model']['name']} on {device}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    if isinstance(checkpoint, dict):
        print(f"[INFO] Checkpoint epoch: {checkpoint.get('epoch', '?')}")

    return model, transform


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = box_area(a)
    area_b = box_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes, scores=None, iou_threshold=0.35):
    if not boxes:
        return []

    if scores is None:
        scores = [box_area(box) for box in boxes]

    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        idx = int(order[0])
        keep.append(boxes[idx])
        rest = []
        for item in order[1:]:
            j = int(item)
            if iou_xyxy(boxes[idx], boxes[j]) < iou_threshold:
                rest.append(j)
        order = np.array(rest, dtype=int)

    return keep


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(1, min(width, round(x2))))
    y2 = int(max(1, min(height, round(y2))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def expand_to_square(box, width, height, padding=0.25):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + padding)
    return clip_box(
        (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2),
        width,
        height,
    )


def filter_face_boxes(boxes, width, height, args):
    if not boxes:
        return []

    image_area = width * height
    min_image_side = min(width, height)
    largest_area = max(box_area(box) for box in boxes)
    filtered = []

    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        area = box_area(box)

        if area / image_area < args.min_face_area_ratio:
            continue
        if min(w, h) / min_image_side < args.min_face_side_ratio:
            continue
        if largest_area > 0 and area / largest_area < args.drop_small_relative:
            continue

        filtered.append(box)

    filtered = sorted(filtered, key=box_area, reverse=True)
    if args.max_faces > 0:
        filtered = filtered[: args.max_faces]
    return filtered


def build_face_detectors(use_profile=False):
    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
    ]
    if use_profile:
        cascade_names.append("haarcascade_profileface.xml")

    detectors = []
    for cascade_name in cascade_names:
        cascade_path = Path(cv2.data.haarcascades) / cascade_name
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            print(f"[WARN] Could not load cascade: {cascade_path}")
            continue
        detectors.append((cascade_name, detector))

    if not detectors:
        raise RuntimeError("No Haar cascade detector could be loaded.")

    return detectors


def detect_faces(frame_bgr, detectors, args):
    height, width = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    boxes = []
    scores = []
    for _, detector in detectors:
        found = detector.detectMultiScale(
            gray,
            scaleFactor=args.scale_factor,
            minNeighbors=args.min_neighbors,
            minSize=(args.min_face_size, args.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, w, h in found:
            box = (int(x), int(y), int(x + w), int(y + h))
            boxes.append(box)
            scores.append(float(w * h))

    candidate_count = len(boxes)
    boxes = nms_boxes(boxes, scores=scores, iou_threshold=args.nms_iou)
    nms_count = len(boxes)
    boxes = filter_face_boxes(boxes, width, height, args)
    kept_count = len(boxes)
    crop_boxes = [expand_to_square(box, width, height, args.box_padding) for box in boxes]
    return boxes, crop_boxes, {
        "candidates": candidate_count,
        "after_nms": nms_count,
        "kept": kept_count,
    }


@torch.inference_mode()
def predict_faces(frame_bgr, display_boxes, crop_boxes, model, transform, device):
    if not crop_boxes:
        return []

    tensors = []
    for x1, y1, x2, y2 in crop_boxes:
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        tensors.append(transform(crop_pil))

    batch = torch.stack(tensors).to(device)
    output = model(batch)
    if isinstance(output, tuple):
        output = output[0]
    probs = torch.softmax(output, dim=1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)

    results = []
    for box, pred, prob_vec in zip(display_boxes, preds, probs):
        pred = int(pred)
        results.append(
            {
                "box": box,
                "label": pred,
                "emotion": EMOTION_DICT[pred],
                "confidence": float(prob_vec[pred]),
                "probs": prob_vec,
            }
        )

    return results


def draw_results(frame_bgr, results, fps=None, detect_stats=None):
    box_color = (145, 235, 170)
    label_bg = (219, 105, 236)
    text_color = (20, 20, 20)

    height, width = frame_bgr.shape[:2]
    line_width = max(2, min(width, height) // 350)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(width, height) / 1200.0)
    thickness = max(1, line_width - 1)

    for result in results:
        x1, y1, x2, y2 = result["box"]
        label = f"{result['emotion']}: {int(round(result['confidence'] * 100))}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_color, line_width)

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad_x, pad_y = 5, 4
        label_x = x2
        label_y = y1 - text_h - baseline - 2 * pad_y
        if label_x + text_w + 2 * pad_x > width:
            label_x = max(0, width - text_w - 2 * pad_x)
        if label_y < 0:
            label_y = y1

        cv2.rectangle(
            frame_bgr,
            (label_x, label_y),
            (label_x + text_w + 2 * pad_x, label_y + text_h + baseline + 2 * pad_y),
            label_bg,
            -1,
        )
        cv2.putText(
            frame_bgr,
            label,
            (label_x + pad_x, label_y + text_h + pad_y),
            font,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    help_text = "q/x/ESC: quit | s: save screenshot"
    cv2.putText(frame_bgr, help_text, (12, 28), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(
            frame_bgr,
            f"FPS: {fps:.1f}",
            (12, 56),
            font,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if detect_stats is not None:
        stats_text = (
            f"Faces: {detect_stats.get('kept', 0)} "
            f"(cand {detect_stats.get('candidates', 0)}, nms {detect_stats.get('after_nms', 0)})"
        )
        cv2.putText(
            frame_bgr,
            stats_text,
            (12, 84),
            font,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame_bgr


def ensure_detection_log(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "face_id",
                    "emotion",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    *[f"prob_{name}" for name in EMOTION_NAMES],
                    "screenshot",
                ]
            )
    return log_path


def ensure_frame_log(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "frame_index",
                    "frame_path",
                    "num_faces",
                    "detections_json",
                ]
            )
    return log_path


def serialize_detections(results):
    detections = []
    for face_id, result in enumerate(results):
        x1, y1, x2, y2 = result["box"]
        detections.append(
            {
                "face_id": face_id,
                "emotion": result["emotion"],
                "confidence": round(float(result["confidence"]), 6),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "probs": {
                    name: round(float(prob), 6)
                    for name, prob in zip(EMOTION_NAMES, result["probs"])
                },
            }
        )
    return detections


def save_frame_if_needed(annotated_frame, results, frame_index, args):
    if not args.save_every_frame:
        return ""

    if args.frame_save_interval > 1 and frame_index % args.frame_save_interval != 0:
        return ""

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    frame_path = args.frame_dir / f"frame_{frame_index:06d}_{timestamp}.jpg"
    cv2.imwrite(str(frame_path), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, args.frame_jpeg_quality])

    with open(args.frame_log_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                timestamp,
                frame_index,
                str(frame_path),
                len(results),
                json.dumps(serialize_detections(results), ensure_ascii=False),
            ]
        )

    return str(frame_path)


def should_log_detection(result, frame_index, recent_logs, args):
    if result["confidence"] < args.save_threshold:
        return False

    for previous in recent_logs:
        if previous["label"] != result["label"]:
            continue
        if frame_index - previous["frame_index"] >= args.save_cooldown_frames:
            continue
        if iou_xyxy(previous["box"], result["box"]) >= args.log_iou_match:
            return False

    return True


def log_detections(results, annotated_frame, frame_index, recent_logs, args, frame_image_path=""):
    to_log = [
        (face_id, result)
        for face_id, result in enumerate(results)
        if should_log_detection(result, frame_index, recent_logs, args)
    ]
    if not to_log:
        return

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    screenshot_path = frame_image_path or ""
    if args.save_screenshot_on_detection and not screenshot_path:
        screenshot_path = str(args.screenshot_dir / f"detection_{timestamp}_f{frame_index:06d}.png")
        cv2.imwrite(screenshot_path, annotated_frame)

    with open(args.log_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for face_id, result in to_log:
            x1, y1, x2, y2 = result["box"]
            writer.writerow(
                [
                    timestamp,
                    face_id,
                    result["emotion"],
                    f"{result['confidence']:.6f}",
                    x1,
                    y1,
                    x2,
                    y2,
                    *[f"{float(prob):.6f}" for prob in result["probs"]],
                    screenshot_path,
                ]
            )
            recent_logs.append(
                {
                    "frame_index": frame_index,
                    "label": result["label"],
                    "box": result["box"],
                }
            )

    keep_after = frame_index - args.save_cooldown_frames
    recent_logs[:] = [item for item in recent_logs if item["frame_index"] >= keep_after]

    print(
        f"[INFO] Logged {len(to_log)} detection(s) >= {args.save_threshold * 100:.0f}%"
        + (f": {screenshot_path}" if screenshot_path else "")
    )


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime webcam facial expression recognition demo."
    )
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", type=str, default="vgg_resnet_region")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam preview horizontally.")
    parser.add_argument("--max-faces", type=int, default=3, help="1 for selfie mode, 0 to keep all faces, 3 to keep up to 3 faces.")
    parser.add_argument("--min-face-size", type=int, default=40)
    parser.add_argument("--scale-factor", type=float, default=1.08)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--use-profile", action="store_true", help="Also use profile-face cascade; may add false positives.")
    parser.add_argument("--box-padding", type=float, default=0.25)
    parser.add_argument("--min-face-area-ratio", type=float, default=0.006)
    parser.add_argument("--min-face-side-ratio", type=float, default=0.04)
    parser.add_argument("--drop-small-relative", type=float, default=0.03)
    parser.add_argument("--nms-iou", type=float, default=0.35)
    parser.add_argument("--screenshot-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "webcam_emotion"))
    parser.add_argument("--log-file", type=str, default=str(PROJECT_ROOT / "outputs" / "webcam_emotion" / "detections.csv"))
    parser.add_argument("--save-threshold", type=float, default=0.50, help="Log detections with confidence >= this value.")
    parser.add_argument("--save-cooldown-frames", type=int, default=60, help="Avoid logging the same label/box again for this many frames.")
    parser.add_argument("--log-iou-match", type=float, default=0.50, help="IoU used to treat a new box as the same recently logged face.")
    parser.add_argument(
        "--save-screenshot-on-detection",
        action="store_true",
        default=True,
        help="Save annotated screenshot when a detection passes the threshold. Enabled by default.",
    )
    parser.add_argument(
        "--no-save-screenshot-on-detection",
        action="store_false",
        dest="save_screenshot_on_detection",
        help="Do not save annotated screenshots for threshold detections.",
    )
    parser.add_argument("--save-every-frame", action="store_true", help="Save every annotated webcam frame to disk.")
    parser.add_argument("--frame-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "webcam_emotion" / "frames"))
    parser.add_argument("--frame-log-file", type=str, default=str(PROJECT_ROOT / "outputs" / "webcam_emotion" / "frames.csv"))
    parser.add_argument("--frame-save-interval", type=int, default=1, help="Save one frame every N frames when --save-every-frame is enabled.")
    parser.add_argument("--frame-jpeg-quality", type=int, default=90)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if args.save_threshold > 1.0:
        args.save_threshold = args.save_threshold / 100.0

    device = resolve_device(args.device)
    model, transform = load_fer_model(args, device)
    detectors = build_face_detectors(use_profile=args.use_profile)

    screenshot_dir = Path(args.screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    args.screenshot_dir = screenshot_dir
    args.log_file = ensure_detection_log(args.log_file)
    args.frame_dir = Path(args.frame_dir)
    args.frame_dir.mkdir(parents=True, exist_ok=True)
    args.frame_log_file = ensure_frame_log(args.frame_log_file)
    args.frame_save_interval = max(1, int(args.frame_save_interval))
    args.frame_jpeg_quality = int(max(1, min(100, args.frame_jpeg_quality)))
    print(f"[INFO] Detection log: {args.log_file}")
    print(f"[INFO] Logging threshold: {args.save_threshold * 100:.0f}%")
    if args.save_every_frame:
        print(f"[INFO] Saving every {args.frame_save_interval} frame(s) to: {args.frame_dir}")
        print(f"[INFO] Frame log: {args.frame_log_file}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("[INFO] Webcam started.")
    print("[INFO] Press q, x, or ESC in the webcam window to quit. Press s to save a screenshot.")
    print("[INFO] You can also close the webcam window.")

    fps = None
    tick_prev = cv2.getTickCount()
    window_name = "FER Webcam Demo"
    frame_index = 0
    recent_logs = []

    while True:
        frame_index += 1
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Could not read frame from webcam.")
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        display_boxes, crop_boxes, detect_stats = detect_faces(frame, detectors, args)
        results = predict_faces(frame, display_boxes, crop_boxes, model, transform, device)

        tick_now = cv2.getTickCount()
        elapsed = (tick_now - tick_prev) / cv2.getTickFrequency()
        tick_prev = tick_now
        current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        fps = current_fps if fps is None else 0.9 * fps + 0.1 * current_fps

        annotated = draw_results(frame, results, fps=fps, detect_stats=detect_stats)
        frame_image_path = save_frame_if_needed(annotated, results, frame_index, args)
        log_detections(results, annotated, frame_index, recent_logs, args, frame_image_path)
        cv2.imshow(window_name, annotated)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Webcam window closed.")
            break

        key = cv2.waitKey(10) & 0xFF
        if key in (ord("q"), ord("x"), 27):
            break
        if key == ord("s"):
            out_path = screenshot_dir / "webcam_emotion_screenshot.png"
            cv2.imwrite(str(out_path), annotated)
            print(f"[INFO] Saved screenshot: {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam stopped.")


if __name__ == "__main__":
    main()
