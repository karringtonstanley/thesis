# Live ASL detection

#works similar to aslcam.py, except it uses trained model to predict letter

#Steps
#1 - Read frame from webcam
#2 detect hand
#3 put bounding box over hand
#4 resize to 224x224 (same aspect for training model)
#5 normailize using the same mean/std used during training
#6 run prediction letter model 
 
# cam/asl_live_min.py

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import mediapipe as mp
# --- Additions for smoothing, crop preview, data capture ---
from collections import deque
from datetime import datetime

# -------------------------
# Settings you can customize
# -------------------------
CKPT_PATH = "models/min_asl.pth"
CAM_INDEX = 0            # 0 = default Mac camera
CROP_PAD = 40            # extra pixels around detected hand
MIN_CONF = 0.60          # green label if confidence >= this
START_MIRROR = True      # True = selfie view

# --- Smoothing and debug settings ---
SMOOTH_FRAMES = 8        # number of frames to average probs over
SHOW_CROP_PREVIEW = False  # toggle with 'c'
DRAW_LANDMARKS = False      # toggle with 'l'
SAVE_LABEL_KEYS = True      # press letter keys to save crop to that label

# ImageNet stats (must match training transforms)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device() -> torch.device:
    """Pick Apple GPU (MPS) if available; otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
    """Rebuild MobileNetV3-small with a new classifier head for num_classes."""
    m = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feats, num_classes)
    return m


def preprocess_crop(bgr_crop: np.ndarray, size: int) -> torch.Tensor:
    """OpenCV BGR crop -> normalized tensor (1,3,size,size) for the model."""
    img = cv2.resize(bgr_crop, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0)


def hand_bbox_from_landmarks(results, W: int, H: int, pad: int) -> tuple[int, int, int, int] | None:
    """Create a padded *square* bounding box from MediaPipe landmarks."""
    if not results.multi_hand_landmarks:
        return None

    lm = results.multi_hand_landmarks[0]
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]

    # Raw pixel bounds (rect)
    x0 = min(xs) * W
    y0 = min(ys) * H
    x1 = max(xs) * W
    y1 = max(ys) * H

    # Make it square by expanding the shorter side
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0)
    h = (y1 - y0)
    side = max(w, h) + 2 * pad

    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(W, cx + side / 2))
    y1 = int(min(H, cy + side / 2))

    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def main() -> None:
    print("✅ asl_live_min.py starting...")

    # ---- 1) Load checkpoint ----
    if not Path(CKPT_PATH).exists():
        raise SystemExit(f"Checkpoint not found: {CKPT_PATH}. Run training.py first.")

    dev = pick_device()
    print("Using device:", dev)

    ckpt = torch.load(CKPT_PATH, map_location=dev)
    classes = ckpt["classes"]
    img_size = int(ckpt.get("img_size", 224))
    print(f"Loaded checkpoint with {len(classes)} classes. img_size={img_size}")

    # ---- 2) Build model ----
    model = build_model(num_classes=len(classes)).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ---- 3) Open camera ----
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
    print("Camera opened?", cap.isOpened())
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Try closing other apps using the camera or change CAM_INDEX.")

    mirror = START_MIRROR
    show_box = True
    show_crop = SHOW_CROP_PREVIEW
    prob_hist = deque(maxlen=max(1, SMOOTH_FRAMES))
    capture_root = Path("data_capture")
    capture_root.mkdir(exist_ok=True)

    # ---- 4) Start MediaPipe Hands ----
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    draw_landmarks = DRAW_LANDMARKS
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Live ASL running. Close window to stop, or press ESC/x to quit. Press letter keys A–Z to save crops to data_capture/<LETTER>.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            H, W = frame.shape[:2]

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if draw_landmarks and results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                )

            bbox = hand_bbox_from_landmarks(results, W, H, pad=CROP_PAD)
            hand_found = bbox is not None

            # If a hand is found, crop it; otherwise use a center crop fallback
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                crop = frame[y0:y1, x0:x1].copy()
                if show_box:
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            else:
                side = min(H, W)
                y0 = (H - side) // 2
                x0 = (W - side) // 2
                crop = frame[y0:y0 + side, x0:x0 + side].copy()

            status = "HAND" if hand_found else "NO HAND (using center crop)"
            cv2.putText(
                frame,
                status,
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Preprocess and predict
            x = preprocess_crop(crop, img_size).to(dev)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            # ---- Smooth probabilities across frames (reduces flicker) ----
            prob_hist.append(probs)
            avg_probs = np.mean(prob_hist, axis=0)

            pred_idx = int(avg_probs.argmax())
            pred_lbl = classes[pred_idx]
            pred_conf = float(avg_probs[pred_idx])

            # Color: green if confident, orange otherwise
            color = (0, 255, 0) if pred_conf >= MIN_CONF else (0, 165, 255)

            # Main label at top
            cv2.putText(
                frame,
                f"{pred_lbl} ({pred_conf:.2f})",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )

            if show_crop:
                cv2.imshow("ASL Crop", crop)

            cv2.imshow("ASL Live - ESC/x quit", frame)
            key = cv2.waitKey(1) & 0xFF

            # Quit on ESC or 'x'
            if key == 27 or key == ord('x'):
                break

            # Save crop to the label you pressed (A–Z)
            if SAVE_LABEL_KEYS and ((ord('a') <= key <= ord('z')) or (ord('A') <= key <= ord('Z'))):
                lab = chr(key).upper()
                out_dir = capture_root / lab
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
                cv2.imwrite(str(out_dir / fname), crop)
                print(f"Saved labeled crop '{lab}' -> {out_dir / fname}")

    finally:
        # Always clean up even if something errors
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        try:
            cv2.destroyWindow("ASL Crop")
        except Exception:
            pass


if __name__ == "__main__":
    main()