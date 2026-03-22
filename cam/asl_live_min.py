# Live ASL detection

#works similar to aslcam.py, except it uses trained model to predict letter

# cam/asl_live_min.py

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import mediapipe as mp
from collections import deque
from datetime import datetime

# 
# camera settings
# 
CKPT_PATH = str(Path(__file__).resolve().parents[1] / "models" / "min_asl.pth") #best checkpoint 
CAM_INDEX = 0            # 0 = default Mac camera
CROP_PAD = 40            # extra pixels around detected hand
MIN_CONF = 0.60          # green label if confidence >= this
START_MIRROR = True      # True = selfie view

# debug settings 
SMOOTH_FRAMES = 8        # number of frames to average probs over
SHOW_CROP_PREVIEW = False  # toggle with 'c'
DRAW_LANDMARKS = False      # toggle with 'l'
SAVE_LABEL_KEYS = True      # press letter keys to save crop to that label

# ImageNet stats 
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#using the apple silicon GPU, if that not available then CPU
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# building model
# replaces the last layer 
def build_model(num_classes: int) -> nn.Module:
    m = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feats, num_classes)
    return m

# converts the opencn amera to a normalized model 
# goes thorugh resizing and normalizing to match training preprocessing
def preprocess_crop(bgr_crop: np.ndarray, size: int) -> torch.Tensor: #covert to tensor 
    img = cv2.resize(bgr_crop, (size, size), interpolation=cv2.INTER_AREA) #resize 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 #change pixels to 0-1
    img = (img - IMAGENET_MEAN) / IMAGENET_STD #uses imagenet stats
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0) #retuned shape is [1, C, H, W]

# bounding box around the hand
def hand_bbox_from_landmarks(results, W: int, H: int, pad: int) -> tuple[int, int, int, int] | None:
    if not results.multi_hand_landmarks:
        return None

    lm = results.multi_hand_landmarks[0] # x and y coordinates 
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]

    # make x and y coordiantes into raw pixel bounds 
    x0 = min(xs) * W
    y0 = min(ys) * H
    x1 = max(xs) * W
    y1 = max(ys) * H

    # finds middle and w and h of box 
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0)
    h = (y1 - y0)
    side = max(w, h) + 2 * pad
    #square crop
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(W, cx + side / 2))
    y1 = int(min(H, cy + side / 2))

    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)

# main functions
def main() -> None:
    print("asl_live_min.py starting...")

    #  Load checkpoint , if it doesnt exist run training first
    if not Path(CKPT_PATH).exists():
        raise SystemExit(f"Checkpoint not found: {CKPT_PATH}. Run training.py first.")

    dev = pick_device()
    print("Using device:", dev) 

    ckpt = torch.load(CKPT_PATH, map_location=dev) #load ckpt data
    classes = ckpt["classes"] # class names 
    img_size = int(ckpt.get("img_size", 224)) # image size 
    print(f"Loaded checkpoint with {len(classes)} classes. img_size={img_size}")

    #rebuilds mobilenetv3 model, laod trained weights
    model = build_model(num_classes=len(classes)).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # opens cam
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
    print("Camera opened?", cap.isOpened())
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    mirror = START_MIRROR
    show_box = True
    show_crop = SHOW_CROP_PREVIEW
    prob_hist = deque(maxlen=max(1, SMOOTH_FRAMES)) # store recent predicitons
    capture_root = Path("data_capture")
    capture_root.mkdir(exist_ok=True)

    # loads mediapipe hand tracking 
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

    print("Live ASL running.  ESC to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror:
                frame = cv2.flip(frame, 1) # selfie mode 

            H, W = frame.shape[:2]

            # MediaPipe expects RGB, convert 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
                # draws mediapipe landmarks on hands
            if draw_landmarks and results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                )
                # crops around hand
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

            status = "HAND" if hand_found else "NO HAND (using center crop)" # detect hand status
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

            # Preprocess to match training and predict
            x = preprocess_crop(crop, img_size).to(dev)
            with torch.no_grad(): # run model and convert logits to probability 
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            # Smooth probabilities across frames (reduces flicker) 
            prob_hist.append(probs)
            avg_probs = np.mean(prob_hist, axis=0)

            pred_idx = int(avg_probs.argmax()) #predict most likely class
            pred_lbl = classes[pred_idx] #predict most likely class
            pred_conf = float(avg_probs[pred_idx]) #predict most likely class

            # Color: green if confident, orange if else
            color = (0, 255, 0) if pred_conf >= MIN_CONF else (0, 165, 255)

            # show prediction and label
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
                cv2.imshow("ASL Crop", crop) # second window

            cv2.imshow("ASL Live - ESC/x quit", frame) # main window
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
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        try:
            cv2.destroyWindow("ASL Crop")
        except Exception:
            pass


if __name__ == "__main__":
    main()