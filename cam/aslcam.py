#asl cam to create datasets
#open camera and click on coinciding key for letter to add a picture for a datset
# Example: Open camera and click "A" key to add to A dataset

import mediapipe as mp
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import os
from datetime import datetime

# Camera 
CAM_INDEX = 0 # open webcam 
USE_HAND_DETECTION = True # crop around hand when detected 
SHOW_WINDOWS = True
EVENT_THRESHOLD = 15
CROP_PADDING = 30
TARGET_SIZE = (224, 224) # saved image dimesions

def find_events(cur_gray, prev_gray, threshold=15): # create a camera event by looking at current and previous frame
    diff = cur_gray.astype(np.int16) - prev_gray.astype(np.int16)
    pos_mask = diff > threshold #brighter
    neg_mask = diff < -threshold #darker

    events = []
    event_mask = np.zeros((cur_gray.shape[0], cur_gray.shape[1], 3), dtype=np.uint8)
    event_mask[pos_mask] = (0, 255, 0) #positive events
    event_mask[neg_mask] = (255, 0, 255) #negative events

    ys_pos, xs_pos = np.where(pos_mask)
    for x, y in zip(xs_pos, ys_pos):
        events.append({"x": int(x), "y": int(y), "type": 1})

    ys_neg, xs_neg = np.where(neg_mask)
    for x, y in zip(xs_neg, ys_neg):
        events.append({"x": int(x), "y": int(y), "type": -1})

    return events, event_mask
# saves images to folder
def save_labeled_frame(img, label, base_dir="../data"):
    # this will create data/A, data/B, etc. 
    dir_path = os.path.join(base_dir, label)
    os.makedirs(dir_path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(dir_path, f"{label}_{ts}.jpg")
    cv2.imwrite(filename, img)
    print(f"[saved] {filename}")

def main(): # opens webcam, crops frames, saves labels when you press a key on keyboard
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open camera.")
        return
        # hand detector uses bounding boxes
    detector = HandDetector(maxHands=1) if USE_HAND_DETECTION else None
    prev_gray = None

    #  press the letter, save the image
    KEY_TO_LABEL = {
        ord('a'): "A",
        ord('b'): "B",
        ord('c'): "C",
        ord('d'): "D",
        ord('e'): "E",
        ord('f'): "F",
        ord('g'): "G",
        ord('h'): "H",
        ord('i'): "I",
        # ord('j'): "J",
        ord('k'): "K",
        ord('l'): "L",
        ord('m'): "M",
        ord('n'): "N",
        ord('o'): "O",
        ord('p'): "P",
        ord('q'): "Q",
        ord('r'): "R",
        ord('s'): "S",
        ord('t'): "T",
        ord('u'): "U",
        ord('v'): "V",
        ord('w'): "W",
        ord('x'): "X",
        ord('y'): "Y",
        # ord('z'): "Z",
        # skip j and z for now , they motion-based
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = frame.copy()

        # detect hand and crop
        if detector is not None:
            hands, img_draw = detector.findHands(img)
            if hands: # cvzone gives bounding box
                hand = hands[0]
                x, y, w, h = hand['bbox']
                x1 = max(x - CROP_PADDING, 0) # makes bounding box bigger so full hand is shown
                y1 = max(y - CROP_PADDING, 0)
                x2 = min(x + w + CROP_PADDING, img.shape[1])
                y2 = min(y + h + CROP_PADDING, img.shape[0])
                crop = img[y1:y2, x1:x2]
            else:
                crop = img # if no hand is found it shows the full frame
                img_draw = img
        else:
            crop = img
            img_draw = img

        crop_resized = cv2.resize(crop, TARGET_SIZE) #crops to model input size
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)

        event_mask = np.zeros_like(crop_resized)
        if prev_gray is not None:
            _, event_mask = find_events(gray, prev_gray, threshold=EVENT_THRESHOLD)
        overlay = cv2.addWeighted(crop_resized, 0.7, event_mask, 0.3, 0)
        prev_gray = gray

        if SHOW_WINDOWS:
            cv2.imshow("Original", img_draw)
            cv2.imshow("ASL Crop", overlay)

        key = cv2.waitKey(1) & 0xFF

        if key in KEY_TO_LABEL:
            label = KEY_TO_LABEL[key] #saves image to key pressed
            save_labeled_frame(crop_resized, label)

        elif key == 27: # esc to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()