#asl cam to create datasets
#open camera and click on coinciding key for letter to add a picture for a datset


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

# compares current frame to previous one, pixels that changed are "event changes"
def find_events(cur_gray, prev_gray, threshold=15): 
    diff = cur_gray.astype(np.int16) - prev_gray.astype(np.int16)
    pos_mask = diff > threshold #pixels that are brighter are positive events
    neg_mask = diff < -threshold #pixels that are darker are negative events

    events = []
    event_mask = np.zeros((cur_gray.shape[0], cur_gray.shape[1], 3), dtype=np.uint8) # color image to show detected evetns
    event_mask[pos_mask] = (0, 255, 0) #the color green shows positive events
    event_mask[neg_mask] = (255, 0, 255) #pink is negative events

    ys_pos, xs_pos = np.where(pos_mask) # maps all positive event locations
    for x, y in zip(xs_pos, ys_pos):
        events.append({"x": int(x), "y": int(y), "type": 1})

    ys_neg, xs_neg = np.where(neg_mask) #maps all negaive event locations
    for x, y in zip(xs_neg, ys_neg):
        events.append({"x": int(x), "y": int(y), "type": -1})

    return events, event_mask

# saves gesture captures to its specific folder 
def save_labeled_frame(img, label, base_dir=None):
    if base_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #dataset folder path 
        base_dir = os.path.join(project_root, "data_active") #dataset folder
    dir_path = os.path.join(base_dir, label)
    os.makedirs(dir_path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f") #timestamp for capture
    filename = os.path.join(dir_path, f"{label}_{ts}.jpg") #file name 
    cv2.imwrite(filename, img)
    print(f"[saved] {filename}") #save image to disk 

def main(): # opens webcam, crops frames, saves labels when you press a key on keyboard
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open camera.")
        return
        # hand detector uses bounding boxes to crop the image 
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

    while True: #reads a frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        img = frame.copy()
        clean_img = frame.copy()

        # detect hand and crop around it 
        if detector is not None:
            hands, img_draw = detector.findHands(img)
            if hands: # cvzone gives bounding box
                hand = hands[0]
                x, y, w, h = hand['bbox']
                x1 = max(x - CROP_PADDING, 0) # makes bounding box bigger so full hand is shown
                y1 = max(y - CROP_PADDING, 0)
                x2 = min(x + w + CROP_PADDING, clean_img.shape[1])
                y2 = min(y + h + CROP_PADDING, clean_img.shape[0])
                crop = clean_img[y1:y2, x1:x2].copy() #crops clean image
            else:
                crop = clean_img.copy() # if no hand is found it shows the full clean frame
                img_draw = img
        else:
            crop = clean_img.copy()
            img_draw = img

        crop_resized = cv2.resize(crop, TARGET_SIZE) #crops to model input size; 224 x 224
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)

        event_mask = np.zeros_like(crop_resized)
        if prev_gray is not None:
            _, event_mask = find_events(gray, prev_gray, threshold=EVENT_THRESHOLD)
        overlay = cv2.addWeighted(crop_resized, 0.7, event_mask, 0.3, 0)
        prev_gray = gray

        if SHOW_WINDOWS: # windows for camera and crop 
            cv2.imshow("Original", img_draw)
            cv2.imshow("ASL Crop", overlay)

        key = cv2.waitKey(1) & 0xFF #keyboard input

        if key in KEY_TO_LABEL:
            label = KEY_TO_LABEL[key] # saves the clean crop, not the overlay with event colors or skeleton
            save_labeled_frame(crop_resized, label)

        elif key == 27: # esc to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()