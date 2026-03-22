#loads trained asl model into backend 
# reads opencv image frames 

from pathlib import Path
print("asl_model: starting imports")

print("asl_model: importing cv2")
import cv2

print("asl_model: importing numpy")
import numpy as np

print("asl_model: importing torch")
import torch

print("asl_model: importing torch.nn")
import torch.nn as nn

#image normalization values
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP_PAD = 40


def pick_device() -> torch.device:
    return torch.device("cpu") # using cpu 


def build_model(num_classes: int) -> nn.Module:
    print("asl_model: importing torchvision.models")
    from torchvision import models

    print("asl_model: building MobileNetV3 model") #creating moel 
    m = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    
    #debug prints 
    print("model type:", type(m))
    print("has classifier:", hasattr(m, "classifier"))
    print("classifier object:", m.classifier)
    print("classifier type:", type(m.classifier))
    print("classifier length:", len(m.classifier))

    
    print("asl_model: reading in_features")
    in_feats = m.classifier[-1].in_features #input features in final layer
    print("asl_model: in_features =", in_feats)

    print("asl_model: replacing final layer")
    m.classifier[-1] = nn.Linear(in_feats, num_classes) #replace final layer
    print("asl_model: final layer replaced")

    return m

#same function. to convert open cv into a tensor 
def preprocess_crop(bgr_crop: np.ndarray, size: int) -> torch.Tensor:
    img = cv2.resize(bgr_crop, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)


def hand_bbox_from_landmarks(results, width: int, height: int, pad: int):
    if not results.multi_hand_landmarks:
        return None

    lm = results.multi_hand_landmarks[0]
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]

    x0 = min(xs) * width
    y0 = min(ys) * height
    x1 = max(xs) * width
    y1 = max(ys) * height

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = x1 - x0
    h = y1 - y0
    side = max(w, h) + 2 * pad

    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(width, cx + side / 2))
    y1 = int(min(height, cy + side / 2))

    if x1 <= x0 or y1 <= y0:
        return None

    return (x0, y0, x1, y1)


class ASLPredictor:
    def __init__(self, ckpt_path: str):
        print("ASLPredictor: init start")
        self.ckpt_path = ckpt_path #load best model 
        self.device = pick_device() # choose device 
        print("ASLPredictor: device selected", self.device)

        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print("ASLPredictor: loading checkpoint")
        ckpt = torch.load(ckpt_path, map_location=self.device) #get best model 
        print("ASLPredictor: checkpoint type", type(ckpt))

        print("ASLPredictor: reading classes")
        self.classes = ckpt["classes"] #get class names 
        self.img_size = int(ckpt.get("img_size", 224)) #get image size 
        print("ASLPredictor: classes loaded", self.classes)
        print("ASLPredictor: image size", self.img_size)

        print("ASLPredictor: building model")
        self.model = build_model(num_classes=len(self.classes)) #build model 

        print("ASLPredictor: moving model to device")
        self.model = self.model.to(self.device)
        print("ASLPredictor: model moved to device")

        print("ASLPredictor: loading state dict")
        self.model.load_state_dict(ckpt["state_dict"]) #load the trained weights for model 
        self.model.eval()
        print("ASLPredictor: model ready")

        print("asl_model: importing mediapipe")
        import mediapipe as mp
        print("asl_model: mediapipe imported")

        self.mp_hands = mp.solutions.hands #set up hand tracking 
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("ASLPredictor: hands ready")

    def predict_from_frame(self, frame_bgr: np.ndarray):
        # run prediction on single opencv frame, return result, repeat
        frame_bgr = cv2.flip(frame_bgr, 1) # mirror frame 

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        bbox = hand_bbox_from_landmarks(results, width, height, pad=CROP_PAD) #detect a hand and build a box around 

        if bbox is not None:
            x0, y0, x1, y1 = bbox
            crop = frame_bgr[y0:y1, x0:x1].copy()
        else:
            side = min(height, width)
            y0 = (height - side) // 2
            x0 = (width - side) // 2
            crop = frame_bgr[y0:y0 + side, x0:x0 + side].copy()

        x = preprocess_crop(crop, self.img_size).to(self.device) #preprocess crop to match models expected size 

        with torch.no_grad(): #run model and covert logits into probablilites 
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # pick class with highest probablility 
        pred_idx = int(probs.argmax()) 
        pred_lbl = self.classes[pred_idx]
        pred_conf = float(probs[pred_idx])

        return { # show results on front end 
            "label": pred_lbl, # letter 
            "confidence": pred_conf, #confidence %
            "hand_detected": bbox is not None, 
            "crop_bgr": crop,
        }

    def close(self):
        self.hands.close()