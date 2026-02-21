""""
eval.py 

Run:
    python eval.py --data data_active --ckpt models/min_asl.pth --bs 32

prints:
- Overall accuracy
- Per-class accuracy
- Confusion matrix
"""""

from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------- using Apple MPS GPU for computing, if that doesnt work uses NVIDIA GPU, if that doesnt work uses CPU ----------
def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
# resizing the models size, translating to tensor, normalize using imagenet mean/std
def tfms_val(sz: int):
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
# the use of MobileNetV3 as our model 
def build_model(num_classes: int, imgnet_weights=True):
    if imgnet_weights:
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
    else:
        m = models.mobilenet_v3_small(weights=None)
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = torch.nn.Linear(in_feats, num_classes)
    return m
# -----------------------------------

# main function: loads the datasets and mobilenetv3 model, prints the evalutation results
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_active", help="root folder with class subfolders")
    ap.add_argument("--ckpt", default="models/min_asl.pth", help="checkpoint path")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    dev = device()
    data_root = Path(args.data)
    ckpt_path = Path(args.ckpt)

    # loads the datasets from the image folder
    ds = datasets.ImageFolder(str(data_root), transform=tfms_val(args.img_size))
    classes = ds.classes
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0)

    # model + load weights
    model = build_model(num_classes=len(classes)).to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev)
    # handle both styles: {"state_dict": ...} or raw state dict
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # eval loop
    # counts total number evaluted and total numer correct
    # cm creates the confusion matrix
    # counts total number in the class and total number correct
    total, correct = 0, 0
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    per_class_total = np.zeros(len(classes), dtype=int)
    per_class_correct = np.zeros(len(classes), dtype=int)

    # no gradients
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb) # raw scores 
            preds = logits.argmax(1) # predicts the class index per example 

            total += yb.size(0)
            correct += (preds == yb).sum().item()
            # update the confusion matrix
            for p, y in zip(preds.cpu().tolist(), yb.cpu().tolist()):
                cm[y, p] += 1
                per_class_total[y] += 1
                if p == y:
                    per_class_correct[y] += 1

    # printing the results
    acc = correct / max(1, total)
    print("Classes:", classes)
    print(f"Overall accuracy: {acc:.3f}")
    for i, cls in enumerate(classes):
        pacc = (per_class_correct[i] / per_class_total[i]) if per_class_total[i] else 0.0
        print(f"  {cls}: {pacc:.3f}  ({per_class_correct[i]}/{per_class_total[i]})")

    print("\nConfusion matrix (rows=true, cols=pred):")
    # pretty print small matrices
    header = "     " + " ".join(f"{c:>4}" for c in classes)
    print(header)
    for i, cls in enumerate(classes):
        row = " ".join(f"{n:>4}" for n in cm[i])
        print(f"{cls:>4} {row}")

if __name__ == "__main__":
    main()