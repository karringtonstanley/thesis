###
###eval.py 

# Run:
    # python eval.py --data data_active --ckpt models/min_asl.pth --bs 32

# prints:
# 
# per-class accuracy
# confusion matrix
####

from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# use CPU 
def device():
    return torch.device("cpu")
# resizing the images again into the same imagenet values to match training setup
def tfms_val(sz: int):
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
# the same MobileNetV3 model 
def build_model(num_classes: int, imgnet_weights=True):
    if imgnet_weights:
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
    else:
        m = models.mobilenet_v3_small(weights=None)
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = torch.nn.Linear(in_feats, num_classes) #replacing last layer again
    return m
# -----------------------------------

# main function: loads the datasets and mobilenetv3 model, prints the evalutation results
def main():
    ap = argparse.ArgumentParser() # command line arguments if needed 
    ap.add_argument("--data", default="data_active", help="root folder with class subfolders")
    ap.add_argument("--ckpt", default="models/min_asl.pth", help="checkpoint path")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    print("Starting evaluation...")
    dev = device()
    print(f"Using device: {dev}")
    data_root = Path(args.data)
    ckpt_path = Path(args.ckpt)
    print(f"Dataset path: {data_root}")
    print(f"Checkpoint path: {ckpt_path}")

    print("Loading dataset...")
    # loads the datasets from the image folder
    ds = datasets.ImageFolder(str(data_root), transform=tfms_val(args.img_size))
    classes = ds.classes
    print(f"Loaded dataset with {len(ds)} images across {len(classes)} classes")
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0)
    print(f"Dataloader ready with batch size {args.bs} and {len(dl)} total batches")

    print("Building model...")
    # build model 
    model = build_model(num_classes=len(classes)).to(dev)
    print("Model built")
    print("Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=dev)
    print("Checkpoint loaded")
    # handle both styles: {"state_dict": ...} or raw state dict
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    print("Loading state dict into model...")
    model.load_state_dict(state_dict)
    model.eval() #eval mode 
    print("Model ready for evaluation")

    # eval loop
    
    total, correct = 0, 0 # total predicitons, total correct predictions
    cm = np.zeros((len(classes), len(classes)), dtype=int) #confusion matrix, rows = true class columns = prdicted class
    per_class_total = np.zeros(len(classes), dtype=int) 
    per_class_correct = np.zeros(len(classes), dtype=int)

    # no gradients since in eval mode 
    with torch.no_grad():
        print(f"Starting evaluation loop with {len(dl)} batches...")
        for batch_idx, (xb, yb) in enumerate(dl, start=1):
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb) # get raw scores 
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(dl):
                print(f"  processed eval batch {batch_idx}/{len(dl)}")
            preds = logits.argmax(1) # choose class with higher score for all images 
            
            #accuracy counts
            total += yb.size(0) 
            correct += (preds == yb).sum().item()
            # update the confusion matrix
            for p, y in zip(preds.cpu().tolist(), yb.cpu().tolist()):
                cm[y, p] += 1
                per_class_total[y] += 1
                if p == y:
                    per_class_correct[y] += 1

    print("Evaluation loop complete")
    # printing the results
    acc = correct / max(1, total) #overall accuracy
    print("Classes:", classes)
    print(f"Overall accuracy: {acc:.3f}")
    for i, cls in enumerate(classes):
        pacc = (per_class_correct[i] / per_class_total[i]) if per_class_total[i] else 0.0
        print(f"  {cls}: {pacc:.3f}  ({per_class_correct[i]}/{per_class_total[i]})")

    print("\nConfusion matrix (rows=true, cols=pred):")
    # print confusion matrix
    header = "     " + " ".join(f"{c:>4}" for c in classes)
    print(header)
    for i, cls in enumerate(classes):
        row = " ".join(f"{n:>4}" for n in cm[i])
        print(f"{cls:>4} {row}")

if __name__ == "__main__":
    main()