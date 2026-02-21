"""
Sanity check for the ASL training pipeline.

What this does:
- Lists classes and image counts from data/ (expects subfolders like A/, B/, C/)
- Loads a small batch and prints its shape
- Builds a MobileNetV3-Small model adapted to your number of classes
- Runs ONE forward pass on your fastest device (MPS on Apple GPU, else CPU/CUDA)

Run:
    python sanity_check.py

Expected:
    - Prints classes like ['A', 'B', 'C']
    - Shows per-class image counts
    - Batch shape like: torch.Size([4, 3, 224, 224])
    - Logits shape like: torch.Size([4, 3])
"""

from collections import Counter
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------- Config (keep in sync with training.py) ----------
DATA_DIR  = "data_active"   # root folder holding class subfolders (A/, B/, C/, ...)
IMG_SIZE  = 224      # image size used in your training
BATCH_SIZE = 4       # small batch, just for a smoke test
# ------------------------------------------------------------

def device():
    """Pick the fastest available device (MPS on Apple, then CUDA, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def tfms(sz: int):
    """
    Minimal transforms (no augmentation): resize, to tensor, normalize using
    ImageNet stats (what MobileNet expects).
    """
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std =[0.229, 0.224, 0.225],  # ImageNet std
        ),
    ])

def build_model(num_classes: int):
    """
    Load a MobileNetV3-Small pretrained on ImageNet and replace the final
    classifier layer so it outputs `num_classes` logits.
    """
    m = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = torch.nn.Linear(in_feats, num_classes)
    return m

def main():
    # 0) Basic folder checks
    root = Path(DATA_DIR)
    if not root.exists():
        print(f"ERROR: '{DATA_DIR}' not found. Create it with subfolders like A/, B/, C/.", file=sys.stderr)
        sys.exit(1)

    # 1) Build dataset (uses folder names as labels)
    ds = datasets.ImageFolder(DATA_DIR, transform=tfms(IMG_SIZE))
    classes = ds.classes  # e.g., ['A', 'B', 'C']
    if len(classes) == 0:
        print(f"ERROR: No class folders found under '{DATA_DIR}'.", file=sys.stderr)
        sys.exit(1)

    # 2) Count images per class for a quick sanity check
    counts = Counter(ds.targets)
    print("Classes:", classes)
    for idx, cls in enumerate(classes):
        print(f"  {cls}: {counts.get(idx, 0)} images")
    total_imgs = sum(counts.values())
    if total_imgs == 0:
        print("ERROR: No images found in your class folders.", file=sys.stderr)
        sys.exit(1)

    # 3) Build a tiny DataLoader and grab one batch
    bs = min(BATCH_SIZE, len(ds))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
    try:
        xb, yb = next(iter(dl))
    except StopIteration:
        print("ERROR: Could not draw a batch from the dataset.", file=sys.stderr)
        sys.exit(1)

    print("Batch shape:", xb.shape, "| Labels:", yb.tolist())

    # 4) Build model, move to device, single forward pass
    dev = device()
    print("Using device:", dev)

    model = build_model(num_classes=len(classes)).to(dev)
    model.eval()

    with torch.no_grad():
        logits = model(xb.to(dev))  # shape [batch, num_classes]
    print("Logits shape:", logits.shape)

    # 5) Quick success signal
    print(" Sanity check passed: dataset + model + forward pass look good.")

if __name__ == "__main__":
    main()