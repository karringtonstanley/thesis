# training.py
# MobileNetV3 transfer learning for A/B/C etc

import json, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

# -----------------------
# Configuration board
# -----------------------
DATA_DIR   = "data_active"   # folder containing A,B,C etc
MODEL_DIR  = "models"        # saves trained model in models
IMG_SIZE   = 224             # 224 is standard for imagenet
EPOCHS     = 12              # # of runs through dataset
BATCH_SIZE = 16              # images per batch
LR         = 5e-4            # AdamW learning rate
VAL_SPLIT  = 0.2             # % for validation split (80% trainingset/20%validationset)
AUG        = True            # data augmentation on train set
SAVE_PATH  = "min_asl.pth"
SEED       = 42              # randomness

# -----------------------
# Device picker (MPS on Apple Silicon if available)
# -----------------------
def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# -----------------------
# augments image
# -----------------------
def tfms_train(sz: int, aug: bool):
    """Train transforms: resize -> (optional) aug -> toTensor -> normalize."""
    ops = [transforms.Resize((sz, sz))]
    if aug:
        ops += [
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomRotation(degrees=10),
            # flips and turns image 
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
    ]
    return transforms.Compose(ops)

def tfms_val(sz: int):
    """Validation transforms: deterministic (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

# -----------------------
# Model: MobileNetV3-Small head swapped for #classes
#
# -----------------------
def build_model(num_classes: int):
    m = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feats, num_classes)
    return m

# -----------------------
# One epoch 
# -----------------------
def run_epoch(model, loader, dev, criterion, optimizer=None):
    is_train = optimizer is not None # if optimizer exists, then train, if not then eval
    model.train(is_train)

    total, correct, loss_sum = 0, 0, 0.0 # metrics
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)

        if is_train:
            optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        if is_train:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * xb.size(0) #accumulates the loss
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item() #accumulates correct predictions
        total += xb.size(0)

    avg_loss = loss_sum / max(1, total) #avg loss and accuracy for a epoch
    acc = correct / max(1, total)
    return acc, avg_loss

# -----------------------
# Main
# -----------------------
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    Path(MODEL_DIR).mkdir(exist_ok=True)

    # 1) Build a plain dataset to read classes 
    plain_ds = datasets.ImageFolder(DATA_DIR)
    classes = plain_ds.classes  # e.g., ['A','B','C']

    n_val   = max(1, int(len(plain_ds) * VAL_SPLIT))
    n_train = len(plain_ds) - n_val
    gen = torch.Generator().manual_seed(SEED)
    # random_split on a range of indices for determinism
    train_idx, val_idx = random_split(range(len(plain_ds)), [n_train, n_val], generator=gen)

    # 2) Build base datasets WITH transforms
    train_tf = tfms_train(IMG_SIZE, AUG)
    val_tf   = tfms_val(IMG_SIZE)
    base_train = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    base_val   = datasets.ImageFolder(DATA_DIR, transform=val_tf)

    # 3) Apply the split indices to those base datasets
    train_ds = Subset(base_train, train_idx.indices)
    val_ds   = Subset(base_val,   val_idx.indices)

    # 4) DataLoaders (num_workers=0 is safe on macOS)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Classes = {classes} | Training={len(train_ds)} Validation={len(val_ds)}")

    # 5) Model / loss / optimizer
    dev = device()
    model = build_model(num_classes=len(classes)).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 6) Train loop with best checkpoint on val accuracy
    best_accuracy = 0.0
    best_checkpoint = Path(MODEL_DIR) / SAVE_PATH

    for epoch in range(1, EPOCHS + 1):
        tr_acc, tr_loss = run_epoch(model, train_dl, dev, criterion, optimizer)
        va_acc, va_loss = run_epoch(model, val_dl,   dev, criterion, optimizer=None)

        print(f"epoch={epoch:02d} | tr_acc={tr_acc:.3f} val_acc={va_acc:.3f} "
              f"tr_loss={tr_loss:.3f} val_loss={va_loss:.3f}")

        if va_acc > best_accuracy:
            best_accuracy = va_acc
            # Save model weights + a tiny metadata dict
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "img_size": IMG_SIZE
            }, best_checkpoint)
            # Also save labels for web/demo reference
            with open(Path(MODEL_DIR) / "web_labels_for_ref.json", "w") as f:
                json.dump({"classes": classes}, f)
            print(f" -> saved best to {best_checkpoint} (val_acc={best_accuracy:.3f})")

    print("Finalized. Best validation accuracy:", best_accuracy)

if __name__ == "__main__":
    main()
    


    


