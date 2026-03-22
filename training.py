# training.py
# MobileNetV3 transfer learning training for A/B/C etc
# reads image folders using torchvision

#Output: min_asl.path - best checkpoint

import json, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

# 
# Configuration board
# 
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR   = PROJECT_ROOT / "data_active"   # folder containing A,B,C etc
MODEL_DIR  = PROJECT_ROOT / "models"        # saves trained model in models
IMG_SIZE   = 224             # 224 is standard for imagenet
EPOCHS     = 12              # # of runs through dataset
BATCH_SIZE = 16              # images per batch
LR         = 5e-4            # AdamW learning rate
VAL_SPLIT  = 0.2             # % for validation split (80% trainingset/20%validationset)
AUG        = True            # data augmentation on train set
SAVE_PATH  = "min_asl.pth"
SEED       = 42              # randomness
USE_PRETRAINED = True       # set True to use ImageNet weights; 
FORCE_CPU  = False           # set True if MPS/GPU causes startup or training issues

# Pytorch device
#option 1 CPU
#option 2 apple gpu
#option 3 nvidia gpu
def device():
    if FORCE_CPU:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# image transform pipeline for training set
# resizes each image, applies augmentation
# converts image into a tensor and normalizes
def tfms_train(sz: int, aug: bool):
    ops = [transforms.Resize((sz, sz))] # resizes image
    if aug:
        ops += [
            transforms.RandomRotation(degrees=10),
            # flips and turns image 
        ]
    ops += [
        transforms.ToTensor(), #changes image into a tensor; number values, this is what mobilenet was orginally trainedo n
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
    ]
    return transforms.Compose(ops) # put all transforms into one group

# image transform pipeline for validaiton set
# resized and normalized only
# no augmentation
def tfms_val(sz: int):
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ), 
    ])

#
# Model: MobileNetV3-Small being built
# uses imagenet pretrained weights
# replace final layer so model predicts just ASL class
def build_model(num_classes: int):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if USE_PRETRAINED else None
    print(f"Building MobileNetV3-Small | pretrained={USE_PRETRAINED}")
    m = models.mobilenet_v3_small(weights=weights)
    in_feats = m.classifier[-1].in_features # # of input features going into last layer
    m.classifier[-1] = nn.Linear(in_feats, num_classes) # replace final layer with a layer sized for classes
    return m

# 
# One epoch run (one full pass over a dataset)
# model - new mobilenetv3, loader - loads images, dev - Apple GPU, criterion - loss funciton
# optimizer none means model only evaulates, optimizer available means training mode
def run_epoch(model, loader, dev, criterion, optimizer=None):
    is_train = optimizer is not None 
    model.train(is_train)

    phase = "train" if is_train else "val"
    total, correct, loss_sum = 0, 0, 0.0 # metrics for correct predictions and total loss 
    print(f"Starting {phase} epoch with {len(loader)} batches...", flush=True)
    for batch_idx, (xb, yb) in enumerate(loader, start=1):
        xb, yb = xb.to(dev), yb.to(dev) # moving dataset images and labels to device 

        if batch_idx % 10 == 0 or batch_idx == 1 or batch_idx == len(loader): #logging for epoch process
            print(f"  {phase} batch {batch_idx}/{len(loader)}", flush=True)

        if is_train:
            optimizer.zero_grad() #clear old gradients before caculating new ones

        logits = model(xb) #running batch thru model to get predicitons
        loss = criterion(logits, yb) # compare predictions/ to correct answers

        if is_train: #updating model weights
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * xb.size(0) #accumulates the loss
        preds = logits.argmax(1) #picks class with highest prediction score
        correct += (preds == yb).sum().item() #accumulates correct predictions
        total += xb.size(0)

    avg_loss = loss_sum / max(1, total) #avg loss and accuracy for a epoch
    acc = correct / max(1, total)
    return acc, avg_loss

# 
# Main training funciotn 
# whole training flow, trains multiple epochs, saves best ckpt
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    Path(MODEL_DIR).mkdir(exist_ok=True)

    # 1) Build a plain dataset to read classes 
    plain_ds = datasets.ImageFolder(DATA_DIR)
    classes = plain_ds.classes  # e.g., ['A','B','C'] # class forlder names

    #computing train/val split
    n_val   = max(1, int(len(plain_ds) * VAL_SPLIT))
    n_train = len(plain_ds) - n_val
    gen = torch.Generator().manual_seed(SEED)
    # random_split on a range of indices for determinism
    train_idx, val_idx = random_split(range(len(plain_ds)), [n_train, n_val], generator=gen)

    # 2) Build base datasets WITH transforms
    train_tf = tfms_train(IMG_SIZE, AUG) #training 
    val_tf   = tfms_val(IMG_SIZE) #validaiton
    base_train = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    base_val   = datasets.ImageFolder(DATA_DIR, transform=val_tf)

    # 3) Apply the split indices to those base datasets
    train_ds = Subset(base_train, train_idx.indices)
    val_ds   = Subset(base_val,   val_idx.indices)

    # 4) dataloaders, feed images accor. to batch size  
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Using dataset folder: {DATA_DIR}")
    print(f"Saving models to: {MODEL_DIR}")
    print(f"Classes = {classes} | Training={len(train_ds)} Validation={len(val_ds)}")

    # 5) Select device and build model 
    dev = device()
    print(f"Selected device: {dev}")
    print("About to build model...")
    model = build_model(num_classes=len(classes))
    print("Model built. Moving model to device...")
    model = model.to(dev)
    print("Model moved to device.")
    criterion = nn.CrossEntropyLoss() #cross entropy loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR) #uses adamw optimizer

    # 6) the best ckpt will be based on highest val accuracy 
    best_accuracy = 0.0
    best_checkpoint = Path(MODEL_DIR) / SAVE_PATH

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====", flush=True)
        tr_acc, tr_loss = run_epoch(model, train_dl, dev, criterion, optimizer) # training set
        va_acc, va_loss = run_epoch(model, val_dl,   dev, criterion, optimizer=None) # eval set

        print(f"epoch={epoch:02d} | tr_acc={tr_acc:.3f} val_acc={va_acc:.3f} "
              f"tr_loss={tr_loss:.3f} val_loss={va_loss:.3f}")

        if va_acc > best_accuracy:
            best_accuracy = va_acc
            # Save best performing model  
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "img_size": IMG_SIZE
            }, best_checkpoint)
            #  save labels for webreference
            with open(Path(MODEL_DIR) / "web_labels_for_ref.json", "w") as f:
                json.dump({"classes": classes}, f)
            print(f" -> saved best to {best_checkpoint} (val_acc={best_accuracy:.3f})")

    print("Finalized. Best validation accuracy:", best_accuracy)

if __name__ == "__main__":
    main()
    


    


