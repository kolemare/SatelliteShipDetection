#!/usr/bin/env python3
import os
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# --- plotting (headless-safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Dataset wrapper (PNG parsing)
# ----------------------------
class ShipsDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        # recurse and accept .png/.PNG
        self.samples = sorted(list(self.root.rglob("*.png")) + list(self.root.rglob("*.PNG")))
        if not self.samples:
            raise RuntimeError(
                f"No PNG files found under: {self.root.resolve()}\n"
                f"Check convnext_base_config.json['dataset_path'] and folder structure."
            )
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        # filename format: 0__sceneid__lon_lat.png
        try:
            label = int(path.name.split("__")[0])
            if label not in (0, 1):
                raise ValueError
        except Exception:
            raise ValueError(f"Cannot parse 0/1 label from filename: {path.name}")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ----------------------------
# Utility
# ----------------------------
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)


def build_dataloaders(config, device):
    data_root = config["dataset_path"]

    # NO CROPPING: upscale only; keep spatial content intact
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),   # set p=0.0 if you want zero "erasing"
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = ShipsDataset(data_root, transform=train_tfms)
    n_total = len(full_dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(full_dataset, [n_train, n_val, n_test])

    # val/test should not use strong augmentation
    val_set.dataset.transform = test_tfms
    test_set.dataset.transform = test_tfms

    pin = (device.type == "cuda")
    dl_train = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                          num_workers=config["num_workers"], pin_memory=pin,
                          worker_init_fn=worker_init_fn, persistent_workers=(config["num_workers"] > 0))
    dl_val = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"], pin_memory=pin,
                        worker_init_fn=worker_init_fn, persistent_workers=(config["num_workers"] > 0))
    dl_test = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False,
                         num_workers=config["num_workers"], pin_memory=pin,
                         worker_init_fn=worker_init_fn, persistent_workers=(config["num_workers"] > 0))

    print(f"[INFO] Samples -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    return dl_train, dl_val, dl_test


# ----------------------------
# Model building
# ----------------------------
def build_model(num_classes=2):
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    # replace classifier
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


# ----------------------------
# Training & eval with per-batch logs (AMP on CUDA)
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch_idx, total_epochs):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Train {epoch_idx+1}/{total_epochs}", ncols=110)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, device=device)

        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        _, preds = outputs.max(1)
        total_correct += preds.eq(labels).sum().item()
        total += batch_size

        running_loss = total_loss / total
        running_acc = total_correct / total
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}", lr=f"{current_lr:.2e}")

    return total_loss / max(1, total), total_correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch_idx, total_epochs, split="Val"):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"{split}  {epoch_idx+1}/{total_epochs}", ncols=110)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, device=device)

        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        _, preds = outputs.max(1)
        total_correct += preds.eq(labels).sum().item()
        total += batch_size

        running_loss = total_loss / total
        running_acc = total_correct / total
        pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

    return total_loss / max(1, total), total_correct / max(1, total)


# ----------------------------
# Plotting
# ----------------------------
def plot_training_curves(metrics, out_dir: str, total_epochs: int):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(metrics["train_acc"]) + 1)

    plt.figure(figsize=(10, 7))

    # Accuracy subplot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, metrics["train_acc"], label="Train Acc")
    plt.plot(epochs, metrics["val_acc"], label="Val Acc")
    plt.ylabel("Accuracy")
    plt.xlim(1, total_epochs)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.title("Training Curves")

    # Loss subplot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(1, total_epochs)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    out_path = os.path.join(out_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved training curves to: {out_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    with open("convnext_base_config.json", "r") as f:
        config = json.load(f)

    torch.manual_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA is available!")
        torch.backends.cudnn.benchmark = True  # speed on fixed-size inputs

    dl_train, dl_val, dl_test = build_dataloaders(config, device)

    model = build_model(num_classes=2)
    if device.type == "cuda":
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    # freeze backbone initially
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    metrics = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(config["epochs"]):
        if epoch == config["warmup_epochs"]:
            # unfreeze all
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"] - epoch)
            print(f"[INFO] Unfroze all layers at epoch {epoch}")

        train_loss, train_acc = train_one_epoch(model, dl_train, criterion, optimizer, device, scaler, epoch, config["epochs"])
        val_loss, val_acc = evaluate(model, dl_val, criterion, device, epoch, config["epochs"], split="Val")
        scheduler.step()

        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{config['epochs']} "
              f"Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f} | "
              f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("pretrained", exist_ok=True)
            torch.save(model.state_dict(), "pretrained/convnext_ships.pt")
            with open("pretrained/metrics.json", "w") as f:
                json.dump(metrics, f)

    # Final test evaluation
    test_loss, test_acc = evaluate(model, dl_test, criterion, device, config["epochs"]-1, config["epochs"], split="Test")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Plot curves at the end
    plot_training_curves(metrics, out_dir="pretrained", total_epochs=config["epochs"])


if __name__ == "__main__":
    main()
