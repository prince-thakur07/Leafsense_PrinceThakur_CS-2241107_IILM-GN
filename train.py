"""
Train EfficientNet-B0 for binary leaf disease classification (Healthy vs Diseased)
using the LeafSense binary dataset. Folder names containing "healthy" -> Healthy (1), else -> Diseased (0).
Build the dataset first: python prepare_leafsense_dataset.py
Saves weights to efficientnet_plantdoc.pth for use with app.py.
"""
import os
import argparse
import random
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from timm import create_model

# Default dataset path: LeafSense binary dataset (build with prepare_leafsense_dataset.py)
DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leafsense_binary_dataset")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # 0 = Diseased, 1 = Healthy (must match app.py)


def is_healthy_class(folder_name: str) -> bool:
    """LeafSense binary: folder 'Healthy' -> 1; 'Diseased' or any other -> 0."""
    return "healthy" in folder_name.lower()


class LeafSenseBinary(Dataset):
    """Load LeafSense binary dataset: 0 = Diseased, 1 = Healthy (two folders)."""

    def __init__(self, root_dir: str, transform=None, max_per_class: int = None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []  # list of (path, label)

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = 1 if is_healthy_class(class_dir.name) else 0
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                for path in class_dir.glob(ext):
                    self.samples.append((str(path), label))

        # Subsample to cap each binary class (faster training)
        if max_per_class and max_per_class > 0:
            by_label = {0: [], 1: []}
            for path, label in self.samples:
                by_label[label].append((path, label))
            if not by_label[0] or not by_label[1]:
                raise ValueError(
                    "Dataset must contain both Healthy and Diseased images. "
                    f"Found: Diseased={len(by_label[0])}, Healthy={len(by_label[1])}."
                )
            random.shuffle(by_label[0])
            random.shuffle(by_label[1])
            self.samples = (by_label[0][:max_per_class] + by_label[1][:max_per_class])
            random.shuffle(self.samples)
        else:
            labels_present = {label for _, label in self.samples}
            if labels_present != {0, 1}:
                raise ValueError(
                    "Dataset must contain both Healthy and Diseased images. "
                    f"Found classes: {labels_present}."
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, _retry=False):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Skipping corrupt or unreadable image: {path} ({e})")
            if _retry:
                raise RuntimeError(f"Failed to load image: {path}") from e
            return self.__getitem__((idx + 1) % len(self.samples), _retry=True)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    """Same preprocessing as app.py for consistency."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = logits.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    n = len(loader)
    return total_loss / n, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        _, pred = logits.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    n = len(loader)
    return total_loss / n, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on LeafSense binary dataset")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATASET,
        help="Path to LeafSense binary dataset root (default: leafsense_binary_dataset)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max-per-class", type=int, default=500, help="Max samples per binary class (0 = use all). Default 500 for faster training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (0–1)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 for Windows safety)")
    parser.add_argument("--save", type=str, default="efficientnet_plantdoc.pth", help="Output model path")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs (0 = only best)")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume (e.g. checkpoint_latest.pth)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.isdir(args.data):
        print(f"Error: Dataset path not found: {args.data}")
        return 1

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.data}")

    transform = get_transforms()
    full_dataset = LeafSenseBinary(args.data, transform=transform, max_per_class=args.max_per_class or None)
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    model = create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    start_epoch = 1
    best_val_acc = 0.0
    if args.resume:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(base_dir, args.resume)
        if os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print(f"Note: Could not load optimizer state ({e}). Using fresh optimizer.")
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            print(f"Resumed from {resume_path} at epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")
        else:
            print(f"Warning: Resume path not found: {resume_path}. Starting from epoch 1.")

    print(f"Train samples: {n_train}, Val samples: {n_val}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}, Max per class: {args.max_per_class or 'all'}")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch:3d}  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(base_dir, args.save)
            torch.save(model.state_dict(), save_path)
            print(f"         -> Saved best model to {save_path}")
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            ckpt_path = os.path.join(base_dir, "checkpoint_latest.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            }, ckpt_path)
            print(f"         -> Saved checkpoint to {ckpt_path}")

    print("-" * 60)
    print(f"Training done. Best validation accuracy: {best_val_acc:.2f}%")
    return 0


if __name__ == "__main__":
    exit(main())
