"""
Brain Tumor Classification — EfficientNet-B0 Training Pipeline (v2)
====================================================================
Fine-tunes a pretrained EfficientNet-B0 on a 4-class MRI dataset:
  Glioma | Meningioma | No Tumor | Pituitary

Key improvements over v1:
  - Unfreeze more feature blocks (from block 4 onwards) so the model
    can adapt low-level features to MRI texture which differs from ImageNet.
  - Richer classifier head: Dropout → 512 → ReLU → Dropout → 4
    (matches ResNet50 head capacity).
  - No label smoothing — it was artificially capping confidence and
    hurting accuracy on this small medical dataset.
  - Lower LR (1e-4) with ReduceLROnPlateau for more stable convergence.
  - Gradient clipping to prevent training instabilities.

Saves to: models/best_model_efficientnet.pth
Optimized for NVIDIA RTX 2050 (~4 GB VRAM) with mixed-precision training.
"""

import os
import sys
import json
import time
import copy
import argparse
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


# ─── Configuration ────────────────────────────────────────────────────────────

CLASSES            = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES        = 4
IMG_SIZE           = 224
BATCH_SIZE         = 16       # smaller batch → more gradient updates per epoch
NUM_WORKERS        = 0
LEARNING_RATE      = 1e-4     # lower LR for stable fine-tuning
WEIGHT_DECAY       = 1e-5
MAX_EPOCHS         = 30
PATIENCE           = 7        # early stopping patience
SCHEDULER_PATIENCE = 3
VAL_SPLIT          = 0.15

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Data Transforms ──────────────────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Model ────────────────────────────────────────────────────────────────────

def create_efficientnet(num_classes=NUM_CLASSES, pretrained=True):
    """
    Build EfficientNet-B0 with a richer classifier head.

    Freezing strategy (v2):
      - Freeze features[0..3]  (stem + first 3 MBConv blocks)
      - Fine-tune features[4..8]  (last 5 blocks — lets the network
        learn MRI-specific textures that differ heavily from ImageNet)
      - Full training of the new classifier head

    Classifier head:
      Dropout(0.3) → Linear(1280, 512) → ReLU → Dropout(0.2) → Linear(512, 4)
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.efficientnet_b0(weights=weights)

    # Freeze only the earliest blocks
    for i in range(4):  # features[0] … features[3]
        for param in model.features[i].parameters():
            param.requires_grad = False

    # Fine-tune the rest
    for i in range(4, len(model.features)):
        for param in model.features[i].parameters():
            param.requires_grad = True

    # Richer classifier head (matches ResNet50 head capacity)
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )

    return model


# ─── Dataset Helpers ──────────────────────────────────────────────────────────

class TransformSubset(torch.utils.data.Dataset):
    """Apply different transforms to a Subset (used for val split)."""
    def __init__(self, subset, transform, original_dataset):
        self.subset           = subset
        self.transform        = transform
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        original_idx = self.subset.indices[idx]
        path, target = self.original_dataset.samples[original_idx]
        image = self.original_dataset.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, target


def load_data(data_dir, batch_size=BATCH_SIZE, val_split=VAL_SPLIT):
    """Load and split training data into train / val / test loaders."""
    train_dir = os.path.join(data_dir, 'Training')
    test_dir  = os.path.join(data_dir, 'Testing')

    full_train = datasets.ImageFolder(train_dir, transform=train_transforms)

    total      = len(full_train)
    val_size   = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(42)
    train_sub, val_sub = random_split(full_train, [train_size, val_size], generator=generator)

    val_dataset  = TransformSubset(val_sub, val_transforms, full_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

    train_loader = DataLoader(train_sub,    batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"[OK] Dataset loaded:")
    print(f"  Training:   {train_size} images")
    print(f"  Validation: {val_size} images")
    print(f"  Testing:    {len(test_dataset)} images")
    print(f"  Classes:    {full_train.classes}")

    return train_loader, val_loader, test_loader, full_train.classes


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, max_epochs):
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch:2d}/{max_epochs} [Train]',
                bar_format='{l_bar}{bar:30}{r_bar}', leave=False, file=sys.stdout)

    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        # Gradient clipping for training stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{correct/total:.4f}'})

    pbar.close()
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc='Eval'):
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0
    all_preds    = []
    all_labels   = []

    pbar = tqdm(loader, desc=f'         [{desc}]',
                bar_format='{l_bar}{bar:30}{r_bar}', leave=False, file=sys.stdout)

    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'acc': f'{correct/total:.4f}'})

    pbar.close()
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ─── Plot Helpers ─────────────────────────────────────────────────────────────

def plot_training_curves(history, output_dir, prefix='efficientnet_'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   'r-o', markersize=4, label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('EfficientNet-B0 — Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-o', markersize=4, label='Train Acc')
    ax2.plot(epochs, history['val_acc'],   'r-o', markersize=4, label='Val Acc')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('EfficientNet-B0 — Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}training_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Training curves saved → {out_path}")


def plot_confusion_matrix(labels, preds, classes, output_dir, prefix='efficientnet_'):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('EfficientNet-B0 — Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}confusion_matrix.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix saved → {out_path}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def train_efficientnet(data_dir='.', output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    output_model_path = os.path.join(output_dir, 'best_model_efficientnet.pth')

    print(f"\n{'='*60}")
    print(f"  Brain Tumor Classification — EfficientNet-B0 Training v2")
    print(f"  Device : {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, test_loader, classes = load_data(data_dir)

    # Model
    model = create_efficientnet(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)

    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param = sum(p.numel() for p in model.parameters())
    print(f"[OK] EfficientNet-B0: {trainable:,} trainable / {total_param:,} total parameters\n")

    # Loss — NO label smoothing (hurts accuracy on small medical datasets)
    criterion = nn.CrossEntropyLoss()

    # AdamW with low LR
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # ReduceLROnPlateau — same strategy that got ResNet50 to 96.9%
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=SCHEDULER_PATIENCE, verbose=True
    )
    scaler = GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_acc               = 0.0
    best_model_state           = None
    epochs_without_improvement = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    print(f"Starting training for up to {MAX_EPOCHS} epochs...\n")
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, epoch, MAX_EPOCHS)

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, DEVICE, desc='Val')

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:2d}/{MAX_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | {epoch_time:.1f}s", flush=True)

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':              best_val_acc,
                'classes':              classes,
            }, output_model_path)
            print(f"  [*] New best model saved → {output_model_path}  (val_acc: {best_val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"\n[!!] Early stopping after {epoch} epochs "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    total_time = time.time() - start_time
    print(f"\n{'-'*60}")
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"{'-'*60}\n")

    # ── Save history ─────────────────────────────────────────────────────────
    hist_path = os.path.join(output_dir, 'efficientnet_training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    # ── Test set evaluation ──────────────────────────────────────────────────
    print("Evaluating best model on test set...\n")
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, DEVICE, desc='Test')

    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Loss     : {test_loss:.4f}\n")

    report = classification_report(test_labels, test_preds, target_names=classes, digits=4)
    print("Classification Report:")
    print(report)

    report_path = os.path.join(output_dir, 'efficientnet_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss    : {test_loss:.4f}\n\n")
        f.write(report)

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_training_curves(history, output_dir)
    plot_confusion_matrix(test_labels, test_preds, classes, output_dir)

    print(f"\n[OK] All EfficientNet-B0 outputs saved to '{output_dir}/'")
    return model


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train EfficientNet-B0 for Brain Tumor Classification (v2)')
    parser.add_argument('--data-dir',   type=str,   default='.',          help='Root data directory')
    parser.add_argument('--output-dir', type=str,   default='models',     help='Output directory')
    parser.add_argument('--epochs',     type=int,   default=MAX_EPOCHS,   help='Max epochs')
    parser.add_argument('--batch-size', type=int,   default=BATCH_SIZE,   help='Batch size')
    parser.add_argument('--lr',         type=float, default=LEARNING_RATE, help='Initial learning rate')
    args = parser.parse_args()

    MAX_EPOCHS    = args.epochs
    BATCH_SIZE    = args.batch_size
    LEARNING_RATE = args.lr

    train_efficientnet(args.data_dir, args.output_dir)
