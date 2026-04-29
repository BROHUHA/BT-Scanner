# BT-Scanner

A deep learning-powered Brain Tumor classification system using ResNet50 and EfficientNet-B0 with Grad-CAM explainability, served through a premium glassmorphic Flask web interface.

## Features

- **Multi-Model Classification** — ResNet50 & EfficientNet-B0 for classifying MRI scans into 4 categories: Glioma, Meningioma, Pituitary, and No Tumor
- **Grad-CAM Visualization** — Visual explainability highlighting diagnostic regions
- **Modern Web UI** — Dark glassmorphic interface with real-time predictions
- **GPU Accelerated** — Optimized for NVIDIA RTX hardware with mixed-precision training

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## Dataset

Place MRI images in `Training/` and `Testing/` directories with subfolders: `glioma`, `meningioma`, `notumor`, `pituitary`.

## Training

```bash
python train.py                # ResNet50
python train_efficientnet.py   # EfficientNet-B0
```
