"""
Grad-CAM Visualization — ResNet50 & EfficientNet-B0
======================================================
Generates class activation heatmaps to show which MRI regions
the model focuses on for its prediction.
"""

import io
import base64
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm


class GradCAM:
    """Gradient-weighted Class Activation Mapping (generic — works with any layer)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        class_probs = probs[0].detach().cpu().numpy()
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()

        gradients = self.gradients[0]   # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        weights = gradients.mean(dim=(1, 2))
        heatmap = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
        heatmap = F.relu(heatmap).cpu().numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap, target_class, class_probs

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def _get_target_layer(model, algorithm: str):
    """Return the appropriate Grad-CAM target layer for each architecture."""
    if algorithm == 'efficientnet':
        # EfficientNet-B0: last conv block inside features
        return model.features[-1]
    else:
        # ResNet50 default
        return model.layer4


def create_heatmap_overlay(original_image, heatmap, alpha=0.5):
    img_array = np.array(original_image.convert('RGB'))
    h, w = img_array.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255)).resize((w, h), Image.BILINEAR)
    ) / 255.0
    colormap = cm.get_cmap('jet')
    heatmap_colored = (colormap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heatmap_colored + (1 - alpha) * img_array).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_gradcam_b64(model, image_pil, transform, device, classes, algorithm='resnet'):
    import gc
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    target_layer = _get_target_layer(model, algorithm)
    grad_cam = GradCAM(model, target_layer)
    heatmap, pred_idx, class_probs = grad_cam.generate(input_tensor)
    grad_cam.remove_hooks()

    # Free gradient memory immediately
    del input_tensor
    gc.collect()

    heatmap_b64 = create_heatmap_overlay(image_pil, heatmap, alpha=0.5)

    del heatmap
    gc.collect()

    return {
        'heatmap_b64': heatmap_b64,
        'predicted_class': classes[pred_idx],
        'predicted_index': pred_idx,
        'confidence': float(class_probs[pred_idx]),
        'all_probs': {classes[i]: float(class_probs[i]) for i in range(len(classes))},
    }


def predict_only(model, image_pil, transform, device, classes):
    """Lightweight prediction without Grad-CAM — uses minimal memory."""
    import gc
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        class_probs = probs[0].cpu().numpy()
        pred_idx = int(probs.argmax(dim=1).item())

    del input_tensor, output, probs
    gc.collect()

    return {
        'heatmap_b64': None,
        'predicted_class': classes[pred_idx],
        'predicted_index': pred_idx,
        'confidence': float(class_probs[pred_idx]),
        'all_probs': {classes[i]: float(class_probs[i]) for i in range(len(classes))},
    }

