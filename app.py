"""
Brain Tumor Classification — Flask Web Application
====================================================
Upload MRI images and get predictions with Grad-CAM visualization.
Supports ResNet50 and EfficientNet-B0 architectures.
"""

import os
import io
import base64
from PIL import Image

import torch
from torchvision import transforms, models
import torch.nn as nn
import requests as http_requests

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from grad_cam import generate_gradcam_b64

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Vercel frontend

# ─── Configuration ───────────────────────────────────────────────────────────
RESNET_MODEL_PATH    = os.path.join('models', 'best_model.pth')
EFFICIENT_MODEL_PATH = os.path.join('models', 'best_model_efficientnet.pth')

# Hugging Face download URLs (set via env vars or defaults)
HF_EFFICIENTNET_URL = os.environ.get(
    'HF_EFFICIENTNET_URL',
    'https://huggingface.co/abinbinoy/bt-scanner-models/resolve/main/best_model_efficientnet.pth'
)
HF_RESNET_URL = os.environ.get(
    'HF_RESNET_URL',
    'https://huggingface.co/abinbinoy/bt-scanner-models/resolve/main/best_model.pth'
)

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_LABELS = {
    'glioma':      'Glioma Tumor',
    'meningioma':  'Meningioma Tumor',
    'notumor':     'No Tumor',
    'pituitary':   'Pituitary Tumor'
}
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224

# Preprocessing shared by both models (ImageNet stats)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Model Builders ─────────────────────────────────────────────────────────

def build_resnet50():
    """Build the ResNet50 classification head matching training."""
    m = models.resnet50(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, 4)
    )
    return m


def build_efficientnet_b0():
    """Build the EfficientNet-B0 classification head (must match training arch)."""
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features   # 1280
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, 4),
    )
    return m


def download_model(url, dest):
    """Download model weights from Hugging Face Hub if not present locally."""
    if os.path.exists(dest):
        return
    print(f"[*] Downloading model from {url} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = http_requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    downloaded = 0
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r[*] Downloading... {pct}%", end='', flush=True)
    print(f"\n[+] Model saved to {dest}  ({downloaded / 1024 / 1024:.1f} MB)")


def load_model(model_obj, path, label):
    """Load checkpoint weights into model_obj if the file exists."""
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)
        model_obj.load_state_dict(state)
        val_acc = checkpoint.get('val_acc', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
        print(f"[+] {label} loaded from {path}  (val_acc={val_acc})")
    else:
        print(f"[!] WARNING: No model found at {path} — {label} will return random predictions.")
    model_obj = model_obj.to(DEVICE)
    model_obj.eval()
    return model_obj


# ─── Download & Load Models at Startup ───────────────────────────────────────
# Download from Hugging Face if .pth files are not present (e.g. on Render)
download_model(HF_EFFICIENTNET_URL, EFFICIENT_MODEL_PATH)

efficientnet_model = load_model(build_efficientnet_b0(), EFFICIENT_MODEL_PATH, 'EfficientNet-B0')

# ResNet50 is large (~270MB) — only load if the file exists locally
# On Render free tier (512MB RAM), loading both will exceed memory
if os.path.exists(RESNET_MODEL_PATH):
    resnet_model = load_model(build_resnet50(), RESNET_MODEL_PATH, 'ResNet50')
else:
    resnet_model = None
    print("[!] ResNet50 skipped (not available — free tier memory optimization)")

MODELS = {
    'efficientnet': (efficientnet_model, 'efficientnet'),
}
if resnet_model is not None:
    MODELS['resnet'] = (resnet_model, 'resnet')


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    algorithm = request.form.get('algorithm', 'resnet')
    if algorithm not in MODELS:
        return jsonify({'error': f'Unknown algorithm: {algorithm}'}), 400

    model, algo_key = MODELS[algorithm]

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Original image → base64
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        original_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Prediction + Grad-CAM
        result = generate_gradcam_b64(model, image, preprocess, DEVICE, CLASSES, algorithm=algo_key)

        algo_labels = {
            'resnet':      'ResNet50',
            'efficientnet': 'EfficientNet-B0',
        }

        return jsonify({
            'success':         True,
            'predicted_class': CLASS_LABELS.get(result['predicted_class'], result['predicted_class']),
            'predicted_key':   result['predicted_class'],
            'confidence':      round(result['confidence'] * 100, 2),
            'all_probs': {
                CLASS_LABELS.get(k, k): round(v * 100, 2)
                for k, v in result['all_probs'].items()
            },
            'original_image':  original_b64,
            'heatmap_image':   result['heatmap_b64'],
            'algorithm_used':  algo_labels.get(algorithm, algorithm),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
