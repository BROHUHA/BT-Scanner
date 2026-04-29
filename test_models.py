"""Quick test: send the same image to both models and print results."""
import requests

img_path = r"E:\Brain Tumor\Testing\glioma\Te-gl_1.jpg"

# --- ResNet50 ---
with open(img_path, "rb") as f:
    resp = requests.post(
        "http://127.0.0.1:5000/predict",
        files={"file": f},
        data={"algorithm": "resnet"},
    )
r = resp.json()
print("=== ResNet50 (glioma image) ===")
print(f"  Predicted : {r['predicted_class']}")
print(f"  Confidence: {r['confidence']}%")
print(f"  Algorithm : {r['algorithm_used']}")
for k, v in r["all_probs"].items():
    print(f"    {k}: {v}%")

print()

# --- EfficientNet-B0 ---
with open(img_path, "rb") as f:
    resp = requests.post(
        "http://127.0.0.1:5000/predict",
        files={"file": f},
        data={"algorithm": "efficientnet"},
    )
r = resp.json()
print("=== EfficientNet-B0 (glioma image) ===")
print(f"  Predicted : {r['predicted_class']}")
print(f"  Confidence: {r['confidence']}%")
print(f"  Algorithm : {r['algorithm_used']}")
for k, v in r["all_probs"].items():
    print(f"    {k}: {v}%")

print()

# --- Test with pituitary image ---
img2 = r"E:\Brain Tumor\Testing\pituitary\Te-pi_1.jpg"
for algo, label in [("resnet", "ResNet50"), ("efficientnet", "EfficientNet-B0")]:
    with open(img2, "rb") as f:
        resp = requests.post(
            "http://127.0.0.1:5000/predict",
            files={"file": f},
            data={"algorithm": algo},
        )
    r = resp.json()
    print(f"=== {label} (pituitary image) ===")
    print(f"  Predicted : {r['predicted_class']}  Confidence: {r['confidence']}%")
