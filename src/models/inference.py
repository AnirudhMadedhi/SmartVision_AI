# src/models/inference.py

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from pathlib import Path

# ------------------------------------------------------------------
# Paths & Config
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATHS = {
    "VGG16": BASE_DIR / "models/vgg16_best.pth",
    "ResNet50": BASE_DIR / "models/resnet50_best.pth",
    "MobileNetV2": BASE_DIR / "models/mobilenetv2_best.pth",
    "EfficientNetB0": BASE_DIR / "models/efficientnetb0_best.pth",
}

NUM_CLASSES = 26
DEVICE = torch.device("cpu")

# ⚠️ MUST MATCH training folder order
CLASS_NAMES = [
    "airplane", "bed", "bench", "bicycle", "bird", "bottle", "bowl",
    "bus", "cake", "car", "cat", "chair", "couch", "cow", "cup",
    "dog", "elephant", "horse", "motorcycle", "person", "pizza",
    "potted plant", "stop sign", "traffic light", "train", "truck"
]

# ------------------------------------------------------------------
# Image Transform
# ------------------------------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ------------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------------
def build_model(name: str):
    if name == "VGG16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = torch.nn.Linear(4096, NUM_CLASSES)

    elif name == "ResNet50":
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    elif name == "MobileNetV2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, NUM_CLASSES
        )

    elif name == "EfficientNetB0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, NUM_CLASSES
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    return model


# ------------------------------------------------------------------
# Load All Models Once (important for Streamlit)
# ------------------------------------------------------------------
_MODELS = {}

def load_models():
    if _MODELS:
        return _MODELS

    for name, path in MODEL_PATHS.items():
        model = build_model(name)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _MODELS[name] = model

    return _MODELS


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
@torch.no_grad()
def run_classification(image: Image.Image):
    models_dict = load_models()

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    results = {}

    for model_name, model in models_dict.items():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0]

        # Top-5
        top5 = torch.topk(probs, k=5)

        top5_preds = [
            {
                "class": CLASS_NAMES[idx],
                "confidence": float(prob)
            }
            for idx, prob in zip(top5.indices.tolist(), top5.values.tolist())
        ]

        # Top-1
        top1_idx = top5.indices[0].item()
        top1_conf = top5.values[0].item()

        results[model_name] = {
            "predicted_class": CLASS_NAMES[top1_idx],
            "confidence": float(top1_conf),
            "top5": top5_preds
        }

    return results
