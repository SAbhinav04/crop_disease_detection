#!/usr/bin/env python3
"""FastAPI PoC backend using pretrained ResNet50 for image inference."""

from __future__ import annotations

import io
from typing import Dict, Tuple

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights


app = FastAPI(title="Crop Disease PoC API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
WEIGHTS = ResNet50_Weights.DEFAULT
TRANSFORM = WEIGHTS.transforms()
MODEL = models.resnet50(weights=WEIGHTS).to(DEVICE).eval()
IMAGENET_CATEGORIES = WEIGHTS.meta["categories"]


def map_prediction_to_domain(label: str) -> Tuple[str, str]:
    lower = label.lower()

    # Basic crop+disease heuristics from ImageNet classes for PoC output format.
    if "apple" in lower:
        if "scab" in lower:
            return "Apple", "Apple_Apple_scab"
        if "rot" in lower:
            return "Apple", "Apple_Black_rot"
        return "Apple", "Apple_Healthy"

    if "grape" in lower:
        if "rot" in lower:
            return "Grape", "Grape_Black_rot"
        return "Grape", "Grape_Healthy"

    if "bell pepper" in lower or "pepper" in lower:
        return "Pepper", "Pepper_Bacterial_spot"

    if "tomato" in lower:
        return "Tomato", "Tomato_Early_blight"

    if "potato" in lower:
        return "Potato", "Potato_Early_blight"

    if "corn" in lower or "maize" in lower:
        return "Corn", "Corn_Common_rust"

    return "Unknown", "Unknown_Disease"


def severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.85:
        return "Severe"
    if confidence >= 0.6:
        return "Moderate"
    return "Mild"


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = MODEL(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    confidence_value = float(confidence.item())
    imagenet_label = IMAGENET_CATEGORIES[int(pred_idx.item())]
    crop, disease = map_prediction_to_domain(imagenet_label)

    return {
        "disease": disease,
        "confidence": round(confidence_value, 4),
        "severity": severity_from_confidence(confidence_value),
        "crop": crop,
    }
