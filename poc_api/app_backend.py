#!/usr/bin/env python3
"""FastAPI backend — serves the fine-tuned ResNet50 plant-disease model.

If poc/best_model.pth and poc/class_names.json are present the server uses the
fine-tuned model (15 classes, ~98.79 % accuracy).  Otherwise it falls back to
the bare ImageNet ResNet50 PoC weights so the API keeps working even before
training.

Key design decisions
--------------------
* /predict now returns the full remedy (English + Kannada) inline so the
  frontend only needs a single round-trip.
* Remedy is looked up by a compound key  <Crop>_<Disease>  (e.g.
  "Tomato_Early Blight") which avoids cross-crop ambiguity.
"""

from __future__ import annotations

import base64
import io
import json
import os
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import torch
import torch.nn as nn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE           = Path(__file__).resolve().parent          # poc_api/
_POC_DIR        = _HERE.parent / "poc"                     # poc/
_MODEL_PATH     = _POC_DIR / "best_model.pth"
_CLASSES_PATH   = _POC_DIR / "class_names.json"

# Load env from repo root and poc_api so secrets can be set in either place.
load_dotenv(_HERE.parent / ".env")
load_dotenv(_HERE / ".env")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_TTS_URL = os.getenv("SARVAM_TTS_URL", "https://api.sarvam.ai/text-to-speech/stream").strip()
SARVAM_TTS_VOICE = os.getenv("SARVAM_TTS_VOICE", "anushka").strip()
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3").strip()
SARVAM_TTS_SAMPLE_RATE = int(os.getenv("SARVAM_TTS_SAMPLE_RATE", "22050"))
SARVAM_TIMEOUT_SECONDS = float(os.getenv("SARVAM_TIMEOUT_SECONDS", "30"))


# ---------------------------------------------------------------------------
# Class-name → (crop, disease) lookup for the 15 PlantVillage classes
# ---------------------------------------------------------------------------

CLASS_LABELS: Dict[str, Tuple[str, str]] = {
    "Pepper__bell___Bacterial_spot":                ("Pepper",  "Bacterial Spot"),
    "Pepper__bell___healthy":                       ("Pepper",  "Healthy"),
    "Potato___Early_blight":                        ("Potato",  "Early Blight"),
    "Potato___Late_blight":                         ("Potato",  "Late Blight"),
    "Potato___healthy":                             ("Potato",  "Healthy"),
    "Tomato_Bacterial_spot":                        ("Tomato",  "Bacterial Spot"),
    "Tomato_Early_blight":                          ("Tomato",  "Early Blight"),
    "Tomato_Late_blight":                           ("Tomato",  "Late Blight"),
    "Tomato_Leaf_Mold":                             ("Tomato",  "Leaf Mold"),
    "Tomato_Septoria_leaf_spot":                    ("Tomato",  "Septoria Leaf Spot"),
    "Tomato_Spider_mites_Two_spotted_spider_mite":  ("Tomato",  "Spider Mites"),
    "Tomato__Target_Spot":                          ("Tomato",  "Target Spot"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus":        ("Tomato",  "Yellow Leaf Curl Virus"),
    "Tomato__Tomato_mosaic_virus":                  ("Tomato",  "Mosaic Virus"),
    "Tomato_healthy":                               ("Tomato",  "Healthy"),
}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_device()


# ---------------------------------------------------------------------------
# Model + transform loading
# ---------------------------------------------------------------------------

def _load_finetuned() -> Tuple[nn.Module, List[str], transforms.Compose]:
    """Load fine-tuned ResNet50 from poc/best_model.pth."""
    with _CLASSES_PATH.open("r", encoding="utf-8") as f:
        class_names: List[str] = json.load(f)

    num_classes = len(class_names)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    print(f"[backend] Loaded fine-tuned model ({num_classes} classes) on {DEVICE}")
    return model, class_names, tf


def _load_imagenet_fallback() -> Tuple[nn.Module, List[str], transforms.Compose]:
    """Fallback: bare ImageNet ResNet50 (low accuracy, but keeps API alive)."""
    weights  = ResNet50_Weights.DEFAULT
    model    = models.resnet50(weights=weights).to(DEVICE).eval()
    tf       = weights.transforms()
    print(f"[backend] Fine-tuned model not found — using ImageNet fallback on {DEVICE}")
    return model, weights.meta["categories"], tf


# Try fine-tuned model first; fall back gracefully
if _MODEL_PATH.exists() and _CLASSES_PATH.exists():
    MODEL, CLASS_NAMES, TRANSFORM = _load_finetuned()
    USING_FINETUNED = True
else:
    MODEL, CLASS_NAMES, TRANSFORM = _load_imagenet_fallback()
    USING_FINETUNED = False

PREDICTION_HISTORY: List[Dict[str, object]] = []
MAX_HISTORY = 100


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Crop Disease API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TtsRequest(BaseModel):
    text: str
    language: str = "kn"


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _predict_finetuned(probs: torch.Tensor) -> Tuple[str, str, float, List[Dict]]:
    """Use class_names.json lookup — exact and fast."""
    confidence, idx = torch.max(probs, dim=1)
    conf_value   = float(confidence.item())
    class_name   = CLASS_NAMES[int(idx.item())]
    crop, disease = CLASS_LABELS.get(class_name, ("Unknown", class_name))

    # Build top-5 candidates for transparency
    top_k = min(5, probs.shape[1])
    top_confs, top_idxs = torch.topk(probs, k=top_k, dim=1)
    candidates = [
        {
            "class_name": CLASS_NAMES[int(i.item())],
            "confidence": round(float(c.item()), 4),
            "crop":    CLASS_LABELS.get(CLASS_NAMES[int(i.item())], ("Unknown", ""))[0],
            "disease": CLASS_LABELS.get(CLASS_NAMES[int(i.item())], ("Unknown", CLASS_NAMES[int(i.item())]))[1],
        }
        for c, i in zip(top_confs[0], top_idxs[0])
    ]
    return crop, disease, conf_value, candidates


def _predict_imagenet_fallback(probs: torch.Tensor) -> Tuple[str, str, float, List[Dict]]:
    """Keyword-matching fallback when fine-tuned model is unavailable."""
    confidence, idx = torch.max(probs, dim=1)
    conf_value      = float(confidence.item())
    label           = CLASS_NAMES[int(idx.item())]

    crop, disease = _guess_from_imagenet_label(label)
    return crop, disease, conf_value, [{"imagenet_label": label, "confidence": round(conf_value, 4)}]


def _guess_from_imagenet_label(label: str) -> Tuple[str, str]:
    lower = label.lower()
    if "tomato" in lower:
        return "Tomato", "Tomato_Healthy" if "healthy" in lower else "Tomato_Unknown"
    if "potato" in lower:
        return "Potato", "Potato_Healthy"
    if "pepper" in lower:
        return "Pepper", "Pepper_Healthy"
    return "Unknown", f"Unknown_{label.replace(' ', '_')}"


def _severity(confidence: float) -> str:
    if confidence >= 0.85:
        return "Severe"
    if confidence >= 0.60:
        return "Moderate"
    return "Mild"


# ---------------------------------------------------------------------------
# Remedy text (English + Kannada) for all 15 classes
# ---------------------------------------------------------------------------

_REMEDY: Dict[str, Dict[str, Dict]] = {
    # ── Pepper ──────────────────────────────────────────────────────────────
    "Pepper_Bacterial Spot": {
        "english": {
            "cause": "Bacterial spot on pepper is caused by Xanthomonas bacteria spread by rain splash and infected seeds.",
            "symptoms": ["Small, water-soaked dark spots on leaves", "Yellow halos around spots", "Leaf drop and fruit lesions"],
            "treatment_steps": ["Remove heavily spotted leaves", "Apply copper-based bactericide", "Avoid overhead irrigation"],
            "prevention": ["Use certified disease-free seed", "Improve plant spacing for airflow", "Sanitize tools between plants"],
            "fertilizer_recommendation": "Avoid excess nitrogen; use balanced NPK to maintain plant vigour.",
        },
        "kannada": {
            "cause": "ಮೆಣಸಿನಕಾಯಿಯ ಬ್ಯಾಕ್ಟೀರಿಯಾ ಕಲೆ ರೋಗವು Xanthomonas ಬ್ಯಾಕ್ಟೀರಿಯಾದಿಂದ ಉಂಟಾಗುತ್ತದೆ.",
            "symptoms": ["ಎಲೆಗಳ ಮೇಲೆ ಸಣ್ಣ ಕಪ್ಪು ಕಲೆಗಳು", "ಕಲೆಗಳ ಸುತ್ತ ಹಳದಿ ವಲಯ", "ಎಲೆ ಉದುರುವಿಕೆ"],
            "treatment_steps": ["ತೀವ್ರ ಸೋಂಕಿತ ಎಲೆ ತೆಗೆಯಿರಿ", "ಕಾಪರ್ ಆಧಾರಿತ ಸ್ಪ್ರೇ ಬಳಸಿ", "ಮೇಲಿನ ನೀರಾವರಿ ತಪ್ಪಿಸಿ"],
            "prevention": ["ರೋಗಮುಕ್ತ ಬೀಜ ಬಳಸಿ", "ಗಾಳಿ ಚಲನೆಗೆ ಸ್ಥಳಾವಕಾಶ ನೀಡಿ", "ಉಪಕರಣ ಶುಚಿಗೊಳಿಸಿ"],
            "fertilizer_recommendation": "ಹೆಚ್ಚಿನ ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ; ಸಮತೋಲನ NPK ಬಳಸಿ.",
        },
    },
    "Pepper_Healthy": {
        "english": {
            "cause": "No disease detected — the pepper plant appears healthy.",
            "symptoms": ["Uniform green leaves", "Vigorous growth", "No visible lesions or spots"],
            "treatment_steps": ["Continue regular monitoring", "Maintain irrigation schedule", "Keep field sanitation"],
            "prevention": ["Regular scouting for early detection", "Balanced fertilization", "Crop rotation"],
            "fertilizer_recommendation": "Maintain soil-test-based balanced nutrition.",
        },
        "kannada": {
            "cause": "ರೋಗ ಕಾಣಿಸಿಲ್ಲ — ಮೆಣಸಿನಕಾಯಿ ಸಸ್ಯ ಆರೋಗ್ಯಕರವಾಗಿದೆ.",
            "symptoms": ["ಸಮಾನ ಹಸಿರು ಎಲೆಗಳು", "ಬಲವಾದ ಬೆಳವಣಿಗೆ", "ಯಾವುದೇ ಕಲೆ ಇಲ್ಲ"],
            "treatment_steps": ["ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ", "ನೀರಾವರಿ ಕ್ರಮ ಕಾಪಾಡಿ", "ಕ್ಷೇತ್ರ ಶುಚಿ ಇಟ್ಟುಕೊಳ್ಳಿ"],
            "prevention": ["ನಿಯಮಿತ ಪರಿಶೀಲನೆ", "ಸಮತೋಲನ ಗೊಬ್ಬರ", "ಬೆಳೆ ಪರಿವರ್ತನೆ"],
            "fertilizer_recommendation": "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಆಧಾರದ ಸಮತೋಲನ ಪೋಷಕಾಂಶ ನೀಡಿ.",
        },
    },

    # ── Potato ──────────────────────────────────────────────────────────────
    "Potato_Early Blight": {
        "english": {
            "cause": "Early blight is caused by the fungus Alternaria solani; it thrives in warm, humid conditions.",
            "symptoms": ["Dark brown target-like spots on lower leaves", "Yellow ring around lesions", "Premature leaf drop"],
            "treatment_steps": ["Remove infected leaves promptly", "Apply chlorothalonil or mancozeb fungicide", "Avoid wetting foliage"],
            "prevention": ["Practice crop rotation (avoid Solanaceae for 2–3 years)", "Water at the base", "Plant certified disease-free seed potatoes"],
            "fertilizer_recommendation": "Balanced NPK with adequate potassium to strengthen plant tissue.",
        },
        "kannada": {
            "cause": "ಆಲೂಗಡ್ಡೆ ಆರಂಭಿಕ ಅಂಗಮಾರಿ ರೋಗವು Alternaria solani ಶಿಲೀಂಧ್ರದಿಂದ ಉಂಟಾಗುತ್ತದೆ.",
            "symptoms": ["ಕೆಳ ಎಲೆಗಳ ಮೇಲೆ ಕಪ್ಪು ಗುರಿ ಮಾದರಿ ಕಲೆಗಳು", "ಗಾಯಗಳ ಸುತ್ತ ಹಳದಿ ವಲಯ", "ಎಲೆ ಮುಂಚಿತವಾಗಿ ಉದುರುವುದು"],
            "treatment_steps": ["ತಕ್ಷಣ ಸೋಂಕಿತ ಎಲೆ ತೆಗೆಯಿರಿ", "ಶಿಫಾರಸು ಮಾಡಿದ ಫಂಗಿಸೈಡ್ ಬಳಸಿ", "ಎಲೆ ತೇವ ತಪ್ಪಿಸಿ"],
            "prevention": ["ಬೆಳೆ ಪರಿವರ್ತನೆ ಅನುಸರಿಸಿ", "ಮೂಲಭಾಗಕ್ಕೆ ನೀರು ನೀಡಿ", "ರೋಗಮುಕ್ತ ಬೀಜ ಆಲೂ ಬಳಸಿ"],
            "fertilizer_recommendation": "ಸಮತೋಲನ NPK ಜೊತೆ ಸಾಕಷ್ಟು ಪೊಟ್ಯಾಶಿಯಂ ನೀಡಿ.",
        },
    },
    "Potato_Late Blight": {
        "english": {
            "cause": "Late blight is caused by Phytophthora infestans; it spreads rapidly in cool, wet weather.",
            "symptoms": ["Water-soaked pale green spots that turn brown-black", "White mold on undersides of leaves", "Rapid plant collapse in wet conditions"],
            "treatment_steps": ["Apply metalaxyl or cymoxanil-based fungicide immediately", "Destroy heavily infected plants", "Improve field drainage"],
            "prevention": ["Use blight-resistant varieties", "Avoid dense planting", "Monitor weather forecasts and spray preventively"],
            "fertilizer_recommendation": "Avoid excess nitrogen; ensure adequate calcium and potassium.",
        },
        "kannada": {
            "cause": "ಆಲೂಗಡ್ಡೆ ತಡ ಅಂಗಮಾರಿ ರೋಗವು Phytophthora infestans ಶಿಲೀಂಧ್ರದಿಂದ ಉಂಟಾಗುತ್ತದೆ.",
            "symptoms": ["ನೀರಿನಂತೆ ತೋರುವ ಕಲೆಗಳು ಕಪ್ಪಾಗುತ್ತವೆ", "ಎಲೆ ಕೆಳಭಾಗದಲ್ಲಿ ಬಿಳಿ ಶಿಲೀಂಧ್ರ", "ತೇವ ವಾತಾವರಣದಲ್ಲಿ ಸಸ್ಯ ಶೀಘ್ರ ಕುಸಿಯುತ್ತದೆ"],
            "treatment_steps": ["ತಕ್ಷಣ ಶಿಫಾರಸು ಫಂಗಿಸೈಡ್ ಸ್ಪ್ರೇ ಮಾಡಿ", "ತೀವ್ರ ಸೋಂಕಿತ ಸಸ್ಯ ನಾಶ ಮಾಡಿ", "ಕ್ಷೇತ್ರ ಒಳಚರಂಡಿ ಸುಧಾರಿಸಿ"],
            "prevention": ["ನಿರೋಧಕ ತಳಿ ಬಳಸಿ", "ದಟ್ಟ ನಾಟಿ ತಪ್ಪಿಸಿ", "ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ ಆಧಾರದ ಸ್ಪ್ರೇ ಮಾಡಿ"],
            "fertilizer_recommendation": "ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ; ಕ್ಯಾಲ್ಸಿಯಂ ಮತ್ತು ಪೊಟ್ಯಾಶಿಯಂ ಒದಗಿಸಿ.",
        },
    },
    "Potato_Healthy": {
        "english": {
            "cause": "No disease detected — the potato plant appears healthy.",
            "symptoms": ["Upright stems with dark green leaves", "No spots or wilting", "Good tuber development"],
            "treatment_steps": ["Continue monitoring", "Maintain irrigation", "Scout for early signs of blight"],
            "prevention": ["Use certified seed potatoes", "Practice crop rotation", "Avoid waterlogging"],
            "fertilizer_recommendation": "Balanced NPK tailored to your soil test results.",
        },
        "kannada": {
            "cause": "ಯಾವುದೇ ರೋಗ ಕಾಣಿಸಿಲ್ಲ — ಆಲೂಗಡ್ಡೆ ಸಸ್ಯ ಆರೋಗ್ಯಕರ.",
            "symptoms": ["ಗಾಢ ಹಸಿರು ಎಲೆಗಳು", "ಕಲೆ ಅಥವಾ ಬಾಡುವಿಕೆ ಇಲ್ಲ", "ಉತ್ತಮ ಗೆಡ್ಡೆ ಬೆಳವಣಿಗೆ"],
            "treatment_steps": ["ನಿರಂತರ ಮೇಲ್ವಿಚಾರಣೆ", "ನೀರಾವರಿ ಕಾಪಾಡಿ", "ಅಂಗಮಾರಿ ಆರಂಭಿಕ ಚಿಹ್ನೆಗಳನ್ನು ಗಮನಿಸಿ"],
            "prevention": ["ರೋಗಮುಕ್ತ ಬೀಜ ಬಳಸಿ", "ಬೆಳೆ ಪರಿವರ್ತನೆ", "ನೀರು ನಿಲ್ಲದಂತೆ ಎಚ್ಚರಿಸಿ"],
            "fertilizer_recommendation": "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಆಧಾರದ ಸಮತೋಲನ NPK ನೀಡಿ.",
        },
    },

    # ── Tomato ──────────────────────────────────────────────────────────────
    "Tomato_Bacterial Spot": {
        "english": {
            "cause": "Caused by Xanthomonas bacteria spread through rain, insects, and contaminated equipment.",
            "symptoms": ["Small dark water-soaked spots on leaves and fruit", "Yellow halo around leaf spots", "Defoliation in severe cases"],
            "treatment_steps": ["Apply copper-based bactericide at first signs", "Remove infected plant parts", "Avoid working with wet plants"],
            "prevention": ["Use resistant varieties and clean seed", "Disinfect tools regularly", "Rotate crops away from Solanaceae"],
            "fertilizer_recommendation": "Balanced NPK; avoid excess nitrogen which promotes soft susceptible tissue.",
        },
        "kannada": {
            "cause": "Xanthomonas ಬ್ಯಾಕ್ಟೀರಿಯಾ ಮಳೆ ಮತ್ತು ಸೋಂಕಿತ ಉಪಕರಣಗಳ ಮೂಲಕ ಹರಡುತ್ತದೆ.",
            "symptoms": ["ಎಲೆ ಮತ್ತು ಹಣ್ಣಿನ ಮೇಲೆ ಸಣ್ಣ ಕಪ್ಪು ಕಲೆಗಳು", "ಕಲೆಗಳ ಸುತ್ತ ಹಳದಿ ವಲಯ", "ತೀವ್ರ ಸಂದರ್ಭದಲ್ಲಿ ಎಲೆ ಉದುರುವಿಕೆ"],
            "treatment_steps": ["ಆರಂಭದಲ್ಲೇ ಕಾಪರ್ ಸ್ಪ್ರೇ ಮಾಡಿ", "ಸೋಂಕಿತ ಭಾಗ ತೆಗೆಯಿರಿ", "ತೇವ ಸಸ್ಯದ ಜೊತೆ ಕೆಲಸ ತಪ್ಪಿಸಿ"],
            "prevention": ["ನಿರೋಧಕ ತಳಿ ಮತ್ತು ಸ್ವಚ್ಛ ಬೀಜ ಬಳಸಿ", "ಉಪಕರಣ ಕ್ರಮಬದ್ಧವಾಗಿ ಸ್ವಚ್ಛಗೊಳಿಸಿ", "ಬೆಳೆ ಪರಿವರ್ತನೆ"],
            "fertilizer_recommendation": "ಸಮತೋಲನ NPK ಬಳಸಿ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ.",
        },
    },
    "Tomato_Early Blight": {
        "english": {
            "cause": "Early blight (Alternaria solani) develops under warm humid conditions and on stressed plants.",
            "symptoms": ["Concentric ring (target) spots on older leaves", "Yellowing around lesions", "Premature leaf drop"],
            "treatment_steps": ["Remove lower infected leaves", "Apply mancozeb or chlorothalonil", "Reduce leaf wetness duration"],
            "prevention": ["Rotate crops", "Mulch to prevent soil splash", "Avoid wetting foliage"],
            "fertilizer_recommendation": "Adequate nitrogen and potassium to keep plants vigorous without excess.",
        },
        "kannada": {
            "cause": "ಟೊಮೇಟೊ ಆರಂಭಿಕ ಅಂಗಮಾರಿ Alternaria solani ಶಿಲೀಂಧ್ರದಿಂದ ಉಂಟಾಗುತ್ತದೆ.",
            "symptoms": ["ಹಳೆಯ ಎಲೆಗಳ ಮೇಲೆ ಚಕ್ರ ಮಾದರಿ ಕಲೆ", "ಗಾಯಗಳ ಸುತ್ತ ಹಳದಿ", "ಮುಂಚಿತ ಎಲೆ ಉದುರುವಿಕೆ"],
            "treatment_steps": ["ಕೆಳ ಸೋಂಕಿತ ಎಲೆ ತೆಗೆಯಿರಿ", "ಶಿಫಾರಿಸು ಫಂಗಿಸೈಡ್ ಬಳಸಿ", "ಎಲೆ ತೇವ ಕಡಿಮೆ ಮಾಡಿ"],
            "prevention": ["ಬೆಳೆ ಪರಿವರ್ತನೆ", "ಮಣ್ಣಿನ ಚಿಮ್ಮು ತಡೆಗೆ ಮಲ್ಚ್ ಬಳಸಿ", "ಎಲೆ ತೇವ ತಪ್ಪಿಸಿ"],
            "fertilizer_recommendation": "ಸಮತೋಲನ ನೈಟ್ರೋಜನ್ ಮತ್ತು ಪೊಟ್ಯಾಶಿಯಂ ನೀಡಿ.",
        },
    },
    "Tomato_Late Blight": {
        "english": {
            "cause": "Late blight (Phytophthora infestans) spreads explosively in cool, wet conditions.",
            "symptoms": ["Large irregular dark blotches on leaves and stems", "White cottony growth on undersides", "Fruit develops brown firm rot"],
            "treatment_steps": ["Apply metalaxyl or cymoxanil fungicide immediately", "Destroy and remove infected plant material", "Improve drainage"],
            "prevention": ["Use resistant varieties", "Avoid dense canopy", "Monitor and spray preventively in wet seasons"],
            "fertilizer_recommendation": "Balanced calcium and potassium; avoid excess nitrogen.",
        },
        "kannada": {
            "cause": "ಟೊಮೇಟೊ ತಡ ಅಂಗಮಾರಿ Phytophthora infestans ಶಿಲೀಂಧ್ರದಿಂದ ತ್ವರಿತವಾಗಿ ಹರಡುತ್ತದೆ.",
            "symptoms": ["ಎಲೆ ಮತ್ತು ಕಾಂಡದ ಮೇಲೆ ದೊಡ್ಡ ಕಪ್ಪು ಚುಕ್ಕೆ", "ಕೆಳ ಎಲೆಯಲ್ಲಿ ಬಿಳಿ ಶಿಲೀಂಧ್ರ", "ಹಣ್ಣು ಕಂದು ಕೊಳೆಯಾಗುತ್ತದೆ"],
            "treatment_steps": ["ತಕ್ಷಣ ಶಿಫಾರಸು ಫಂಗಿಸೈಡ್ ಸ್ಪ್ರೇ ಮಾಡಿ", "ಸೋಂಕಿತ ಸಸ್ಯ ನಾಶ ಮಾಡಿ", "ಒಳಚರಂಡಿ ಸುಧಾರಿಸಿ"],
            "prevention": ["ನಿರೋಧಕ ತಳಿ ಬಳಸಿ", "ದಟ್ಟ ಎಲೆ ಚಂದವ ತಪ್ಪಿಸಿ", "ಮಳೆಗಾಲದಲ್ಲಿ ಮುಂಜಾಗ್ರತೆ ಸ್ಪ್ರೇ ಮಾಡಿ"],
            "fertilizer_recommendation": "ಕ್ಯಾಲ್ಸಿಯಂ ಮತ್ತು ಪೊಟ್ಯಾಶಿಯಂ ಸಮತೋಲನದಲ್ಲಿ ಒದಗಿಸಿ.",
        },
    },
    "Tomato_Leaf Mold": {
        "english": {
            "cause": "Caused by the fungus Passalora fulva; thrives in high humidity (>85%) and poor ventilation.",
            "symptoms": ["Pale yellow spots on upper leaf surface", "Olive-green to brown velvet mold on undersides", "Leaves curl and die"],
            "treatment_steps": ["Reduce humidity by improving ventilation", "Apply fungicide (chlorothalonil/copper)", "Remove affected leaves"],
            "prevention": ["Stake and prune tomatoes properly", "Avoid dense planting", "Use drip irrigation"],
            "fertilizer_recommendation": "Balanced NPK; avoid excess nitrogen which promotes dense leaf growth.",
        },
        "kannada": {
            "cause": "ಟೊಮೇಟೊ ಎಲೆ ಶಿಲೀಂಧ್ರ Passalora fulva ಶಿಲೀಂಧ್ರದಿಂದ ಉಂಟಾಗುತ್ತದೆ.",
            "symptoms": ["ಎಲೆಯ ಮೇಲ್ಭಾಗದಲ್ಲಿ ತಿಳಿ ಹಳದಿ ಕಲೆ", "ಕೆಳಭಾಗದಲ್ಲಿ ಹಳದಿ-ಕಂದು ಶಿಲೀಂಧ್ರ", "ಎಲೆ ಮಡಚಿ ಒಣಗುತ್ತದೆ"],
            "treatment_steps": ["ಗಾಳಿಯಾಡುವಿಕೆ ಹೆಚ್ಚಿಸಿ ತೇವ ಕಡಿಮೆ ಮಾಡಿ", "ಶಿಫಾರಸು ಫಂಗಿಸೈಡ್ ಬಳಸಿ", "ಪೀಡಿತ ಎಲೆ ತೆಗೆಯಿರಿ"],
            "prevention": ["ಸರಿಯಾಗಿ ಗಳ ನೆಟ್ಟು ಕಟ್ಟಿ", "ದಟ್ಟ ನಾಟಿ ತಪ್ಪಿಸಿ", "ಹನಿ ನೀರಾವರಿ ಬಳಸಿ"],
            "fertilizer_recommendation": "ಸಮತೋಲನ NPK ಬಳಸಿ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ.",
        },
    },
    "Tomato_Septoria Leaf Spot": {
        "english": {
            "cause": "Caused by the fungus Septoria lycopersici; spreads through water splash and infected debris.",
            "symptoms": ["Numerous small circular spots with grey centres and dark borders", "Begins on lower leaves, moves upward", "Yellowing and premature leaf drop"],
            "treatment_steps": ["Remove lower infected leaves", "Apply fungicide at first sign", "Avoid water splash on leaves"],
            "prevention": ["Mulch the soil surface", "Stake plants to improve airflow", "Follow crop rotation"],
            "fertilizer_recommendation": "Balanced fertilization; excess nitrogen worsens disease severity.",
        },
        "kannada": {
            "cause": "Septoria lycopersici ಶಿಲೀಂಧ್ರ ನೀರಿನ ಚಿಮ್ಮಿನಿಂದ ಹರಡುತ್ತದೆ.",
            "symptoms": ["ಸಣ್ಣ ಬೂದಿ ಕೇಂದ್ರದ ಕಲೆಗಳು", "ಕೆಳ ಎಲೆಗಳಿಂದ ಶುರುವಾಗಿ ಮೇಲೆ ಹರಡುತ್ತದೆ", "ಹಳದಿ ಮತ್ತು ಎಲೆ ಉದುರುವಿಕೆ"],
            "treatment_steps": ["ಕೆಳ ಸೋಂಕಿತ ಎಲೆ ತೆಗೆಯಿರಿ", "ಆರಂಭದಲ್ಲೇ ಫಂಗಿಸೈಡ್ ಬಳಸಿ", "ನೀರಿನ ಚಿಮ್ಮು ತಪ್ಪಿಸಿ"],
            "prevention": ["ಮಣ್ಣಿನ ಮೇಲ್ಮೈ ಮಲ್ಚ್ ಮಾಡಿ", "ಸಸ್ಯ ಗಳ ಹಾಕಿ ಗಾಳಿ ಚಲನೆ ಹೆಚ್ಚಿಸಿ", "ಬೆಳೆ ಪರಿವರ್ತನೆ"],
            "fertilizer_recommendation": "ಸಮತೋಲನ ಗೊಬ್ಬರ ಬಳಸಿ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ.",
        },
    },
    "Tomato_Spider Mites": {
        "english": {
            "cause": "Two-spotted spider mites (Tetranychus urticae) thrive in hot, dry conditions; not a fungal disease.",
            "symptoms": ["Tiny yellow stippling on leaves", "Fine webbing on undersides", "Leaves turn bronze/brown and dry out"],
            "treatment_steps": ["Apply miticide or neem oil spray", "Increase plant humidity by misting", "Remove heavily infested leaves"],
            "prevention": ["Avoid excessive nitrogen fertilization", "Maintain adequate soil moisture", "Introduce natural predators (predatory mites)"],
            "fertilizer_recommendation": "Balanced nutrition; avoid excess nitrogen which promotes pest-susceptible growth.",
        },
        "kannada": {
            "cause": "ಎರಡು-ಚುಕ್ಕೆ ಜೇಡ ಕೀಟ (Tetranychus urticae) ಬಿಸಿ, ಒಣ ವಾತಾವರಣದಲ್ಲಿ ಅಭಿವೃದ್ಧಿ ಹೊಂದುತ್ತದೆ.",
            "symptoms": ["ಎಲೆಗಳ ಮೇಲೆ ಸಣ್ಣ ಹಳದಿ ಚುಕ್ಕೆ", "ಕೆಳಭಾಗದಲ್ಲಿ ಸೂಕ್ಷ್ಮ ಜಾಲ", "ಎಲೆ ಕಂದು ಬಣ್ಣ ಪಡೆದು ಒಣಗುತ್ತದೆ"],
            "treatment_steps": ["ನೀಮ್ ಎಣ್ಣೆ ಅಥವಾ ಮಿಟಿಸೈಡ್ ಸ್ಪ್ರೇ ಮಾಡಿ", "ಸಸ್ಯದ ತೇವ ಹೆಚ್ಚಿಸಿ", "ತೀವ್ರ ಸೋಂಕಿತ ಎಲೆ ತೆಗೆಯಿರಿ"],
            "prevention": ["ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ", "ಸಮರ್ಪಕ ಮಣ್ಣಿನ ತೇವ ಕಾಪಾಡಿ", "ನೈಸರ್ಗಿಕ ಶತ್ರುಗಳನ್ನು ಪ್ರೋತ್ಸಾಹಿಸಿ"],
            "fertilizer_recommendation": "ಸಮತೋಲನ ಪೋಷಕಾಂಶ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ.",
        },
    },
    "Tomato_Target Spot": {
        "english": {
            "cause": "Target spot (Corynespora cassiicola) is favoured by wet, warm weather and plant stress.",
            "symptoms": ["Dark brown circular spots with concentric rings", "Affects leaves, stems, and fruit", "Severe defoliation in wet seasons"],
            "treatment_steps": ["Remove infected tissue", "Apply azoxystrobin or chlorothalonil", "Improve canopy airflow"],
            "prevention": ["Avoid overhead irrigation", "Prune lower leaves", "Crop rotation and field sanitation"],
            "fertilizer_recommendation": "Balanced nutrition especially potassium to improve disease tolerance.",
        },
        "kannada": {
            "cause": "ಟಾರ್ಗೆಟ್ ಸ್ಪಾಟ್ Corynespora cassiicola ಶಿಲೀಂಧ್ರದಿಂದ ತೇವ-ಬಿಸಿ ವಾತಾವರಣದಲ್ಲಿ ಹರಡುತ್ತದೆ.",
            "symptoms": ["ಚಕ್ರ ಮಾದರಿ ಕಪ್ಪು ಕಲೆ", "ಎಲೆ, ಕಾಂಡ ಮತ್ತು ಹಣ್ಣಿಗೆ ತಗಲುತ್ತದೆ", "ಮಳೆಗಾಲದಲ್ಲಿ ತೀವ್ರ ಎಲೆ ಉದುರುವಿಕೆ"],
            "treatment_steps": ["ಸೋಂಕಿತ ಭಾಗ ತೆಗೆಯಿರಿ", "ಶಿಫಾರಸು ಫಂಗಿಸೈಡ್ ಬಳಸಿ", "ಮರದ ಗಾಳಿ ಚಲನೆ ಹೆಚ್ಚಿಸಿ"],
            "prevention": ["ಮೇಲಿನ ನೀರಾವರಿ ತಪ್ಪಿಸಿ", "ಕೆಳ ಎಲೆ ಕತ್ತರಿಸಿ", "ಬೆಳೆ ಪರಿವರ್ತನೆ"],
            "fertilizer_recommendation": "ರೋಗ ನಿರೋಧಕ ಸಾಮರ್ಥ್ಯಕ್ಕಾಗಿ ಪೊಟ್ಯಾಶಿಯಂ ಒದಗಿಸಿ.",
        },
    },
    "Tomato_Yellow Leaf Curl Virus": {
        "english": {
            "cause": "Tomato Yellow Leaf Curl Virus (TYLCV) is transmitted by whiteflies (Bemisia tabaci).",
            "symptoms": ["Upward curling and yellowing of young leaves", "Stunted plant growth", "Flower drop and poor fruit set"],
            "treatment_steps": ["Control whiteflies with imidacloprid or neem", "Remove and destroy infected plants", "Use reflective mulch to repel whiteflies"],
            "prevention": ["Use virus-resistant varieties", "Install insect-proof netting on seedling beds", "Monitor and control whitefly populations early"],
            "fertilizer_recommendation": "Balanced nutrition; stressed plants are more susceptible to virus transmission.",
        },
        "kannada": {
            "cause": "TYLCV ವೈರಸ್ ಬಿಳಿ ನೊಣ (Bemisia tabaci) ಮೂಲಕ ಹರಡುತ್ತದೆ.",
            "symptoms": ["ಎಳೆ ಎಲೆಗಳ ಮೇಲ್ಮುಖ ಮಡಿಕೆ ಮತ್ತು ಹಳದಿ", "ಸಸ್ಯ ಬೆಳವಣಿಗೆ ಕುಂಠಿತ", "ಹೂ ಉದುರುವಿಕೆ ಮತ್ತು ಕಡಿಮೆ ಇಳುವರಿ"],
            "treatment_steps": ["ಬಿಳಿ ನೊಣ ನಿಯಂತ್ರಿಸಿ", "ಸೋಂಕಿತ ಸಸ್ಯ ತೆಗೆದು ನಾಶ ಮಾಡಿ", "ಪ್ರತಿಫಲಕ ಮಲ್ಚ್ ಬಳಸಿ"],
            "prevention": ["ವೈರಸ್ ನಿರೋಧಕ ತಳಿ ಬಳಸಿ", "ಸಸಿ ಮಡಿಯಲ್ಲಿ ಕೀಟ ನಿರೋಧಕ ಜಾಲ ಅಳವಡಿಸಿ", "ಆರಂಭದಲ್ಲೇ ಬಿಳಿ ನೊಣ ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ"],
            "fertilizer_recommendation": "ಸಮತೋಲನ ಪೋಷಕಾಂಶ; ಒತ್ತಡದ ಸಸ್ಯಗಳು ಹೆಚ್ಚು ಸೋಂಕಿಗೆ ಒಳಗಾಗುತ್ತವೆ.",
        },
    },
    "Tomato_Mosaic Virus": {
        "english": {
            "cause": "Tomato mosaic virus (ToMV) spreads through infected tools, hands, and plant debris.",
            "symptoms": ["Mosaic pattern (light and dark green patches) on leaves", "Leaf distortion and puckering", "Reduced fruit size and quality"],
            "treatment_steps": ["Remove and destroy infected plants immediately", "Disinfect hands and tools with soap/bleach", "There is no chemical cure — prevention is key"],
            "prevention": ["Use certified virus-free seed", "Wash hands before handling plants", "Control aphids which may assist spread"],
            "fertilizer_recommendation": "Maintain good plant nutrition to reduce stress; avoid excess nitrogen.",
        },
        "kannada": {
            "cause": "ಟೊಮೇಟೊ ಮೊಸಾಯಿಕ್ ವೈರಸ್ ಸೋಂಕಿತ ಉಪಕರಣ ಮತ್ತು ಕೈಗಳ ಮೂಲಕ ಹರಡುತ್ತದೆ.",
            "symptoms": ["ಎಲೆಗಳ ಮೇಲೆ ಮೊಸಾಯಿಕ್ ಮಾದರಿ (ತಿಳಿ/ಗಾಢ ಹಸಿರು)", "ಎಲೆ ವಿರೂಪ", "ಹಣ್ಣಿನ ಗಾತ್ರ ಮತ್ತು ಗುಣಮಟ್ಟ ಕಡಿಮೆ"],
            "treatment_steps": ["ಸೋಂಕಿತ ಸಸ್ಯ ತಕ್ಷಣ ತೆಗೆದು ನಾಶ ಮಾಡಿ", "ಕೈ ಮತ್ತು ಉಪಕರಣ ಸ್ವಚ್ಛಗೊಳಿಸಿ", "ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ ಇಲ್ಲ — ತಡೆಗಟ್ಟುವಿಕೆ ಮುಖ್ಯ"],
            "prevention": ["ರೋಗಮುಕ್ತ ಬೀಜ ಬಳಸಿ", "ಸಸ್ಯ ಸ್ಪರ್ಶಕ್ಕೆ ಮೊದಲು ಕೈ ತೊಳೆಯಿರಿ", "ಗಿಡಹೇನು ನಿಯಂತ್ರಿಸಿ"],
            "fertilizer_recommendation": "ಉತ್ತಮ ಪೋಷಕಾಂಶ ಕಾಪಾಡಿ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ.",
        },
    },
    "Tomato_Healthy": {
        "english": {
            "cause": "No disease detected — the tomato plant appears healthy.",
            "symptoms": ["Deep green uniform leaves", "Strong stem growth", "No spots, curling, or lesions"],
            "treatment_steps": ["Continue monitoring", "Maintain watering and fertilization", "Scout weekly for early signs"],
            "prevention": ["Rotate crops annually", "Use disease-resistant varieties", "Maintain field hygiene"],
            "fertilizer_recommendation": "Regular balanced NPK based on growth stage; supplement with calcium to prevent blossom end rot.",
        },
        "kannada": {
            "cause": "ಯಾವುದೇ ರೋಗ ಕಾಣಿಸಿಲ್ಲ — ಟೊಮೇಟೊ ಸಸ್ಯ ಆರೋಗ್ಯಕರ.",
            "symptoms": ["ಗಾಢ ಹಸಿರು ಸಮಾನ ಎಲೆಗಳು", "ಬಲವಾದ ಕಾಂಡ", "ಕಲೆ, ಮಡಿಕೆ ಅಥವಾ ಗಾಯ ಇಲ್ಲ"],
            "treatment_steps": ["ನಿರಂತರ ಮೇಲ್ವಿಚಾರಣೆ", "ನೀರಾವರಿ ಮತ್ತು ಗೊಬ್ಬರ ಕಾಪಾಡಿ", "ವಾರಕ್ಕೊಮ್ಮೆ ಪರಿಶೀಲನೆ"],
            "prevention": ["ವಾರ್ಷಿಕ ಬೆಳೆ ಪರಿವರ್ತನೆ", "ರೋಗ ನಿರೋಧಕ ತಳಿ ಬಳಸಿ", "ಕ್ಷೇತ್ರ ಸ್ವಚ್ಛ ಇಡಿ"],
            "fertilizer_recommendation": "ಬೆಳವಣಿಗೆ ಹಂತಕ್ಕೆ ತಕ್ಕ ಸಮತೋಲನ NPK; ಕ್ಯಾಲ್ಸಿಯಂ ಒದಗಿಸಿ.",
        },
    },
}


def build_recommendation(disease_key: str) -> Dict[str, object]:
    """Return English + Kannada remedy for the given disease key."""
    entry = _REMEDY.get(disease_key)
    if entry:
        return {"english": entry["english"], "kannada": entry["kannada"]}

    # Generic fallback
    generic_en = {
        "cause": f"{disease_key} — symptom patterns may be linked to moisture stress, pathogen pressure, or nutrient imbalance.",
        "symptoms": ["Leaf spots or lesions", "Colour change", "Reduced vigour"],
        "treatment_steps": ["Isolate affected plants", "Apply crop-appropriate fungicide per local guidance", "Improve irrigation and sanitation"],
        "prevention": ["Improve air circulation", "Avoid prolonged leaf wetness", "Follow crop rotation"],
        "fertilizer_recommendation": "Use soil-test-based balanced NPK with micronutrients.",
    }
    generic_kn = {
        "cause": f"{disease_key} — ತೇವಾಂಶ, ರೋಗಾಣು ಅಥವಾ ಪೋಷಕಾಂಶ ಅಸಮತೋಲನದಿಂದ ಕಾಣಿಸಬಹುದು.",
        "symptoms": ["ಎಲೆಗಳ ಮೇಲೆ ಕಲೆ", "ಬಣ್ಣ ಬದಲಾವಣೆ", "ಬೆಳವಣಿಗೆ ಕಡಿಮೆ"],
        "treatment_steps": ["ಪೀಡಿತ ಸಸ್ಯ ಬೇರ್ಪಡಿಸಿ", "ಸ್ಥಳೀಯ ಸಲಹೆಯಂತೆ ಸ್ಪ್ರೇ ಬಳಸಿ", "ನೀರಾವರಿ ಮತ್ತು ಸ್ವಚ್ಛತೆ ಸುಧಾರಿಸಿ"],
        "prevention": ["ಗಾಳಿ ಚಲನೆ ಹೆಚ್ಚಿಸಿ", "ಎಲೆ ತೇವ ತಪ್ಪಿಸಿ", "ಬೆಳೆ ಪರಿವರ್ತನೆ"],
        "fertilizer_recommendation": "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಆಧಾರದ ಸಮತೋಲನ NPK ಮತ್ತು ಸೂಕ್ಷ್ಮಪೋಷಕಾಂಶ ನೀಡಿ.",
    }
    return {"english": generic_en, "kannada": generic_kn}


# ---------------------------------------------------------------------------
# TTS stub
# ---------------------------------------------------------------------------

def _silence_wav_b64(duration: float = 1.2, sr: int = 16_000) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<h", 0) * int(duration * sr))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _map_tts_language(language: str) -> str:
    normalized = language.strip().lower()
    mapping = {
        "kn": "kn-IN",
        "kannada": "kn-IN",
        "en": "en-IN",
        "english": "en-IN",
    }
    return mapping.get(normalized, "kn-IN")


def _extract_audio_base64(response: requests.Response) -> Tuple[str, str]:
    content_type = (response.headers.get("content-type") or "").lower()

    # Sarvam's streaming endpoint typically returns raw audio bytes.
    if content_type.startswith("audio/"):
        return base64.b64encode(response.content).decode("ascii"), content_type.split(";", 1)[0]

    # Some deployments may still return JSON with audio already encoded.
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="TTS provider returned non-JSON/non-audio response") from exc

    mime_type = payload.get("mime_type") or payload.get("mimeType") or "audio/mpeg"
    candidates = [
        payload.get("audio_base64"),
        payload.get("audio"),
        payload.get("data"),
        payload.get("audioContent"),
    ]

    for value in candidates:
        if isinstance(value, dict):
            nested = value.get("base64") or value.get("audio_base64")
            if isinstance(nested, str) and nested:
                value = nested
        if isinstance(value, str) and value:
            if value.startswith("data:audio") and "," in value:
                _, encoded = value.split(",", 1)
                return encoded, mime_type
            return value, mime_type

    raise HTTPException(status_code=502, detail="TTS provider response did not include audio payload")


def _generate_tts_audio(text: str, language: str) -> Tuple[str, str]:
    if not SARVAM_API_KEY:
        raise HTTPException(status_code=500, detail="SARVAM_API_KEY is not configured")

    if not SARVAM_TTS_URL:
        raise HTTPException(status_code=500, detail="SARVAM_TTS_URL is not configured")

    language_code = _map_tts_language(language)
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "target_language_code": language_code,
        "speaker": SARVAM_TTS_VOICE,
        "model": SARVAM_TTS_MODEL,
        "pace": 1.0,
        "speech_sample_rate": SARVAM_TTS_SAMPLE_RATE,
        "output_audio_codec": "mp3",
        "enable_preprocessing": True,
    }

    response = requests.post(
        SARVAM_TTS_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=SARVAM_TIMEOUT_SECONDS,
    )
    if response.ok:
        return _extract_audio_base64(response)

    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    raise HTTPException(status_code=502, detail=f"TTS provider error: {str(detail)[:240]}")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "model": "fine-tuned" if USING_FINETUNED else "imagenet-fallback",
        "device": str(DEVICE),
        "num_classes": len(CLASS_NAMES),
    }


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

    print("\n===== DEBUG =====")
    print("Top 5 probs:", torch.topk(probs, 5))
    print("Max prob:", torch.max(probs).item())
    print("=================\n")

    if USING_FINETUNED:
        crop, disease, confidence, candidates = _predict_finetuned(probs)
    else:
        crop, disease, confidence, candidates = _predict_imagenet_fallback(probs)

    # Build remedy inline — avoids a second round-trip from the frontend.
    # Key format: "<Crop>_<Disease>"  e.g. "Tomato_Early Blight"
    remedy_key = f"{crop}_{disease}"
    remedy     = build_recommendation(remedy_key)

    response = {
        "disease":        disease,
        "confidence":     round(confidence, 4),
        "severity":       _severity(confidence),
        "crop":           crop,
        "top_candidates": candidates,
        "model":          "fine-tuned" if USING_FINETUNED else "imagenet-fallback",
        "remedy":         remedy,
    }

    # Store a lightweight copy in history (omit full remedy to save memory)
    history_entry = {k: v for k, v in response.items() if k != "remedy"}
    history_entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    PREDICTION_HISTORY.append(history_entry)
    if len(PREDICTION_HISTORY) > MAX_HISTORY:
        del PREDICTION_HISTORY[:-MAX_HISTORY]

    return response


@app.get("/history")
def history(limit: int = Query(default=10, ge=1, le=100)) -> List[Dict[str, object]]:
    return PREDICTION_HISTORY[-limit:]


@app.get("/remedy-llm")
def remedy_llm(
    disease: str = Query(..., min_length=1),
    crop:    str = Query(default=""),
) -> Dict[str, object]:
    """Return remedy for a given disease (and optionally crop).

    Preferred call: /remedy-llm?crop=Tomato&disease=Early+Blight
    Fallback call : /remedy-llm?disease=Early+Blight  (fuzzy crop match)
    """
    # 1. Exact compound key lookup  (most precise)
    if crop:
        exact_key = f"{crop}_{disease}"
        if exact_key in _REMEDY:
            return build_recommendation(exact_key)

    # 2. Fuzzy search across all keys
    disease_lower = disease.lower()
    crop_lower    = crop.lower()
    for key in _REMEDY:
        key_lower = key.lower()
        if disease_lower in key_lower and (not crop_lower or crop_lower in key_lower):
            return build_recommendation(key)

    # 3. Generic fallback
    return build_recommendation(disease)


@app.post("/tts")
def tts(payload: TtsRequest) -> Dict[str, str]:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    audio_base64, mime_type = _generate_tts_audio(payload.text, payload.language)
    return {"audio_base64": audio_base64, "mime_type": mime_type}
