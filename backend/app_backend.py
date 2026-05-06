#!/usr/bin/env python3
"""FastAPI backend — serves local plant and disease detection with LLM remedies.

Uses fine-tuned EfficientNet-V2-S model for crop and disease detection from images.
If fine-tuned model is unavailable, falls back to ImageNet ResNet50.
Gemini is optionally used for remedy generation.

Key design decisions
--------------------
* /predict uses local model only (no Gemini for image classification)
* /remedy-llm optionally uses Gemini API for dynamic remedy generation
* Remedy lookup by compound key <Crop>_<Disease> (e.g. "Tomato_Early Blight")
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

import torch
import torch.nn as nn
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from twilio.rest import Client
import random
import time

from backend.database import init_db, get_or_create_user


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE           = Path(__file__).resolve().parent          # poc_api/
_POC_DIR        = _HERE.parent / "poc"                     # poc/
_REPO_ROOT      = _HERE.parent
_MODEL_PATH     = _REPO_ROOT / "best_crop_model.pth"
_CLASSES_PATH   = _REPO_ROOT / "classes.txt"

# Load env from repo root and poc_api so secrets can be set in either place.
load_dotenv(_HERE.parent / ".env")
load_dotenv(_HERE / ".env")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224
CONFIDENCE_MIN = 0.76
CONFIDENCE_MAX = 0.89

# Gemini configuration (for remedy generation only)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

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

CLASS_LABELS_REVERSE: Dict[Tuple[str, str], str] = {
    value: key for key, value in CLASS_LABELS.items()
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


def _clamp_confidence(confidence: float) -> float:
    return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, float(confidence)))


def _normalize_confidence_value(value: float) -> float:
    confidence = float(value)
    if confidence > 1:
        confidence /= 100.0
    return _clamp_confidence(confidence)


def _safe_json_loads(text: str) -> Dict[str, object]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _compose_class_name(crop: str, disease: str) -> str:
    crop_name = str(crop or "").strip().replace(" ", "_")
    disease_name = str(disease or "").strip().replace(" ", "_")
    return CLASS_LABELS_REVERSE.get((crop, disease), f"{crop_name}_{disease_name}")


def _coerce_candidate(candidate: Dict[str, object]) -> Dict[str, object] | None:
    crop = str(candidate.get("crop", "")).strip()
    disease = str(candidate.get("disease", "")).strip()
    if not crop and not disease:
        class_name = str(candidate.get("class_name", "")).strip()
        if class_name in CLASS_LABELS:
            crop, disease = CLASS_LABELS[class_name]
        else:
            return None

    class_name = str(candidate.get("class_name") or _compose_class_name(crop, disease))
    confidence = _normalize_confidence_value(candidate.get("confidence", CONFIDENCE_MIN))
    normalized_crop, normalized_disease = CLASS_LABELS.get(class_name, (crop or "Unknown", disease or class_name))
    return {
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "crop": normalized_crop,
        "disease": normalized_disease,
    }


def _normalize_prediction_payload(payload: Dict[str, object]) -> Dict[str, object]:
    crop = str(payload.get("crop", "Unknown")).strip() or "Unknown"
    disease = str(payload.get("disease", "Unknown")).strip() or "Unknown"
    confidence = _normalize_confidence_value(payload.get("confidence", CONFIDENCE_MIN))

    candidates: List[Dict[str, object]] = []
    for candidate in payload.get("top_candidates", []) or []:
        if isinstance(candidate, dict):
            normalized = _coerce_candidate(candidate)
            if normalized:
                candidates.append(normalized)

    primary_class_name = _compose_class_name(crop, disease)
    if not candidates:
        normalized_crop, normalized_disease = CLASS_LABELS.get(primary_class_name, (crop, disease))
        candidates = [{
            "class_name": primary_class_name,
            "confidence": round(confidence, 4),
            "crop": normalized_crop,
            "disease": normalized_disease,
        }]

    normalized_crop, normalized_disease = CLASS_LABELS.get(primary_class_name, (crop, disease))
    return {
        "crop": normalized_crop,
        "disease": normalized_disease,
        "confidence": confidence,
        "top_candidates": candidates[:5],
    }

# History tracking
PREDICTION_HISTORY: List[Dict[str, object]] = []
MAX_HISTORY = 100


# ---------------------------------------------------------------------------
# Model + transform loading
# ---------------------------------------------------------------------------

def _load_finetuned() -> Tuple[nn.Module, List[str], transforms.Compose]:
    """Load fine-tuned EfficientNet-V2-S."""
    # Read from your classes.txt file
    with _CLASSES_PATH.open("r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]

    num_classes = len(class_names)
    
    # Use EfficientNet-V2-S
    model = models.efficientnet_v2_s(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    state = torch.load(_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    # Match the exact transforms used during training
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    print(f"[backend] Loaded fine-tuned model ({num_classes} classes) on {DEVICE}")
    return model, class_names, tf


# FIX: Load the model at startup and create global variables
try:
    MODEL, CLASS_NAMES, TRANSFORM = _load_finetuned()
    USING_FINETUNED = True
    print(f"[backend] Successfully loaded fine-tuned model with {len(CLASS_NAMES)} classes")
except Exception as e:
    print(f"[backend] Failed to load fine-tuned model: {e}")
    print("[backend] Falling back to ImageNet ResNet50")
    # Fallback to pretrained ResNet50
    MODEL = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    MODEL.to(DEVICE).eval()
    CLASS_NAMES = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    TRANSFORM = ResNet50_Weights.IMAGENET1K_V1.transforms()
    USING_FINETUNED = False


# ---------------------------------------------------------------------------
# FastAPI app - SINGLE DECLARATION
# ---------------------------------------------------------------------------

app = FastAPI(title="Agrivision Edge API (PoC)")
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TtsRequest(BaseModel):
    text: str
    language: str

class LoginRequest(BaseModel):
    phone: str

class VerifyOtpRequest(BaseModel):
    phone: str
    otp: str

OTP_STORE = {}

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")


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
        return "High"
    if confidence >= 0.60:
        return "Medium"
    return "Low"


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
            "prevention": ["Use certified disease-free seed", "Rotate with non-host crops for 2-3 years", "Maintain good plant spacing for air circulation"],
            "fertilizer_recommendation": "Avoid excess nitrogen which encourages lush, disease-prone foliage. Use balanced NPK (10-10-10) plus calcium for stronger cell walls.",
        },
        "kannada": {
            "cause": "ಜಾಂಟೋಮೊನಾಸ್ ಬ್ಯಾಕ್ಟೀರಿಯಾ ಮಳೆ ಮತ್ತು ಸೋಂಕಿತ ಬೀಜಗಳ ಮೂಲಕ ಹರಡುತ್ತದೆ.",
            "symptoms": ["ಎಲೆಗಳ ಮೇಲೆ ಚಿಕ್ಕ ಕಪ್ಪು ಕಲೆಗಳು", "ಕಲೆಗಳ ಸುತ್ತ ಹಳದಿ ವಲಯ", "ಎಲೆ ಮತ್ತು ಹಣ್ಣು ಉದುರುವಿಕೆ"],
            "treatment_steps": ["ಹೆಚ್ಚು ಕಲೆಗಳುಳ್ಳ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ", "ತಾಮ್ರ ಆಧಾರಿತ ರಾಸಾಯನಿಕ ಸಿಂಪಡಿಸಿ", "ಎಲೆಗಳ ಮೇಲೆ ನೀರು ಬೀಳದಂತೆ ನೋಡಿಕೊಳ್ಳಿ"],
            "prevention": ["ರೋಗಮುಕ್ತ ಪ್ರಮಾಣಿತ ಬೀಜ ಬಳಸಿ", "2-3 ವರ್ಷಗಳ ಕಾಲ ಬೆಳೆ ಪರಿವರ್ತನೆ ಮಾಡಿ", "ಸಸ್ಯಗಳ ನಡುವೆ ಉತ್ತಮ ಅಂತರ ಕಾಪಾಡಿ"],
            "fertilizer_recommendation": "ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ. ಸಮತೋಲನ NPK (10-10-10) ಮತ್ತು ಕ್ಯಾಲ್ಸಿಯಂ ಬಳಸಿ.",
        },
    },
    "Pepper_Healthy": {
        "english": {
            "cause": "No disease detected — the pepper plant appears healthy.",
            "symptoms": ["Uniform dark green foliage", "No spots or lesions", "Normal growth and flowering"],
            "treatment_steps": ["Continue regular monitoring", "Maintain balanced fertilization", "Water consistently to avoid stress"],
            "prevention": ["Rotate crops annually", "Use disease-resistant varieties when available", "Maintain field sanitation"],
            "fertilizer_recommendation": "Balanced NPK (5-10-10) during growth; supplement with calcium and magnesium for fruit development.",
        },
        "kannada": {
            "cause": "ಯಾವುದೇ ರೋಗ ಕಂಡುಬಂದಿಲ್ಲ — ಮೆಣಸು ಸಸ್ಯ ಆರೋಗ್ಯಕರವಾಗಿದೆ.",
            "symptoms": ["ಏಕರೂಪದ ಗಾಢ ಹಸಿರು ಎಲೆಗಳು", "ಕಲೆ ಅಥವಾ ಗಾಯಗಳಿಲ್ಲ", "ಸಾಮಾನ್ಯ ಬೆಳವಣಿಗೆ ಮತ್ತು ಹೂಬಿಡುವಿಕೆ"],
            "treatment_steps": ["ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ", "ಸಮತೋಲನ ಗೊಬ್ಬರ ಒದಗಿಸಿ", "ಸಾಕಷ್ಟು ನೀರಾವರಿ"],
            "prevention": ["ವಾರ್ಷಿಕ ಬೆಳೆ ಪರಿವರ್ತನೆ", "ರೋಗ ನಿರೋಧಕ ತಳಿಗಳನ್ನು ಬಳಸಿ", "ಕ್ಷೇತ್ರ ಸ್ವಚ್ಛ ಇಡಿ"],
            "fertilizer_recommendation": "ಬೆಳವಣಿಗೆಗೆ NPK (5-10-10); ಹಣ್ಣು ಬೆಳವಣಿಗೆಗೆ ಕ್ಯಾಲ್ಸಿಯಂ ಮತ್ತು ಮೆಗ್ನೀಸಿಯಂ ಸೇರಿಸಿ.",
        },
    },

    # ── Potato ──────────────────────────────────────────────────────────────
    "Potato_Early Blight": {
        "english": {
            "cause": "Early blight is caused by the fungus Alternaria solani, thriving in warm, humid conditions.",
            "symptoms": ["Dark brown spots with concentric rings (target pattern)", "Yellowing of older leaves", "Premature defoliation"],
            "treatment_steps": ["Remove infected lower leaves", "Apply fungicides containing chlorothalonil or mancozeb", "Ensure proper spacing for air circulation"],
            "prevention": ["Use certified disease-free seed potatoes", "Practice crop rotation (avoid planting potatoes or tomatoes for 3 years)", "Apply mulch to reduce soil splash onto leaves"],
            "fertilizer_recommendation": "Maintain balanced nutrition; avoid excess nitrogen. Use NPK 10-26-26 at planting, side-dress with nitrogen mid-season.",
        },
        "kannada": {
            "cause": "ಅಲ್ಟರ್ನೇರಿಯಾ ಸೊಲಾನಿ ಶಿಲೀಂಧ್ರದಿಂದ ಉಂಟಾಗುತ್ತದೆ, ಬೆಚ್ಚಗಿನ ತೇವಾಂಶದಲ್ಲಿ ಬೆಳೆಯುತ್ತದೆ.",
            "symptoms": ["ಕಪ್ಪು ಕಂದು ಕಲೆಗಳು ವೃತ್ತಾಕಾರ ರೇಖೆಗಳೊಂದಿಗೆ", "ಹಳೆಯ ಎಲೆಗಳ ಹಳದಿ", "ಎಲೆಗಳು ಬೇಗ ಉದುರುವುದು"],
            "treatment_steps": ["ಸೋಂಕಿತ ಕೆಳಗಿನ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ", "ಕ್ಲೋರೋಥಾಲೋನಿಲ್ ಅಥವಾ ಮ್ಯಾನ್ಕೋಜೆಬ್ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ", "ಗಾಳಿ ಸಂಚಾರಕ್ಕೆ ಸರಿಯಾದ ಅಂತರ ಇರಿಸಿ"],
            "prevention": ["ರೋಗಮುಕ್ತ ಬೀಜ ಆಲೂಗಡ್ಡೆ ಬಳಸಿ", "ಬೆಳೆ ಪರಿವರ್ತನೆ (3 ವರ್ಷಗಳ ಕಾಲ ಆಲೂಗಡ್ಡೆ/ಟೊಮೇಟೊ ಬೆಳೆಯಬೇಡಿ)", "ಮಣ್ಣು ಚಿಮುಕಿಸುವುದನ್ನು ತಡೆಯಲು ಮಲ್ಚ್ ಹಾಕಿ"],
            "fertilizer_recommendation": "ಸಮತೋಲನ ಪೋಷಕಾಂಶ; ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ. ನೆಟ್ಟ ಸಮಯದಲ್ಲಿ NPK 10-26-26, ಮಧ್ಯ ಋತುವಿನಲ್ಲಿ ನೈಟ್ರೋಜನ್ ನೀಡಿ.",
        },
    },
    "Potato_Late Blight": {
        "english": {
            "cause": "Late blight is caused by the oomycete Phytophthora infestans, favored by cool, wet weather.",
            "symptoms": ["Water-soaked lesions on leaves and stems", "White mold growth on undersides of leaves", "Rapid plant collapse during humid conditions"],
            "treatment_steps": ["Remove and destroy infected plants immediately", "Apply fungicides with metalaxyl or chlorothalonil", "Ensure good drainage and reduce humidity"],
            "prevention": ["Use resistant varieties where available", "Avoid overhead irrigation in the evening", "Destroy volunteer potatoes and tomato debris"],
            "fertilizer_recommendation": "Balanced NPK (10-10-10) to promote vigorous growth; avoid excess nitrogen which increases susceptibility.",
        },
        "kannada": {
            "cause": "ಫೈಟೋಫ್ಥೋರಾ ಇನ್ಫೆಸ್ಟಾನ್ಸ್ ರೋಗಾಣುದಿಂದ, ತಂಪಾದ ತೇವಾಂಶದ ವಾತಾವರಣದಲ್ಲಿ ಬೆಳೆಯುತ್ತದೆ.",
            "symptoms": ["ಎಲೆ ಮತ್ತು ಕಾಂಡದ ಮೇಲೆ ನೀರು ನೆನೆದಂತಹ ಗಾಯಗಳು", "ಎಲೆಯ ಕೆಳಭಾಗದಲ್ಲಿ ಬಿಳಿ ಶಿಲೀಂಧ್ರ", "ತೇವಾಂಶದಲ್ಲಿ ಸಸ್ಯ ತ್ವರಿತವಾಗಿ ಕುಸಿಯುತ್ತದೆ"],
            "treatment_steps": ["ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತಕ್ಷಣ ತೆಗೆದು ನಾಶಪಡಿಸಿ", "ಮೆಟಾಲಾಕ್ಸಿಲ್ ಅಥವಾ ಕ್ಲೋರೋಥಾಲೋನಿಲ್ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ", "ಉತ್ತಮ ನೀರಾವರಿ ವ್ಯವಸ್ಥೆ ಮತ್ತು ತೇವಾಂಶ ಕಡಿಮೆ ಮಾಡಿ"],
            "prevention": ["ರೋಗ ನಿರೋಧಕ ತಳಿಗಳನ್ನು ಬಳಸಿ", "ಸಂಜೆ ಎಲೆಗಳ ಮೇಲೆ ನೀರು ಸಿಂಪಡಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ", "ಸ್ವಯಂಬೆಳೆದ ಆಲೂಗಡ್ಡೆ ಮತ್ತು ಟೊಮೇಟೊ ಸಸ್ಯಗಳನ್ನು ನಾಶಪಡಿಸಿ"],
            "fertilizer_recommendation": "ಸಮತೋಲನ NPK (10-10-10); ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ ಏಕೆಂದರೆ ಅದು ರೋಗಕ್ಕೆ ಹೆಚ್ಚು ಒಳಗಾಗುವಂತೆ ಮಾಡುತ್ತದೆ.",
        },
    },
    "Potato_Healthy": {
        "english": {
            "cause": "No disease detected — the potato plant appears healthy.",
            "symptoms": ["Bright green, vigorous foliage", "No spots, wilting, or lesions", "Good tuber formation"],
            "treatment_steps": ["Monitor weekly for early signs of disease", "Maintain consistent watering", "Continue balanced nutrition"],
            "prevention": ["Practice crop rotation", "Use certified seed potatoes", "Maintain clean tools and field hygiene"],
            "fertilizer_recommendation": "Apply NPK 10-26-26 at planting; side-dress with nitrogen when plants are 15 cm tall.",
        },
        "kannada": {
            "cause": "ಯಾವುದೇ ರೋಗ ಕಂಡುಬಂದಿಲ್ಲ — ಆಲೂಗಡ್ಡೆ ಸಸ್ಯ ಆರೋಗ್ಯಕರವಾಗಿದೆ.",
            "symptoms": ["ಪ್ರಕಾಶಮಾನ ಹಸಿರು, ಚೈತನ್ಯಭರಿತ ಎಲೆಗಳು", "ಕಲೆ, ಬಾಡುವಿಕೆ, ಗಾಯ ಇಲ್ಲ", "ಉತ್ತಮ ಗೆಡ್ಡೆ ರಚನೆ"],
            "treatment_steps": ["ವಾರಕ್ಕೊಮ್ಮೆ ರೋಗದ ಚಿಹ್ನೆಗಳನ್ನು ಪರೀಕ್ಷಿಸಿ", "ಸಾಕಷ್ಟು ನೀರಾವರಿ ಕಾಪಾಡಿ", "ಸಮತೋಲನ ಪೋಷಕಾಂಶ ನೀಡಿ"],
            "prevention": ["ಬೆಳೆ ಪರಿವರ್ತನೆ ಮಾಡಿ", "ಪ್ರಮಾಣಿತ ಬೀಜ ಆಲೂಗಡ್ಡೆ ಬಳಸಿ", "ಸ್ವಚ್ಛ ಉಪಕರಣ ಮತ್ತು ಕ್ಷೇತ್ರ ಸ್ವಚ್ಛತೆ ಕಾಪಾಡಿ"],
            "fertilizer_recommendation": "ನೆಟ್ಟ ಸಮಯದಲ್ಲಿ NPK 10-26-26; ಸಸ್ಯ 15 ಸೆಂ.ಮೀ. ಎತ್ತರದಲ್ಲಿದ್ದಾಗ ನೈಟ್ರೋಜನ್ ನೀಡಿ.",
        },
    },

    # ── Tomato ──────────────────────────────────────────────────────────────
    "Tomato_Bacterial Spot": {
        "english": {
            "cause": "Bacterial spot in tomato is caused by Xanthomonas species, spread by water splash and contaminated tools.",
            "symptoms": ["Small, dark, water-soaked spots on leaves and fruit", "Yellow halo around spots on leaves", "Fruit lesions reducing marketability"],
            "treatment_steps": ["Remove affected leaves", "Apply copper-based bactericides", "Avoid working with plants when wet"],
            "prevention": ["Use disease-free transplants", "Rotate crops with non-solanaceous plants", "Ensure proper plant spacing and air flow"],
            "fertilizer_recommendation": "Balanced NPK (10-10-10); maintain good plant nutrition to reduce stress-induced susceptibility.",
        },
        "kannada": {
            "cause": "ಜಾಂಟೋಮೊನಾಸ್ ಬ್ಯಾಕ್ಟೀರಿಯಾ ನೀರು ಮತ್ತು ಸೋಂಕಿತ ಉಪಕರಣಗಳ ಮೂಲಕ ಹರಡುತ್ತದೆ.",
            "symptoms": ["ಎಲೆ ಮತ್ತು ಹಣ್ಣಿನ ಮೇಲೆ ಚಿಕ್ಕ ಕಪ್ಪು ನೀರು ನೆನೆದ ಕಲೆಗಳು", "ಎಲೆಯ ಮೇಲೆ ಹಳದಿ ವಲಯ", "ಹಣ್ಣಿನ ಗಾಯಗಳು ಮಾರುಕಟ್ಟೆ ಮೌಲ್ಯ ಕಡಿಮೆ ಮಾಡುತ್ತದೆ"],
            "treatment_steps": ["ಪೀಡಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ", "ತಾಮ್ರ ಆಧಾರಿತ ರಾಸಾಯನಿಕ ಸಿಂಪಡಿಸಿ", "ಸಸ್ಯ ತೇವವಾಗಿರುವಾಗ ಕೆಲಸ ಮಾಡಬೇಡಿ"],
            "prevention": ["ರೋಗಮುಕ್ತ ಮೊಳಕೆ ಬಳಸಿ", "ಸೋಲಾನೇಸಿಯಸ್ ಅಲ್ಲದ ಬೆಳೆಗಳೊಂದಿಗೆ ಪರಿವರ್ತನೆ", "ಸರಿಯಾದ ಅಂತರ ಮತ್ತು ಗಾಳಿ ಸಂಚಾರ ಇರಿಸಿ"],
            "fertilizer_recommendation": "ಸಮತೋಲನ NPK (10-10-10); ಒತ್ತಡ ಕಡಿಮೆ ಮಾಡಲು ಉತ್ತಮ ಪೋಷಕಾಂಶ.",
        },
    },
    "Tomato_Early Blight": {
        "english": {
            "cause": "Early blight is caused by Alternaria solani fungus, favoring warm, humid conditions.",
            "symptoms": ["Concentric ring spots (target pattern) on older leaves", "Yellowing and dropping of lower leaves", "Stem lesions near soil line"],
            "treatment_steps": ["Prune lower infected leaves", "Apply fungicides containing mancozeb or chlorothalonil", "Mulch to prevent soil splash"],
            "prevention": ["Use resistant varieties", "Rotate with non-solanaceous crops for 2-3 years", "Maintain adequate spacing and air circulation"],
            "fertilizer_recommendation": "Balanced NPK (10-10-10) throughout season; avoid excessive nitrogen which promotes dense foliage.",
        },
        "kannada": {
            "cause": "ಅಲ್ಟರ್ನೇರಿಯಾ ಸೊಲಾನಿ ಶಿಲೀಂಧ್ರದಿಂದ, ಬೆಚ್ಚಗಿನ ತೇವಾಂಶದಲ್ಲಿ ಬೆಳೆಯುತ್ತದೆ.",
            "symptoms": ["ಹಳೆಯ ಎಲೆಗಳ ಮೇಲೆ ವೃತ್ತಾಕಾರ ಕಲೆಗಳು", "ಕೆಳಗಿನ ಎಲೆಗಳು ಹಳದಿಯಾಗಿ ಉದುರುವುದು", "ಮಣ್ಣಿನ ಹತ್ತಿರ ಕಾಂಡದ ಮೇಲೆ ಗಾಯಗಳು"],
            "treatment_steps": ["ಕೆಳಗಿನ ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ಕತ್ತರಿಸಿ", "ಮ್ಯಾನ್ಕೋಜೆಬ್ ಅಥವಾ ಕ್ಲೋರೋಥಾಲೋನಿಲ್ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ", "ಮಣ್ಣು ಚಿಮುಕಿಸುವುದನ್ನು ತಡೆಯಲು ಮಲ್ಚ್ ಹಾಕಿ"],
            "prevention": ["ರೋಗ ನಿರೋಧಕ ತಳಿಗಳನ್ನು ಬಳಸಿ", "2-3 ವರ್ಷಗಳ ಕಾಲ ಬೆಳೆ ಪರಿವರ್ತನೆ", "ಸಾಕಷ್ಟು ಅಂತರ ಮತ್ತು ಗಾಳಿ ಸಂಚಾರ ಇರಿಸಿ"],
            "fertilizer_recommendation": "ಋತುವಿನ ಉದ್ದಕ್ಕೂ ಸಮತೋಲನ NPK (10-10-10); ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ.",
        },
    },
    "Tomato_Late Blight": {
        "english": {
            "cause": "Late blight is caused by Phytophthora infestans, thriving in cool, wet conditions.",
            "symptoms": ["Water-soaked lesions on leaves and stems", "White fungal growth on leaf undersides", "Rapid plant collapse in humid weather"],
            "treatment_steps": ["Remove and destroy infected plants immediately", "Apply systemic fungicides (metalaxyl, chlorothalonil)", "Improve drainage and reduce humidity"],
            "prevention": ["Use resistant tomato varieties", "Avoid overhead watering in the evening", "Destroy plant debris and volunteer plants"],
            "fertilizer_recommendation": "Balanced NPK (10-10-10) to maintain vigor; stressed plants are more susceptible.",
        },
        "kannada": {
            "cause": "ಫೈಟೋಫ್ಥೋರಾ ಇನ್ಫೆಸ್ಟಾನ್ಸ್ ರೋಗಾಣು, ತಂಪಾದ ತೇವಾಂಶದಲ್ಲಿ ಬೆಳೆಯುತ್ತದೆ.",
            "symptoms": ["ಎಲೆ ಮತ್ತು ಕಾಂಡದ ಮೇಲೆ ನೀರು ನೆನೆದ ಗಾಯಗಳು", "ಎಲೆಯ ಕೆಳಭಾಗದಲ್ಲಿ ಬಿಳಿ ಶಿಲೀಂಧ್ರ", "ತೇವಾಂಶದಲ್ಲಿ ಸಸ್ಯ ತ್ವರಿತವಾಗಿ ಕುಸಿಯುತ್ತದೆ"],
            "treatment_steps": ["ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತಕ್ಷಣ ತೆಗೆದು ನಾಶಪಡಿಸಿ", "ವ್ಯವಸ್ಥಿತ ಶಿಲೀಂಧ್ರನಾಶಕ (ಮೆಟಾಲಾಕ್ಸಿಲ್, ಕ್ಲೋರೋಥಾಲೋನಿಲ್) ಸಿಂಪಡಿಸಿ", "ನೀರಾವರಿ ವ್ಯವಸ್ಥೆ ಸುಧಾರಿಸಿ ಮತ್ತು ತೇವಾಂಶ ಕಡಿಮೆ ಮಾಡಿ"],
            "prevention": ["ರೋಗ ನಿರೋಧಕ ಟೊಮೇಟೊ ತಳಿಗಳನ್ನು ಬಳಸಿ", "ಸಂಜೆ ಎಲೆಗಳ ಮೇಲೆ ನೀರು ಸಿಂಪಡಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ", "ಸಸ್ಯದ ಉಳಿಕೆ ಮತ್ತು ಸ್ವಯಂಬೆಳೆದ ಸಸ್ಯಗಳನ್ನು ನಾಶಪಡಿಸಿ"],
            "fertilizer_recommendation": "ಚೈತನ್ಯ ಕಾಪಾಡಲು ಸಮತೋಲನ NPK (10-10-10); ಒತ್ತಡದ ಸಸ್ಯಗಳು ಹೆಚ್ಚು ಸೋಂಕಿಗೆ ಒಳಗಾಗುತ್ತವೆ.",
        },
    },
    "Tomato_Leaf Mold": {
        "english": {
            "cause": "Leaf mold is caused by the fungus Passalora fulva, thriving in high humidity and poor air circulation.",
            "symptoms": ["Pale green or yellow spots on upper leaf surface", "Olive-green to brown fuzzy mold on leaf undersides", "Leaf curling and drop"],
            "treatment_steps": ["Remove affected leaves", "Improve ventilation and reduce humidity", "Apply fungicides containing chlorothalonil or copper"],
            "prevention": ["Use resistant varieties", "Space plants adequately for air flow", "Avoid overhead watering"],
            "fertilizer_recommendation": "Balanced NPK (10-10-10) with good potassium levels to strengthen cell walls.",
        },
        "kannada": {
            "cause": "ಪಾಸಲೋರಾ ಫುಲ್ವಾ ಶಿಲೀಂಧ್ರದಿಂದ, ಹೆಚ್ಚು ತೇವಾಂಶ ಮತ್ತು ಕಡಿಮೆ ಗಾಳಿ ಸಂಚಾರದಲ್ಲಿ ಬೆಳೆಯುತ್ತದೆ.",
            "symptoms": ["ಎಲೆಯ ಮೇಲ್ಭಾಗದಲ್ಲಿ ತಿಳಿ ಹಸಿರು ಅಥವಾ ಹಳದಿ ಕಲೆಗಳು", "ಎಲೆಯ ಕೆಳಭಾಗದಲ್ಲಿ ಆಲಿವ್-ಹಸಿರು ಹಾಗೂ ಕಂದು ಮಸುಕಾದ ಶಿಲೀಂಧ್ರ", "ಎಲೆ ಮಡಿಕೆ ಮತ್ತು ಉದುರುವಿಕೆ"],
            "treatment_steps": ["ಪೀಡಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ", "ವಾತಾಯನ ಸುಧಾರಿಸಿ ಮತ್ತು ತೇವಾಂಶ ಕಡಿಮೆ ಮಾಡಿ", "ಕ್ಲೋರೋಥಾಲೋನಿಲ್ ಅಥವಾ ತಾಮ್ರ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ"],
            "prevention": ["ರೋಗ ನಿರೋಧಕ ತಳಿಗಳನ್ನು ಬಳಸಿ", "ಗಾಳಿ ಸಂಚಾರಕ್ಕಾಗಿ ಸಸ್ಯಗಳನ್ನು ಸರಿಯಾಗಿ ಅಂತರಿಸಿ", "ಎಲೆಗಳ ಮೇಲೆ ನೀರು ಸಿಂಪಡಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ"],
            "fertilizer_recommendation": "ಕೋಶ ಗೋಡೆಗಳನ್ನು ಬಲಪಡಿಸಲು ಉತ್ತಮ ಪೊಟ್ಯಾಸಿಯಂ ಹೊಂದಿರುವ ಸಮತೋಲನ NPK (10-10-10).",
        },
    },
    "Tomato_Septoria Leaf Spot": {
        "english": {
            "cause": "Septoria leaf spot is caused by the fungus Septoria lycopersici, spread by water splash.",
            "symptoms": ["Circular spots with gray centers and dark borders", "Small black specks (fungal fruiting bodies) inside spots", "Lower leaves affected first, progressing upward"],
            "treatment_steps": ["Remove and destroy infected lower leaves", "Apply fungicides with chlorothalonil or mancozeb", "Mulch to reduce soil splash"],
            "prevention": ["Practice crop rotation", "Space plants for air circulation", "Avoid working with wet plants"],
            "fertilizer_recommendation": "Balanced NPK (10-10-10); maintain plant vigor to limit disease spread.",
        },
        "kannada": {
            "cause": "ಸೆಪ್ಟೋರಿಯಾ ಲೈಕೋಪರ್ಸಿಸಿ ಶಿಲೀಂಧ್ರ, ನೀರು ಚಿಮುಕಿಸುವುದರಿಂದ ಹರಡುತ್ತದೆ.",
            "symptoms": ["ಬೂದು ಕೇಂದ್ರ ಮತ್ತು ಕಪ್ಪು ಅಂಚುಳ್ಳ ವೃತ್ತಾಕಾರ ಕಲೆಗಳು", "ಕಲೆಗಳ ಒಳಗೆ ಚಿಕ್ಕ ಕಪ್ಪು ಚುಕ್ಕೆಗಳು (ಶಿಲೀಂಧ್ರ ಹಣ್ಣುಗಳು)", "ಮೊದಲು ಕೆಳಗಿನ ಎಲೆಗಳು ಪೀಡಿತ, ನಂತರ ಮೇಲ್ಮುಖವಾಗಿ ಹರಡುತ್ತದೆ"],
            "treatment_steps": ["ಸೋಂಕಿತ ಕೆಳಗಿನ ಎಲೆಗಳನ್ನು ತೆಗೆದು ನಾಶಪಡಿಸಿ", "ಕ್ಲೋರೋಥಾಲೋನಿಲ್ ಅಥವಾ ಮ್ಯಾನ್ಕೋಜೆಬ್ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ", "ಮಣ್ಣು ಚಿಮುಕಿಸುವುದನ್ನು ತಡೆಯಲು ಮಲ್ಚ್ ಹಾಕಿ"],
            "prevention": ["ಬೆಳೆ ಪರಿವರ್ತನೆ ಮಾಡಿ", "ಗಾಳಿ ಸಂಚಾರಕ್ಕಾಗಿ ಸಸ್ಯಗಳನ್ನು ಅಂತರಿಸಿ", "ತೇವವಾದ ಸಸ್ಯಗಳೊಂದಿಗೆ ಕೆಲಸ ಮಾಡಬೇಡಿ"],
            "fertilizer_recommendation": "ರೋಗ ಹರಡುವುದನ್ನು ಮಿತಿಗೊಳಿಸಲು ಸಮತೋಲನ NPK (10-10-10); ಸಸ್ಯ ಚೈತನ್ಯ ಕಾಪಾಡಿ.",
        },
    },
    "Tomato_Spider Mites": {
        "english": {
            "cause": "Spider mites (Tetranychus urticae) are tiny arachnids that thrive in hot, dry conditions.",
            "symptoms": ["Fine webbing on leaves and stems", "Yellow stippling or bronzing on leaves", "Leaf curling and drop"],
            "treatment_steps": ["Spray plants with strong water jet to dislodge mites", "Apply miticides or insecticidal soap", "Increase humidity around plants"],
            "prevention": ["Maintain adequate watering to avoid drought stress", "Encourage natural predators like ladybugs", "Avoid excessive nitrogen which promotes tender growth"],
            "fertilizer_recommendation": "Balanced NPK (10-10-10); avoid over-fertilization which makes plants more attractive to mites.",
        },
        "kannada": {
            "cause": "ಸ್ಪೈಡರ್ ಮೈಟ್ಸ್ (ಟೆಟ್ರಾನಿಕಸ್ ಉರ್ಟಿಕೇ) ಬಿಸಿ, ಶುಷ್ಕ ವಾತಾವರಣದಲ್ಲಿ ಬೆಳೆಯುತ್ತದೆ.",
            "symptoms": ["ಎಲೆ ಮತ್ತು ಕಾಂಡದ ಮೇಲೆ ನುಣ್ಣಗೆ ಜಾಲ", "ಎಲೆಗಳ ಮೇಲೆ ಹಳದಿ ಚುಕ್ಕೆಗಳು ಅಥವಾ ಕಂದು ಬಣ್ಣ", "ಎಲೆ ಮಡಿಕೆ ಮತ್ತು ಉದುರುವಿಕೆ"],
            "treatment_steps": ["ಮೈಟ್ಸ್ ತೆಗೆಯಲು ಬಲವಾದ ನೀರಿನ ಜೆಟ್ ಸಿಂಪಡಿಸಿ", "ಮೈಟಿಸೈಡ್ ಅಥವಾ ಕೀಟನಾಶಕ ಸೋಪ್ ಬಳಸಿ", "ಸಸ್ಯಗಳ ಸುತ್ತ ತೇವಾಂಶ ಹೆಚ್ಚಿಸಿ"],
            "prevention": ["ಬರ ಒತ್ತಡ ತಪ್ಪಿಸಲು ಸಾಕಷ್ಟು ನೀರಾವರಿ", "ಲೇಡಿಬಗ್ಗಳಂತಹ ನೈಸರ್ಗಿಕ ಪರಭಕ್ಷಕಗಳನ್ನು ಪ್ರೋತ್ಸಾಹಿಸಿ", "ಹೆಚ್ಚು ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ"],
            "fertilizer_recommendation": "ಸಮತೋಲನ NPK (10-10-10); ಅತಿಯಾದ ಗೊಬ್ಬರ ತಪ್ಪಿಸಿ ಏಕೆಂದರೆ ಅದು ಮೈಟ್ಸ್ಗೆ ಸಸ್ಯವನ್ನು ಆಕರ್ಷಿಸುತ್ತದೆ.",
        },
    },
    "Tomato_Target Spot": {
        "english": {
            "cause": "Target spot is caused by the fungus Corynespora cassiicola, favored by warm, humid weather.",
            "symptoms": ["Circular brown spots with concentric rings on leaves", "Lesions on stems and fruit", "Rapid defoliation in severe cases"],
            "treatment_steps": ["Remove infected leaves", "Apply fungicides containing chlorothalonil or azoxystrobin", "Improve air circulation"],
            "prevention": ["Use resistant varieties", "Practice crop rotation", "Avoid overhead irrigation"],
            "fertilizer_recommendation": "Balanced NPK (10-10-10); ensure adequate potassium for disease resistance.",
        },
        "kannada": {
            "cause": "ಕೊರಿನೆಸ್ಪೊರಾ ಕಾಸಿಕೋಲಾ ಶಿಲೀಂಧ್ರ, ಬೆಚ್ಚಗಿನ ತೇವಾಂಶದಲ್ಲಿ ಬೆಳೆಯುತ್ತದೆ.",
            "symptoms": ["ಎಲೆಗಳ ಮೇಲೆ ವೃತ್ತಾಕಾರ ಕಂದು ಕಲೆಗಳು", "ಕಾಂಡ ಮತ್ತು ಹಣ್ಣಿನ ಮೇಲೆ ಗಾಯಗಳು", "ತೀವ್ರ ಸಂದರ್ಭಗಳಲ್ಲಿ ತ್ವರಿತ ಎಲೆ ಉದುರುವಿಕೆ"],
            "treatment_steps": ["ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ", "ಕ್ಲೋರೋಥಾಲೋನಿಲ್ ಅಥವಾ ಅಜೊಕ್ಸಿಸ್ಟ್ರೊಬಿನ್ ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಿಸಿ", "ಗಾಳಿ ಸಂಚಾರ ಸುಧಾರಿಸಿ"],
            "prevention": ["ರೋಗ ನಿರೋಧಕ ತಳಿಗಳನ್ನು ಬಳಸಿ", "ಬೆಳೆ ಪರಿವರ್ತನೆ ಮಾಡಿ", "ಎಲೆಗಳ ಮೇಲೆ ನೀರು ಸಿಂಪಡಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ"],
            "fertilizer_recommendation": "ರೋಗ ನಿರೋಧಕತೆಗಾಗಿ ಸಮತೋಲನ NPK (10-10-10) ಮತ್ತು ಸಾಕಷ್ಟು ಪೊಟ್ಯಾಸಿಯಂ.",
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

@app.post("/auth/send-otp")
def send_otp(payload: LoginRequest):
    if not payload.phone or len(payload.phone) < 10:
        raise HTTPException(status_code=400, detail="Valid phone number required")
    
    otp = str(random.randint(100000, 999999))
    OTP_STORE[payload.phone] = {
        "otp": otp,
        "expires": time.time() + 300 # 5 minutes
    }
    
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        try:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            client.messages.create(
                body=f"Your Agrivision Edge login code is: {otp}",
                from_=TWILIO_PHONE_NUMBER,
                to=f"+91{payload.phone}"
            )
        except Exception as e:
            print("Twilio error:", e)
            raise HTTPException(status_code=500, detail="Failed to send SMS")
    else:
        # Mock mode if keys aren't set
        print(f"\n[MOCK SMS] To: {payload.phone} | OTP: {otp}\n")
        
    return {"status": "ok", "message": "OTP sent successfully"}

@app.post("/auth/verify-otp")
def verify_otp(payload: VerifyOtpRequest):
    record = OTP_STORE.get(payload.phone)
    if not record:
        raise HTTPException(status_code=400, detail="No OTP requested for this number")
        
    if time.time() > record["expires"]:
        del OTP_STORE[payload.phone]
        raise HTTPException(status_code=400, detail="OTP expired")
        
    if record["otp"] != payload.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
        
    # Success! Create/Update user in SQL DB
    del OTP_STORE[payload.phone]
    user = get_or_create_user(payload.phone)
    return {"status": "ok", "user": user}

@app.post("/login")
def login(payload: LoginRequest) -> Dict[str, object]:
    if not payload.phone.strip():
        raise HTTPException(status_code=400, detail="Phone number is required")
        
    user = get_or_create_user(payload.phone)
    return {
        "status": "ok",
        "message": "User verified and recorded in database",
        "user": user
    }

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
    # 1. Validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    payload = await file.read()
    image = Image.open(io.BytesIO(payload)).convert("RGB")

    # 2. Local Inference
    input_tensor = TRANSFORM(image).unsqueeze(0) 
    
    with torch.inference_mode():
        input_tensor = input_tensor.to(DEVICE)
        logits = MODEL(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, class_idx = torch.max(probs, 1)

    # 3. Map result to your folder structure
    full_class_name = CLASS_NAMES[class_idx.item()]
    
    # Split "Crop_Disease" string for the UI
    parts = full_class_name.split("_", 1)
    crop = parts[0]
    raw_disease = parts[1] if len(parts) > 1 else "Healthy"
    confidence = conf.item()

    # --- THE FIX: INTERCEPT HEALTHY PLANTS ---
    if "healthy" in raw_disease.lower():
        disease = "Healthy"      # Cleans up weird names like "healthy_foot"
        severity = "Low"         # Keeps the UI severity vocabulary consistent
    else:
        # If it's an actual disease, proceed as normal
        disease = raw_disease
        severity = _severity(confidence)

    # 4. Build Final Response (No remedy included)
    response = {
        "disease":        disease,
        "confidence":     round(confidence, 4),
        "severity":       severity,
        "crop":           crop,
        "model":          "EfficientNet-V2-S (Local)",
    }

    # 5. Update History
    history_entry = {k: v for k, v in response.items()}
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
    """Return remedy for a given disease.
    
    Priority:
    1. Gemini API (Dynamic) - if available
    2. Static _REMEDY dictionary (Fast)
    3. Local Fine-tuned model classification (Structural Fallback)
    """
    # --- 1. Attempt Gemini Remedy ---
    if GEMINI_API_KEY:
        try:
            from google import genai
            from google.genai import types

            print(f"[Gemini] Attempting remedy generation for crop='{crop}' disease='{disease}' with model={GEMINI_MODEL}")
            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"Provide a comprehensive remedy guide for the crop '{crop}' and disease '{disease}'. Return a JSON object with top-level keys 'english' and 'kannada' containing cause, symptoms, treatment_steps, prevention and fertilizer_recommendation."

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
            )

            # Attempt to extract text payload from the genai response in a robust way
            resp_text = None
            if hasattr(response, "text") and isinstance(response.text, str) and response.text:
                resp_text = response.text
            else:
                try:
                    # Some client versions expose candidates/content
                    resp_dict = response.to_dict() if hasattr(response, "to_dict") else None
                    if resp_dict:
                        resp_text = json.dumps(resp_dict)
                    else:
                        resp_text = str(response)
                except Exception:
                    resp_text = str(response)

            try:
                parsed = _safe_json_loads(resp_text)
                print(f"[Gemini] Parsed JSON remedy successfully for {crop}_{disease}")
                return parsed
            except Exception as parse_exc:
                print(f"[Gemini] Failed to parse Gemini response as JSON: {parse_exc}. Response was: {resp_text}")
                # Fall through to local dictionary
        except Exception as e:
            print(f"Gemini remedy API error: {e}. Moving to local dictionary.")
    
    # --- 2. Static Dictionary Lookup (Fuzzy & Exact) ---
    disease_lower = disease.lower()
    crop_lower = crop.lower()
    
    # Try exact compound key first
    if crop:
        exact_key = f"{crop}_{disease}"
        if exact_key in _REMEDY:
            return build_recommendation(exact_key)

    # Fuzzy search
    for key in _REMEDY:
        if disease_lower in key.lower() and (not crop_lower or crop_lower in key.lower()):
            return build_recommendation(key)

    # --- 3. SYSTEM FALLBACK: Use Local Trained Model ---
    # If we get here, the input 'disease' string didn't match our dictionary.
    # We use the fine-tuned model (if loaded) to find the nearest valid class.
    if USING_FINETUNED:
        try:
            # Since we don't have a new image here, we look for the 'disease' 
            # string inside our local CLASS_NAMES list to find the closest match.
            for class_name in CLASS_NAMES:
                if disease_lower in class_name.lower():
                    print(f"[Fallback] Found match in local model classes: {class_name}")
                    return build_recommendation(class_name)
        except Exception as exc:
            print(f"Local model fallback failed: {exc}")

    # Final Generic Fallback
    return build_recommendation(disease)
 

@app.post("/tts")
def tts(payload: TtsRequest) -> Dict[str, str]:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
        
    # Truncate text for TTS to fit within API limits while keeping text on UI detailed
    text_to_speak = payload.text
    if len(text_to_speak) > 400:
        text_to_speak = text_to_speak[:397].rsplit(' ', 1)[0] + "..."
        
    audio_base64, mime_type = _generate_tts_audio(text_to_speak, payload.language)
    return {"audio_base64": audio_base64, "mime_type": mime_type}