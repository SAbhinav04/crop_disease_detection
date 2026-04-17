#!/usr/bin/env python3
"""FastAPI PoC backend using pretrained ResNet50 for image inference."""

from __future__ import annotations

import base64
import io
import struct
import wave
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
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
PREDICTION_HISTORY: List[Dict[str, object]] = []
MAX_HISTORY = 100


class TtsRequest(BaseModel):
    text: str
    language: str = "kn"


def infer_crop_disease_from_label(label: str) -> Tuple[str, str]:
    lower = label.lower()

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
        if "spot" in lower or "blight" in lower:
            return "Pepper", "Pepper_Bacterial_spot"
        return "Pepper", "Pepper_Healthy"

    if "tomato" in lower:
        if "blight" in lower:
            return "Tomato", "Tomato_Early_blight"
        if "mold" in lower:
            return "Tomato", "Tomato_Leaf_Mold"
        if "spot" in lower:
            return "Tomato", "Tomato_Septoria_leaf_spot"
        return "Tomato", "Tomato_Healthy"

    if "potato" in lower:
        if "blight" in lower:
            return "Potato", "Potato_Early_blight"
        return "Potato", "Potato_Healthy"

    if "corn" in lower or "maize" in lower:
        if "rust" in lower:
            return "Corn", "Corn_Common_rust"
        if "blight" in lower:
            return "Corn", "Corn_Northern_Leaf_Blight"
        if "spot" in lower:
            return "Corn", "Corn_Cercospora_leaf_spot"
        return "Corn", "Corn_Healthy"

    if "leaf" in lower:
        return "Unknown", "General_Leaf_Stress"

    return "Unknown", f"Unknown_{label.replace(' ', '_')}"


def map_prediction_to_domain(label: str) -> Tuple[str, str]:
    return infer_crop_disease_from_label(label)


def pick_best_prediction(probs: torch.Tensor) -> Tuple[str, float, List[Dict[str, object]]]:
    """Inspect top candidates and prefer the most specific crop/disease mapping."""
    top_k = min(5, probs.shape[1])
    confidences, indices = torch.topk(probs, k=top_k, dim=1)

    candidates: List[Dict[str, object]] = []
    best_candidate: Dict[str, object] | None = None

    for confidence_tensor, pred_idx_tensor in zip(confidences[0], indices[0]):
        confidence_value = float(confidence_tensor.item())
        imagenet_label = IMAGENET_CATEGORIES[int(pred_idx_tensor.item())]
        crop, disease = infer_crop_disease_from_label(imagenet_label)
        candidate = {
            "imagenet_label": imagenet_label,
            "confidence": round(confidence_value, 4),
            "crop": crop,
            "disease": disease,
        }
        candidates.append(candidate)

        if best_candidate is None:
            best_candidate = candidate
            continue

        current_specificity = 0 if candidate["crop"] == "Unknown" else 1
        best_specificity = 0 if best_candidate["crop"] == "Unknown" else 1

        if current_specificity > best_specificity:
            best_candidate = candidate
        elif current_specificity == best_specificity and candidate["confidence"] > best_candidate["confidence"]:
            best_candidate = candidate

    assert best_candidate is not None
    return best_candidate["crop"], best_candidate["confidence"], candidates


def severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.85:
        return "Severe"
    if confidence >= 0.6:
        return "Moderate"
    return "Mild"


def build_recommendation(disease: str, language: str = "english") -> Dict[str, object]:
    """Return lightweight PoC advice payload compatible with frontend contract."""
    is_kn = language == "kannada"

    disease_lower = disease.lower()
    crop_name = disease.split("_")[0] if "_" in disease else disease
    if "apple_scab" in disease_lower:
        english_cause = "Apple scab is usually linked to wet, cool conditions and leaf surface infection."
        english_symptoms = ["Olive-green or dark spots on leaves", "Leaf drop", "Reduced fruit quality"]
        english_treatment = ["Remove infected leaves", "Apply label-approved fungicide", "Improve canopy ventilation"]
        english_prevention = ["Prune for airflow", "Avoid overhead irrigation", "Use resistant varieties where possible"]
        english_fertilizer = "Use balanced nutrition and avoid excess nitrogen that drives soft growth."

        kannada_cause = "ಆಪಲ್ ಸ್ಕ್ಯಾಬ್ ಸಾಮಾನ್ಯವಾಗಿ ತೇವಯುಕ್ತ, ಚಳಿ ವಾತಾವರಣದಲ್ಲಿ ಎಲೆಗಳ ಮೇಲೆ ಸೋಂಕಿನಿಂದ ಉಂಟಾಗುತ್ತದೆ."
        kannada_symptoms = ["ಎಲೆಗಳ ಮೇಲೆ ಕಪ್ಪು/ಹಸಿರು ಕಲೆಗಳು", "ಎಲೆ ಉದುರುವುದು", "ಹಣ್ಣಿನ ಗುಣಮಟ್ಟ ಕುಸಿಯುವುದು"]
        kannada_treatment = ["ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ", "ಲೇಬಲ್ ಅನುಮೋದಿತ ಸ್ಪ್ರೇ ಬಳಸಿ", "ಮರದೊಳಗಿನ ಗಾಳಿಯ ಹರಿವನ್ನು ಹೆಚ್ಚಿಸಿ"]
        kannada_prevention = ["ಕತ್ತರಿಸುವ ಮೂಲಕ ಗಾಳಿಯ ಚಲನೆ ಹೆಚ್ಚಿಸಿ", "ತಲೆಯ ಮೇಲಿನಿಂದ ನೀರಿನ ತೇವ ತಪ್ಪಿಸಿ", "ಪ್ರತಿರೋಧಿ ತಳಿಗಳನ್ನು ಬಳಸಿ"]
        kannada_fertilizer = "ಸಮತೋಲನ ಪೋಷಕಾಂಶ ನೀಡಿ; ಹೆಚ್ಚುವರಿ ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ."

    elif "black_rot" in disease_lower:
        english_cause = "Black rot is commonly associated with humid conditions and fungal spread on plant tissue."
        english_symptoms = ["Brown or black lesions", "Leaf curling", "Fruit or stem decay"]
        english_treatment = ["Remove infected material", "Use recommended fungicide", "Sanitize tools and field debris"]
        english_prevention = ["Ensure sunlight and airflow", "Avoid prolonged wet leaves", "Use clean planting material"]
        english_fertilizer = "Apply balanced fertilizer based on soil test to maintain plant vigor."

        kannada_cause = "ಕಪ್ಪು ಕೊಳೆ (Black rot) ತೇವಯುಕ್ತ ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ ಶಿಲೀಂಧ್ರದಿಂದ ವೇಗವಾಗಿ ಹರಡಬಹುದು."
        kannada_symptoms = ["ಕಂದು/ಕಪ್ಪು ಗಾಯಗಳು", "ಎಲೆ ಮಡಚಿಕೊಳ್ಳುವುದು", "ಹಣ್ಣು ಅಥವಾ ಕಾಂಡ ಕೊಳೆಯುವುದು"]
        kannada_treatment = ["ಸೋಂಕಿತ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ", "ಶಿಫಾರಸು ಮಾಡಿದ ಫಂಗಿಸೈಡ್ ಬಳಸಿ", "ಉಪಕರಣಗಳು ಮತ್ತು ಕಸದ ನಿರ್ವಹಣೆ ಸ್ವಚ್ಛವಾಗಿಡಿ"]
        kannada_prevention = ["ಸೂರ್ಯಪ್ರಕಾಶ ಮತ್ತು ಗಾಳಿಯ ಹರಿವು ಕಾಪಾಡಿ", "ಎಲೆ ಹೆಚ್ಚು ಸಮಯ ತೇವವಾಗದಂತೆ ನೋಡಿ", "ಸ್ವಚ್ಛ ಬೀಜ/ನಾಟಿ ಬಳಸಿ"]
        kannada_fertilizer = "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಆಧಾರದ ಮೇಲೆ ಸಮತೋಲನ ರಸಗೊಬ್ಬರ ಬಳಸಿ."

    elif "early_blight" in disease_lower or "late_blight" in disease_lower:
        english_cause = f"{disease} often appears under humid conditions and spread is accelerated by wet foliage."
        english_symptoms = ["Target-like leaf spots", "Yellowing around lesions", "Premature leaf drop"]
        english_treatment = ["Remove affected leaves", "Use fungicide if locally recommended", "Reduce leaf wetness"]
        english_prevention = ["Rotate crops", "Water at the base", "Keep field residues under control"]
        english_fertilizer = "Maintain balanced NPK and avoid excess nitrogen during disease pressure."

        kannada_cause = f"{disease} ಸಾಮಾನ್ಯವಾಗಿ ತೇವಯುಕ್ತ ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ ಕಾಣಿಸಿಕೊಳ್ಳುತ್ತದೆ ಮತ್ತು ಒದ್ದೆಯಾದ ಎಲೆಗಳಿಂದ ವೇಗವಾಗಿ ಹರಡುತ್ತದೆ."
        kannada_symptoms = ["ಲಕ್ಷ್ಯ-ಮಾದರಿ ಕಲೆಗಳು", "ಗಾಯಗಳ ಸುತ್ತ ಹಳದಿ ಬಣ್ಣ", "ಎಲೆ ಮುಂಚಿತವಾಗಿ ಉದುರುವುದು"]
        kannada_treatment = ["ಪೀಡಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ", "ಸ್ಥಳೀಯ ಸಲಹೆಯಂತೆ ಫಂಗಿಸೈಡ್ ಬಳಸಿ", "ಎಲೆ ತೇವ ಕಡಿಮೆ ಮಾಡಿ"]
        kannada_prevention = ["ಬೆಳೆ ಪರಿವರ್ತನೆ ಅನುಸರಿಸಿ", "ಮೂಲಭಾಗಕ್ಕೆ ನೀರು ನೀಡಿ", "ಕ್ಷೇತ್ರದ ಅವಶೇಷ ನಿಯಂತ್ರಿಸಿ"]
        kannada_fertilizer = "ರೋಗ ಒತ್ತಡದ ಸಮಯದಲ್ಲಿ ಸಮತೋಲನ NPK ಬಳಸಿ; ಹೆಚ್ಚುವರಿ ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ."

    elif "bacterial_spot" in disease_lower:
        english_cause = f"{disease} usually spreads through moisture, splash, and contaminated plant surfaces."
        english_symptoms = ["Small dark lesions", "Leaf spotting", "Yellow halo around spots"]
        english_treatment = ["Remove heavily infected leaves", "Use copper-based sprays if recommended", "Avoid overhead irrigation"]
        english_prevention = ["Use clean seed/seedlings", "Improve spacing", "Keep tools sanitized"]
        english_fertilizer = "Avoid excess nitrogen; keep nutrition balanced to reduce soft growth."

        kannada_cause = f"{disease} ತೇವಾಂಶ, ನೀರಿನ ಚಿಮ್ಮು ಮತ್ತು ಸೋಂಕಿತ ಮೇಲ್ಮೈಗಳಿಂದ ಹರಡಬಹುದು."
        kannada_symptoms = ["ಸಣ್ಣ ಕಪ್ಪು ಕಲೆಗಳು", "ಎಲೆಗಳ ಮೇಲೆ ದಾಗಗಳು", "ಕಲೆಗಳ ಸುತ್ತ ಹಳದಿ ವಲಯ"]
        kannada_treatment = ["ತೀವ್ರವಾಗಿ ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ", "ಶಿಫಾರಸು ಇದ್ದರೆ ಕಾಪರ್ ಸ್ಪ್ರೇ ಬಳಸಿ", "ಮೇಲಿನಿಂದ ನೀರಾವರಿ ತಪ್ಪಿಸಿ"]
        kannada_prevention = ["ಸ್ವಚ್ಛ ಬೀಜ/ಸಸಿ ಬಳಸಿ", "ಸಸ್ಯಗಳ ನಡುವಿನ ಅಂತರ ಹೆಚ್ಚಿಸಿ", "ಉಪಕರಣ ಸ್ವಚ್ಛತೆ ಕಾಪಾಡಿ"]
        kannada_fertilizer = "ಹೆಚ್ಚುವರಿ ನೈಟ್ರೋಜನ್ ತಪ್ಪಿಸಿ; ಸಮತೋಲನ ಪೋಷಕಾಂಶ ಕಾಪಾಡಿ."

    elif "healthy" in disease_lower:
        english_cause = f"{crop_name} appears healthy with no strong disease symptom pattern detected."
        english_symptoms = ["Vigorous growth", "Uniform leaf color", "No visible lesions"]
        english_treatment = ["Continue monitoring", "Maintain irrigation discipline", "Keep field sanitation"]
        english_prevention = ["Regular scouting", "Balanced fertilization", "Crop rotation"]
        english_fertilizer = "Maintain soil-test-based balanced nutrition."

        kannada_cause = f"{crop_name} ಗೆ ಸ್ಪಷ್ಟ ರೋಗ ಲಕ್ಷಣಗಳು ಕಾಣುತ್ತಿಲ್ಲ; ಸಸ್ಯವು ಆರೋಗ್ಯಕರವಾಗಿ ಕಾಣುತ್ತದೆ."
        kannada_symptoms = ["ಬಲವಾದ ಬೆಳವಣಿಗೆ", "ಎಲೆಗಳ ಸಮಾನ ಬಣ್ಣ", "ಗಾಯಗಳು ಕಾಣಿಸಿಲ್ಲ"]
        kannada_treatment = ["ನಿರಂತರ ವೀಕ್ಷಣೆ ಮಾಡಿ", "ನೀರಿನ ನಿರ್ವಹಣೆ ಕಾಪಾಡಿ", "ಕ್ಷೇತ್ರ ಸ್ವಚ್ಛತೆ ಇಟ್ಟುಕೊಳ್ಳಿ"]
        kannada_prevention = ["ನಿಯಮಿತ ಪರಿಶೀಲನೆ", "ಸಮತೋಲನ ಗೊಬ್ಬರ", "ಬೆಳೆ ಪರಿವರ್ತನೆ"]
        kannada_fertilizer = "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಆಧಾರದ ಮೇಲೆ ಸಮತೋಲನ ಪೋಷಕಾಂಶ ನೀಡಿ."

    else:
        english_cause = f"{disease} symptoms can be associated with moisture stress, pathogen pressure, or nutrient imbalance."
        english_symptoms = ["Leaf spots or lesions", "Color change", "Reduced vigor"]
        english_treatment = [
            "Isolate heavily affected leaves/plants",
            "Apply crop-appropriate fungicide/bactericide as per local guidance",
            "Improve irrigation and field sanitation",
        ]
        english_prevention = [
            "Improve air circulation",
            "Avoid prolonged leaf wetness",
            "Follow crop rotation and clean field practices",
        ]
        english_fertilizer = "Use soil-test-based balanced NPK with micronutrients."

        kannada_cause = f"{disease} ಲಕ್ಷಣಗಳು ತೇವಾಂಶ, ಹವಾಮಾನ ಬದಲಾವಣೆ ಅಥವಾ ಪೋಷಕಾಂಶ ಅಸಮತೋಲನದಿಂದ ಕಾಣಿಸಬಹುದು."
        kannada_symptoms = ["ಇಲೆ ಮೇಲೆ ಕಲೆಗಳು", "ಬಣ್ಣ ಬದಲಾವಣೆ", "ವೃದ್ಧಿಯಲ್ಲಿ ಕುಗ್ಗುವಿಕೆ"]
        kannada_treatment = [
            "ಪೀಡಿತ ಎಲೆಗಳನ್ನು ಬೇರ್ಪಡಿಸಿ",
            "ಸ್ಥಳೀಯ ಕೃಷಿ ಅಧಿಕಾರಿಗಳ ಸಲಹೆಯಂತೆ ಸ್ಪ್ರೇ ಬಳಸಿ",
            "ನೀರಿನ ನಿರ್ವಹಣೆ ಸಮತೋಲನದಲ್ಲಿರಿಸಿ",
        ]
        kannada_prevention = [
            "ಕ್ಷೇತ್ರದಲ್ಲಿ ಗಾಳಿಯ ಸಂಚಾರ ಕಾಪಾಡಿ",
            "ಅತಿಯಾದ ನೀರಿನ ನಿಲ್ಲುವಿಕೆಯನ್ನು ತಪ್ಪಿಸಿ",
            "ಪರಿವರ್ತಿತ ಬೆಳೆ ಕ್ರಮ ಅನುಸರಿಸಿ",
        ]
        kannada_fertilizer = "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಆಧಾರದ ಮೇಲೆ ಸಮತೋಲನ NPK ಮತ್ತು ಸೂಕ್ಷ್ಮಪೋಷಕಾಂಶ ನೀಡಿ."

    if is_kn:
        return {
            "cause": kannada_cause,
            "symptoms": kannada_symptoms,
            "treatment_steps": kannada_treatment,
            "prevention": kannada_prevention,
            "fertilizer_recommendation": kannada_fertilizer,
        }

    return {
        "cause": english_cause,
        "symptoms": english_symptoms,
        "treatment_steps": english_treatment,
        "prevention": english_prevention,
        "fertilizer_recommendation": english_fertilizer,
    }


def synthesize_silence_wav_base64(duration_seconds: float = 1.2, sample_rate: int = 16000) -> str:
    """Generate a short silent WAV payload as base64 for PoC TTS compatibility."""
    num_frames = int(duration_seconds * sample_rate)
    audio_buffer = io.BytesIO()

    with wave.open(audio_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        silence_frame = struct.pack("<h", 0)
        wav_file.writeframes(silence_frame * num_frames)

    return base64.b64encode(audio_buffer.getvalue()).decode("ascii")


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
    crop, disease, candidates = pick_best_prediction(probs)

    if disease == "General_Leaf_Stress":
        # Use the raw top label to avoid repeating the same generic fallback for every leaf-like image.
        imagenet_label = IMAGENET_CATEGORIES[int(pred_idx.item())]
        if "leaf" not in imagenet_label.lower():
            _, disease = map_prediction_to_domain(imagenet_label)

    response = {
        "disease": disease,
        "confidence": round(confidence_value, 4),
        "severity": severity_from_confidence(confidence_value),
        "crop": crop,
        "top_candidates": candidates,
    }

    PREDICTION_HISTORY.append(
        {
            **response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    if len(PREDICTION_HISTORY) > MAX_HISTORY:
        del PREDICTION_HISTORY[:-MAX_HISTORY]

    return response


@app.get("/history")
def history(limit: int = Query(default=10, ge=1, le=100)) -> List[Dict[str, object]]:
    return PREDICTION_HISTORY[-limit:]


@app.get("/remedy-llm")
def remedy_llm(disease: str = Query(..., min_length=2)) -> Dict[str, object]:
    return {
        "english": build_recommendation(disease, "english"),
        "kannada": build_recommendation(disease, "kannada"),
    }


@app.post("/tts")
def tts(payload: TtsRequest) -> Dict[str, str]:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    return {
        "audio_base64": synthesize_silence_wav_base64(),
        "mime_type": "audio/wav",
    }
