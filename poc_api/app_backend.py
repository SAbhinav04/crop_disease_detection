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

    return "Unknown", "General_Leaf_Stress"


def severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.85:
        return "Severe"
    if confidence >= 0.6:
        return "Moderate"
    return "Mild"


def build_recommendation(disease: str, language: str = "english") -> Dict[str, object]:
    """Return lightweight PoC advice payload compatible with frontend contract."""
    is_kn = language == "kannada"
    if is_kn:
        return {
            "cause": f"{disease} ಲಕ್ಷಣಗಳು ತೇವಾಂಶ, ಹವಾಮಾನ ಬದಲಾವಣೆ ಅಥವಾ ಪೋಷಕಾಂಶ ಅಸಮತೋಲನದಿಂದ ಕಾಣಿಸಬಹುದು.",
            "symptoms": ["ಇಲೆ ಮೇಲೆ ಕಲೆಗಳು", "ಬಣ್ಣ ಬದಲಾವಣೆ", "ವೃದ್ಧಿಯಲ್ಲಿ ಕುಗ್ಗುವಿಕೆ"],
            "treatment_steps": [
                "ಪೀಡಿತ ಎಲೆಗಳನ್ನು ಬೇರ್ಪಡಿಸಿ",
                "ಸ್ಥಳೀಯ ಕೃಷಿ ಅಧಿಕಾರಿಗಳ ಸಲಹೆಯಂತೆ ಸ್ಪ್ರೇ ಬಳಸಿ",
                "ನೀರಿನ ನಿರ್ವಹಣೆ ಸಮತೋಲನದಲ್ಲಿರಿಸಿ",
            ],
            "prevention": [
                "ಕ್ಷೇತ್ರದಲ್ಲಿ ಗಾಳಿಯ ಸಂಚಾರ ಕಾಪಾಡಿ",
                "ಅತಿಯಾದ ನೀರಿನ ನಿಲ್ಲುವಿಕೆಯನ್ನು ತಪ್ಪಿಸಿ",
                "ಪರಿವರ್ತಿತ ಬೆಳೆ ಕ್ರಮ ಅನುಸರಿಸಿ",
            ],
            "fertilizer_recommendation": "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯ ಆಧಾರದ ಮೇಲೆ ಸಮತೋಲನ NPK ಮತ್ತು ಸೂಕ್ಷ್ಮಪೋಷಕಾಂಶ ನೀಡಿ.",
        }

    return {
        "cause": f"{disease} symptoms can be associated with moisture stress, pathogen pressure, or nutrient imbalance.",
        "symptoms": ["Leaf spots or lesions", "Color change", "Reduced vigor"],
        "treatment_steps": [
            "Isolate heavily affected leaves/plants",
            "Apply crop-appropriate fungicide/bactericide as per local guidance",
            "Improve irrigation and field sanitation",
        ],
        "prevention": [
            "Improve air circulation",
            "Avoid prolonged leaf wetness",
            "Follow crop rotation and clean field practices",
        ],
        "fertilizer_recommendation": "Use soil-test-based balanced NPK with micronutrients.",
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
    imagenet_label = IMAGENET_CATEGORIES[int(pred_idx.item())]
    crop, disease = map_prediction_to_domain(imagenet_label)

    response = {
        "disease": disease,
        "confidence": round(confidence_value, 4),
        "severity": severity_from_confidence(confidence_value),
        "crop": crop,
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
