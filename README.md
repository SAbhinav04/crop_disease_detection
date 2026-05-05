# AgriVision Edge - Crop Disease Detection System

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Node](https://img.shields.io/badge/Node-16%2B-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## 📱 Quick Overview

AgriVision Edge is an AI-powered crop disease detection system for Indian farmers. Upload a leaf image, get instant disease diagnosis, and receive treatment recommendations in English or Kannada with audio support.

**Key Features:**
✅ Real-time disease detection using deep learning  
✅ Bilingual interface (English & Kannada)  
✅ AI-powered remedy generation  
✅ Text-to-speech support  
✅ Prediction history tracking  
✅ Mobile-responsive design

---

## 🚀 Quick Start (5 minutes)

### 1. Clone & Setup Backend

```bash
cd crop_disease_detetction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file in project root:
```bash
GEMINI_API_KEY=your_gemini_api_key
SARVAM_API_KEY=your_sarvam_api_key
VITE_API_URL=http://localhost:8000
```

### 3. Run Backend Server

```bash
source .venv/bin/activate
uvicorn backend.app_backend:app --reload --host 0.0.0.0 --port 8000
```

✅ Backend ready at: http://localhost:8000  
📚 API Docs: http://localhost:8000/docs

### 4. Run Frontend (New Terminal)

```bash
cd frontend
npm install
npm run dev
```

✅ Frontend ready at: http://localhost:5173

---

## 🔑 Dev Credentials

**Login to test the app:**
```
Phone: 1234567890
OTP: 123456
```

---

## 📖 Full Documentation

For detailed architecture and component breakdown:
👉 **[See ARCHITECTURE.md](./ARCHITECTURE.md)**

For model specifications and usage:
👉 **[See MODEL.md](./MODEL.md)**

For model training and optimization:
👉 **[See MODEL_TRAINING.md](./MODEL_TRAINING.md)**

---

## 📁 Project Structure

```
crop_disease_detetction/
├── frontend/              # React + Vite app
├── backend/               # FastAPI server  
├── ml_pipeline/          # Data preprocessing
├── data/                 # Datasets
├── best_crop_model.pth   # Trained model
├── ARCHITECTURE.md       # Full architecture & components
├── MODEL.md              # Model guide & specifications
├── MODEL_TRAINING.md     # Model training & retraining
├── QUICK_REFERENCE.md    # Navigation by role
└── README.md             # This file
```

---

## 🎯 Key Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/predict` | Detect disease from image |
| GET | `/remedy-llm` | Get AI treatment advice |
| GET | `/history` | Fetch user predictions |
| POST | `/tts` | Generate audio (Kannada/English) |

---

## 💻 Component Overview

**Frontend Structure:**
```
App.jsx (Main)
├── Login           (Authentication)
├── UploadSection   (Image upload - Tab 01)
├── ResultsSection  (Results display - Tab 02)
├── AIAdviceSection (Treatment details)
├── AudioPlayer     (TTS controls)
└── DiseaseHistory  (Past predictions)
```

**Backend Structure:**
```
FastAPI
├── /predict       → EfficientNet-V2-S model
├── /remedy-llm    → Gemini LLM API
├── /history       → SQLite database
└── /tts           → Sarvam AI TTS
```

---

## 🔗 Data Flow

```
User uploads image
    ↓
Frontend sends to /predict
    ↓
Backend inference (2-3 sec)
    ↓
Returns: { disease, crop, confidence }
    ↓
Display result with severity badge
    ↓
Click "Get AI Advice"
    ↓
Query Gemini LLM for treatment
    ↓
Display remedy in English + Kannada
    ↓
Click "Hear in Kannada"
    ↓
Generate audio via Sarvam AI
    ↓
Play in browser
```

---

## 📊 Supported Crops & Diseases

- **Tomato** (8 conditions)
- **Potato** (3 conditions)
- **Pepper** (2 conditions)
- **Grape** (2+ conditions)
- **Corn/Maize** (conditions available)

Total: 15+ crop-disease combinations

---

## ⚙️ Configuration

### Environment Variables

```bash
# APIs
GEMINI_API_KEY=your_key          # LLM for remedies
SARVAM_API_KEY=your_key          # Text-to-speech
TWILIO_ACCOUNT_SID=your_sid      # SMS (optional)
TWILIO_AUTH_TOKEN=your_token     # SMS (optional)

# URLs
VITE_API_URL=http://localhost:8000
SARVAM_TTS_URL=https://api.sarvam.ai/text-to-speech/stream

# TTS Config
SARVAM_TTS_VOICE=anushka
SARVAM_TTS_MODEL=bulbul:v3
SARVAM_TTS_SAMPLE_RATE=22050

# Model Config
CONFIDENCE_MIN=0.76
CONFIDENCE_MAX=0.89
```

---

## 🧪 Testing

### Test Image Classification
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"
```

### Test Remedy Generation
```bash
curl "http://localhost:8000/remedy-llm?disease=Early%20Blight&crop=Tomato"
```

### Test TTS
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Disease detected","language":"kn"}'
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend won't start | Ensure Python 3.9+, pip packages installed |
| 404 on API calls | Verify backend running on port 8000 |
| Image upload fails | Check file format (JPEG/PNG), size < 25MB |
| TTS not working | Verify Sarvam API key in `.env` |
| Gemini errors | Check GEMINI_API_KEY is valid |
| CORS errors | Backend CORS already configured for localhost |

---

## 📱 Frontend Languages

- **English (en)** - Default
- **Kannada (ಕನ್ನಡ)** - Full support

Toggle in navbar (top-right)

---

## 🔐 Authentication

Uses phone number + OTP verification (Twilio).

**Dev Mode:**
- Phone validation still works
- OTP is hardcoded: `123456`
- Default phone: `1234567890`

---

## 📊 Model Info

- **Architecture:** EfficientNet-V2-S
- **Fallback:** ResNet50 (ImageNet)
- **Input Size:** 224×224 px
- **Framework:** PyTorch
- **File:** `best_crop_model.pth`

---

## 🎨 UI Technologies

- **Framework:** React 18 + Vite
- **Styling:** Tailwind CSS
- **State:** React Hooks
- **Server:** Vite Dev Server

---

## 🛠️ Common Commands

```bash
# Backend
source .venv/bin/activate
uvicorn backend.app_backend:app --reload --port 8000

# Frontend
npm run dev          # Dev server
npm run build        # Production build
npm run preview      # Preview build
npm run lint        # Check code

# Database
# Reset history: Delete users.db (will recreate on next run)
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Quick start & overview (you are here) |
| `ARCHITECTURE.md` | Full technical architecture & component guide |
| `MODEL.md` | Model specifications, usage, and inference guide |
| `MODEL_TRAINING.md` | Model training guide, retraining, & optimization |
| `QUICK_REFERENCE.md` | Navigation guide for different team roles |
| `docs/` | Additional documentation |

👉 **Read ARCHITECTURE.md for component details and data flow!**  
👉 **Read MODEL.md for model specs and how to use it!**  
👉 **Read MODEL_TRAINING.md for model training and optimization!**

---

## 🤝 Contributing

1. Create a feature branch
2. Make changes
3. Test locally
4. Submit pull request

---

## 📄 License

MIT License - See LICENSE file

---

## 💬 Questions?

Refer to `ARCHITECTURE.md` for:
- Component breakdown
- API endpoint details
- Data flow diagrams
- Troubleshooting guide
- Environment setup

---

**Version:** 1.0  
**Last Updated:** May 4, 2026  
**Team:** AgriVision Edge Development
