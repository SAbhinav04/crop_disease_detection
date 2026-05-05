# 📖 Documentation Quick Reference

## For Different Roles

### 👨‍💻 **Frontend Developer**
Start with: **README.md** (Quick Start)
Then read: **ARCHITECTURE.md** → Frontend Components section
Reference: **ARCHITECTURE.md** → Frontend Data Flow

**Key files:**
- `frontend/src/App.jsx` - Main component
- `frontend/src/components/` - All UI components
- `frontend/src/hooks/useApi.js` - API communication
- `frontend/src/utils/api.js` - Backend endpoints

---

### 🔧 **Backend Developer**
Start with: **README.md** (Quick Start)
Then read: **ARCHITECTURE.md** → Backend Architecture section
Reference: **MODEL_TRAINING.md** for model details

**Key files:**
- `backend/app_backend.py` - FastAPI endpoints
- `backend/database.py` - SQLite operations
- `best_crop_model.pth` - Pre-trained model
- `classes.txt` - Disease class labels

---

### 🤖 **ML Engineer**
Start with: **MODEL.md** (Model usage & architecture)
Then read: **MODEL_TRAINING.md** (How to train/retrain)
Reference: **ARCHITECTURE.md** → Backend Architecture → Model Configuration

**Key files:**
- `MODEL.md` - Model specs, classes, usage examples
- `MODEL_TRAINING.md` - Training process, retraining guide
- `best_crop_model.pth` - Current production model
- `model2.py` - Training script (reference)

---

### 🧪 **QA / Testing**
Start with: **README.md** (Quick Start)
Then read: **ARCHITECTURE.md** → Common Workflows section
Reference: **README.md** → Testing section

**Test flows:**
1. Image upload & analysis
2. Results display
3. AI advice generation
4. Audio playback
5. History tracking

---

### 📊 **DevOps / Deployment**
Start with: **README.md** → Setup sections
Then read: **ARCHITECTURE.md** → Environment Variables
Reference: **MODEL_TRAINING.md** → Model Deployment

**Key configs:**
- `.env` - Environment variables
- `requirements.txt` - Python dependencies
- `frontend/package.json` - Node dependencies
- `backend/app_backend.py` → CORS configuration

---

## 📁 Documentation Files at a Glance

```
crop_disease_detetction/
│
├── README.md
│   └── Quick start (5 min setup)
│       ├── Prerequisites
│       ├── Backend setup
│       ├── Frontend setup
│       ├── Dev credentials
│       └── Common commands
│
├── QUICK_REFERENCE.md (You are here)
│   ├── Role-based guides
│   ├── File navigation
│   ├── Command reference
│   └── Troubleshooting quick links
│
├── ARCHITECTURE.md (Detailed Reference)
│   ├── Frontend Architecture
│   │   ├── Component hierarchy
│   │   ├── State management
│   │   ├── 10+ component guides
│   │   ├── Utilities (API, i18n, colors)
│   │   └── Data flow diagrams
│   │
│   ├── Backend Architecture
│   │   ├── 4 main endpoints
│   │   ├── Database schema
│   │   ├── External APIs
│   │   ├── Environment variables
│   │   └── CORS configuration
│   │
│   ├── Integration Guide
│   │   ├── Frontend ↔️ Backend flow
│   │   ├── Authentication
│   │   ├── Component dependencies
│   │   └── Common workflows
│   │
│   └── Troubleshooting & Next Steps
│
├── MODEL.md (Model Usage Guide) ⭐ NEW
│   ├── Model architecture & specs
│   ├── 15 supported classes
│   ├── How to use for inference
│   ├── Backend integration example
│   ├── Input/output format
│   ├── Performance characteristics
│   ├── Common issues & solutions
│   ├── Fallback mechanism
│   └── Advanced modifications
│
├── MODEL_TRAINING.md (ML Reference)
│   ├── Training configuration
│   ├── How to retrain
│   ├── Improving accuracy
│   ├── Performance optimization
│   ├── Validation & testing
│   ├── Common issues & solutions
│   └── Deployment
│
└── Related files
    ├── best_crop_model.pth (trained weights)
    └── classes.txt (class labels)
```

---

## 🚀 Quickest Path to Running the App

```bash
# 1. Backend (Terminal 1)
cd crop_disease_detetction
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app_backend:app --reload --port 8000
# ✅ Backend ready at http://localhost:8000

# 2. Frontend (Terminal 2)
cd frontend
npm install
npm run dev
# ✅ Frontend ready at http://localhost:5173

# 3. Login with dev credentials
Phone: 1234567890
OTP: 123456
```

**Time required:** ~5 minutes

---

## 🔑 Key Concepts to Understand

### 1. **Authentication Flow**
```
User → Login page → Phone verification → OTP check → App access
Documentation: ARCHITECTURE.md → Authentication Flow
```

### 2. **Image Analysis Pipeline**
```
Upload image → /predict endpoint → Model inference (2-3s) → Results
Documentation: ARCHITECTURE.md → Endpoints → POST /predict
```

### 3. **Remedy Generation**
```
Get Advice → /remedy-llm endpoint → Gemini LLM → Treatment in 2 languages
Documentation: ARCHITECTURE.md → Endpoints → GET /remedy-llm
```

### 4. **Audio Generation**
```
Hear in Kannada → /tts endpoint → Sarvam AI → Audio blob → Play
Documentation: ARCHITECTURE.md → Endpoints → POST /tts
```

### 5. **History Tracking**
```
View history → /history endpoint → SQLite query → Last 5 predictions
Documentation: ARCHITECTURE.md → Backend Architecture → Database Schema
```

---

## 🛠️ Common Tasks

### "I need to add a new component"
1. Create `frontend/src/components/NewComponent.jsx`
2. Import in `App.jsx`
3. Pass props from App state
4. Reference: ARCHITECTURE.md → Frontend Components

### "I need to add a new API endpoint"
1. Add route to `backend/app_backend.py`
2. Add function in `frontend/src/utils/api.js`
3. Export from `frontend/src/hooks/useApi.js`
4. Use `useApi()` hook in component
5. Reference: ARCHITECTURE.md → Backend Integration

### "I need to improve model accuracy"
1. See MODEL_TRAINING.md → Improving Model Performance
2. Options: more data, more epochs, better augmentation, different architecture
3. Run retraining: `python model2.py`
4. Deploy new model: Move `.pth` file to backend

### "I need to debug model inference"
1. Check MODEL.md → Common Issues & Solutions
2. Verify image preprocessing matches normalization
3. Test with known good image
4. Check model weights loaded correctly

### "I need to add a new language"
1. Add translations to `frontend/src/utils/i18n.js`
2. Update TTS if needed (check Sarvam AI language support)
3. Reference: ARCHITECTURE.md → Frontend Utilities → i18n.js

### "I need to debug a frontend issue"
1. Check browser console (F12)
2. Check network tab for API calls
3. Verify backend is running: `curl http://localhost:8000/docs`
4. See README.md → Troubleshooting → Frontend Issues

### "I need to debug a backend issue"
1. Check terminal where uvicorn is running
2. Verify model file exists
3. Check .env file for API keys
4. See README.md → Troubleshooting → Backend Issues

---

## 📞 File Location Reference

| Need | File/Folder | Location | Doc Reference |
|------|-------------|----------|---|
| UI Components | Frontend Components | `frontend/src/components/` | ARCHITECTURE.md |
| API Client | api.js | `frontend/src/utils/api.js` | ARCHITECTURE.md |
| Custom Hooks | useApi.js | `frontend/src/hooks/useApi.js` | ARCHITECTURE.md |
| Translations | i18n.js | `frontend/src/utils/i18n.js` | ARCHITECTURE.md |
| Colors/Severity | colors.js | `frontend/src/utils/colors.js` | ARCHITECTURE.md |
| Main App | App.jsx | `frontend/src/App.jsx` | ARCHITECTURE.md |
| Backend Routes | app_backend.py | `backend/app_backend.py` | ARCHITECTURE.md |
| Database | database.py | `backend/database.py` | ARCHITECTURE.md |
| Model Specs | MODEL.md | `MODEL.md` | This doc! |
| Model Training | MODEL_TRAINING.md | `MODEL_TRAINING.md` | This doc! |
| **Model File** | **best_crop_model.pth** | **`best_crop_model.pth`** | **MODEL.md** |
| **Classes** | **classes.txt** | **`classes.txt`** | **MODEL.md** |
| Training Script | model2.py | `crop_disease_model_retrained/model2.py` | MODEL_TRAINING.md |
| Dependencies | requirements.txt | `requirements.txt` | README.md |
| Env Config | .env | `.env` (create this) | ARCHITECTURE.md |

---

## 🎯 Documentation Navigation Matrix

|  | **Quick Start** | **Architecture** | **Model Usage** | **Model Training** | **Reference** |
|---|---|---|---|---|---|
| **Setup** | ✅ README | ARCHITECTURE | - | - | QUICK_REF |
| **Frontend Dev** | README | ✅ ARCHITECTURE | - | - | QUICK_REF |
| **Backend Dev** | README | ✅ ARCHITECTURE | MODEL | - | QUICK_REF |
| **ML Engineer** | - | ARCHITECTURE | ✅ MODEL | ✅ MODEL_TRAINING | QUICK_REF |
| **QA Testing** | ✅ README | ARCHITECTURE | - | - | QUICK_REF |
| **DevOps** | ✅ README | ARCHITECTURE | - | MODEL_TRAINING | QUICK_REF |

---

## 💡 Pro Tips

1. **Always check `.env` first** - Missing env vars cause 90% of runtime errors
2. **Backend logs are your friend** - Read uvicorn output carefully
3. **Browser DevTools is essential** - Use Network tab to see API calls
4. **Clear cache before debugging** - Old data causes weird bugs
5. **Test with dev credentials first** - Phone: 1234567890, OTP: 123456
6. **Model loading takes time** - Don't worry if first inference takes 5+ seconds
7. **GPU not required** - App works fine on CPU (just slower)

---

## 🔗 Quick Links

**Setup & Run:**
- Quick Start: [README.md](README.md#-quick-start-5-minutes)
- Backend Command: `uvicorn backend.app_backend:app --reload --port 8000`
- Frontend Command: `npm run dev`

**Architecture Deep Dive:**
- [Full Architecture](ARCHITECTURE.md)
- [Frontend Components](ARCHITECTURE.md#frontend-architecture)
- [Backend Endpoints](ARCHITECTURE.md#key-endpoints)
- [Data Flow](ARCHITECTURE.md#-frontend--backend-integration)

**Model & AI:**
- [Model Usage Guide](MODEL.md)
- [Model Specifications](MODEL.md#model-specifications)
- [How to Use the Model](MODEL.md#how-to-use-the-model)
- [Training Guide](MODEL_TRAINING.md)
- [How to Retrain](MODEL_TRAINING.md#-how-to-retrain-the-model)
- [Performance Optimization](MODEL_TRAINING.md#-improving-model-performance)

---

## 📊 System Overview (1-minute read)

```
┌─────────────────────┐
│   User Browser      │
│  (React + Vite)     │
└──────────┬──────────┘
           │ HTTP
           ↓
┌─────────────────────┐
│  FastAPI Backend    │
│  (Python)           │
└──────────┬──────────┘
           │
    ┌──────┼──────┬──────────┐
    ↓      ↓      ↓          ↓
  Model  SQLite Gemini   Sarvam AI
  (PyTorch) (DB) (LLM)   (TTS)
```

**Flow:**
1. User uploads image → Frontend sends to backend
2. Backend runs model inference → Returns disease prediction
3. User clicks "Get Advice" → Backend calls Gemini LLM → Returns remedy
4. User clicks "Hear" → Backend calls Sarvam AI → Returns audio
5. Predictions saved in SQLite → Can view history

---

## ✅ Verification Checklist

- [ ] Backend running: `curl http://localhost:8000/docs`
- [ ] Frontend running: Visit `http://localhost:5173`
- [ ] Model loaded: Check backend logs for model loading message
- [ ] Login works: Use phone `1234567890`, OTP `123456`
- [ ] Upload works: Try uploading a test image
- [ ] API calls work: Check Network tab in browser DevTools
- [ ] All env vars set: Check `.env` file has all required keys

---

## 🆘 Need Help?

1. **Setup issues?** → See README.md → Troubleshooting
2. **Component questions?** → See ARCHITECTURE.md → Component names
3. **Model questions?** → See MODEL_TRAINING.md
4. **API issues?** → See ARCHITECTURE.md → Backend Integration
5. **Can't find something?** → Search this Quick Reference

---

**Last Updated:** May 4, 2026  
**For Questions:** Refer to appropriate documentation file above
