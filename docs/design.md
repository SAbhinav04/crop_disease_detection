# Crop Disease Detection System — Design Document

**Last Updated:** 29 April 2026  
**Version:** 2.0.0

---

## 1. System Overview

**Agrivision Edge** is a multilingual (English/Kannada) AI-powered crop disease detection platform for Indian farmers. The system enables farmers to upload leaf images, get instant disease predictions, receive treatment advice in their preferred language, and access prediction history with confidence trends.

### Key Capabilities

- 📸 Leaf image upload with instant AI prediction
- 🌾 Support for 15 crop disease classes (15 classes, ~98.79% accuracy)
- 🎤 Kannada text-to-speech remedies (via Sarvam AI)
- 📊 Prediction history with confidence trend visualization
- 🔐 Phone-based OTP authentication
- 📱 Fully responsive, mobile-first design

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Browser                          │
│               (React 18 + Vite Frontend)                │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/CORS
                       ↓
┌─────────────────────────────────────────────────────────┐
│                 FastAPI Backend (Python)                 │
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │  ResNet50 Model  │  │  Remedy Database │            │
│  │  (Fine-tuned)    │  │  (JSON Lookup)   │            │
│  └──────────────────┘  └──────────────────┘            │
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │  Twilio OTP API  │  │  Sarvam TTS API  │            │
│  └──────────────────┘  └──────────────────┘            │
└─────────────────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   ┌─────────┐   ┌─────────┐   ┌──────────┐
   │ PyTorch │   │ SQLite  │   │  External│
   │ Models  │   │ Database│   │  APIs    │
   └─────────┘   └─────────┘   └──────────┘
```

---

## 3. Component Architecture

### Frontend (React)

```
App.jsx (Main Container)
├── Login.jsx (Authentication)
│   ├── Phone Input
│   ├── OTP Verification
│   └── Demo Bypass (1234567890 / 123456)
├── Header.jsx (Navigation & Language Toggle)
├── ResponsiveLayout.jsx (Grid Layout)
├── UploadSection.jsx
│   ├── Image Upload
│   ├── Drag & Drop
│   └── Preview
├── ResultsSection.jsx
│   ├── ResultCard (Disease, Confidence, Severity, Crop)
│   ├── Copy & Share Actions
│   └── AI Advice Trigger
├── AIAdviceSection.jsx
│   ├── English Tab
│   ├── Kannada Tab
│   └── Expandable UI
├── DiseaseHistory.jsx
│   ├── Recharts Confidence Trend
│   ├── Latest Predictions List
│   └── History Refresh
└── AudioPlayer.jsx
    └── Kannada TTS Playback
```

### Backend (FastAPI)

```
app_backend.py (Main Server)
├── /health (Status Check)
├── /predict (Image → Disease Classification)
├── /remedy-llm (Disease → Treatment Advice)
├── /tts (Text → Kannada Audio)
├── /history (Prediction History)
├── /auth/send-otp (Phone → OTP)
└── /auth/verify-otp (OTP → Session)

database.py (SQLite ORM)
├── User Management
├── OTP Validation
└── Prediction History
```

---

## 4. Data Flow

### A. Prediction Flow

```
1. User uploads leaf image
2. Frontend sends POST /predict with image file
3. Backend:
   - Loads image via PIL
   - Applies ResNet50 transforms (224x224, normalize)
   - Runs inference on GPU/CPU/MPS
   - Maps class ID to (Crop, Disease) via class_names.json
   - Calculates severity (Low/Moderate/Severe based on confidence)
   - Looks up remedy from JSON database
   - Returns: { disease, confidence, severity, crop, remedy_en, remedy_kn }
4. Frontend displays:
   - ResultCard with disease, confidence bar, severity badge
   - Expandable AI advice panel (EN/KN tabs)
   - Option to request Kannada TTS
5. Prediction added to history and SQLite DB
```

### B. Authentication Flow

```
1. User enters phone number
2. Check for demo bypass (1234567890)
   ├─ YES: Pre-fill OTP field with 123456
   └─ NO: Call Twilio API to send real OTP

3. User enters OTP code
4. Check for bypass match
   ├─ YES (1234567890 + 123456): Skip backend, redirect to main
   └─ NO: Call POST /auth/verify-otp with backend

5. On success: Set isLoggedIn = true, show main app
6. On failure: Display error, stay on login screen
```

### C. History Flow

```
1. App mounts → fetch /history
2. Backend queries SQLite: last 100 predictions
3. Frontend receives JSON array:
   [
     { timestamp, disease, confidence, crop },
     ...
   ]
4. Display in DiseaseHistory:
   - Recharts line chart (confidence over time)
   - Compact list below chart
   - Refresh button to reload
```

---

## 5. API Specification

### Authentication Endpoints

#### `POST /auth/send-otp`

**Request:**
```json
{
  "phone": "9876543210"
}
```

**Response (Success):**
```json
{
  "status": "otp_sent",
  "message": "OTP sent to your number"
}
```

**Response (Bypass Demo):**
- For phone `1234567890`: client-side skips backend call

---

#### `POST /auth/verify-otp`

**Request:**
```json
{
  "phone": "9876543210",
  "otp": "123456"
}
```

**Response (Success):**
```json
{
  "status": "verified",
  "user_id": "user_abc123",
  "message": "OTP verified successfully"
}
```

**Response (Bypass Demo):**
- For phone `1234567890` + OTP `123456`: frontend handles directly

---

### Prediction Endpoints

#### `POST /predict`

**Request:**
```
multipart/form-data
- file: <image file> (jpg, png, webp)
```

**Response:**
```json
{
  "disease": "Tomato_Early_Blight",
  "confidence": 0.9245,
  "severity": "Moderate",
  "crop": "Tomato",
  "remedy_en": {
    "cause": "Fungal infection caused by...",
    "symptoms": ["Brown spots...", "Concentric rings..."],
    "treatment_steps": ["Remove infected leaves...", "Apply fungicide..."],
    "prevention": ["Crop rotation...", "Avoid overhead watering..."],
    "fertilizer_recommendation": "NPK 10-26-26"
  },
  "remedy_kn": {
    "cause": "ಫಂಗಸ್ ಸೋಂಕು...",
    ...
  }
}
```

---

#### `GET /remedy-llm?disease=Tomato_Early_Blight`

**Response:**
```json
{
  "disease": "Tomato_Early_Blight",
  "english": { "cause": "...", "symptoms": [...], ... },
  "kannada": { "cause": "...", "symptoms": [...], ... }
}
```

---

#### `POST /tts`

**Request:**
```json
{
  "text": "ಭೇದಿ ಪುಟ್ಟದ ಹೆಚ್ಚಿನ ಸೋಂಕು ನಿಯಂತ್ರಣಕ್ಕೆ...",
  "language": "kn"
}
```

**Response:**
```json
{
  "audio_base64": "SUQzBAAAAAAAI1...",
  "mime_type": "audio/wav"
}
```

Or binary WAV file directly.

---

#### `GET /history`

**Response:**
```json
[
  {
    "id": 1,
    "timestamp": "2026-04-29T10:30:45Z",
    "phone": "9876543210",
    "disease": "Tomato_Early_Blight",
    "confidence": 0.9245,
    "crop": "Tomato"
  },
  ...
]
```

---

#### `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model": "fine-tuned",
  "device": "cuda",
  "num_classes": 15
}
```

---

## 6. Frontend Architecture

### Technology Stack

| Layer        | Technology              |
|--------------|------------------------|
| **Build**    | Vite, Node.js          |
| **Framework**| React 18               |
| **Styling**  | Tailwind CSS            |
| **Charts**   | Recharts               |
| **HTTP**     | Fetch API              |
| **i18n**     | Custom JSON (EN/KN)    |

### Key Components

#### Login.jsx
- **Purpose:** Phone-based OTP authentication
- **State:** phone, otp, step, loading, error, lang
- **Demo Bypass:** 
  - Phone: `1234567890`
  - OTP: `123456`
  - Skips all backend calls
- **Localization:** English & Kannada UI

#### App.jsx
- **Purpose:** Main app container & state management
- **State:** isLoggedIn, language, selectedFile, prediction, remedy, history, error, loading states
- **Logic:**
  - Conditional render: Login vs Main App
  - Load history on mount
  - Handle predictions, advice requests, TTS requests
  - Manage language switching

#### UploadSection.jsx
- **Features:**
  - File input (image only)
  - Drag & drop support
  - Preview URL generation
  - Validation messages
  - Loading state during prediction

#### ResultsSection.jsx
- **Features:**
  - ResultCard display (disease, confidence, severity, crop)
  - Copy to clipboard (with toast feedback)
  - WhatsApp share URL generation
  - AI advice trigger button
  - History trigger

#### AIAdviceSection.jsx
- **Features:**
  - Expandable panel
  - EN/KN tab toggle
  - Displays: cause, symptoms, treatment steps, prevention, fertilizer recommendation
  - Language switching

#### DiseaseHistory.jsx
- **Features:**
  - Recharts line chart (confidence trend)
  - Compact prediction list below chart
  - Manual refresh button
  - Error handling with retry

#### AudioPlayer.jsx
- **Features:**
  - HTML5 audio player
  - Playback controls
  - Loading state
  - Nonce-based key for re-rendering

### State Management

**App.jsx Central State:**
```js
const [isLoggedIn, setIsLoggedIn] = useState(false);
const [language, setLanguage] = useState('en');
const [selectedFile, setSelectedFile] = useState(null);
const [prediction, setPrediction] = useState(null);
const [remedy, setRemedy] = useState(null);
const [history, setHistory] = useState([]);
const [loadingPrediction, setLoadingPrediction] = useState(false);
const [loadingAdvice, setLoadingAdvice] = useState(false);
const [loadingHistory, setLoadingHistory] = useState(false);
const [loadingAudio, setLoadingAudio] = useState(false);
const [audioSrc, setAudioSrc] = useState(null);
const [error, setError] = useState(null);
```

---

## 7. Backend Architecture

### Technology Stack

| Layer       | Technology           |
|-------------|----------------------|
| **Framework**| FastAPI              |
| **Server**  | Uvicorn              |
| **ML Model**| PyTorch + ResNet50   |
| **Image**   | PIL (Pillow)         |
| **Database**| SQLite               |
| **Auth**    | Twilio SMS           |
| **TTS**     | Sarvam AI            |

### Model Architecture

**ResNet50 with Transfer Learning:**
- **Backbone:** ImageNet pretrained ResNet50
- **Input:** 224×224 RGB image
- **Normalization:** ImageNet mean/std (IMAGENET_MEAN, IMAGENET_STD)
- **Head:** Custom FC layer (1000 → 15 classes)
- **Output:** Logits → softmax → probabilities
- **Confidence:** max(probabilities)
- **Device:** Auto-select (CUDA > MPS > CPU)

**Fallback:**
- If `poc/best_model.pth` not found → use bare ImageNet ResNet50
- Keeps API alive even without fine-tuned weights
- Accuracy trade-off but prevents startup failure

### Class Mapping

**15 Classes (PlantVillage):**
```python
{
    "Pepper__bell___Bacterial_spot": ("Pepper", "Bacterial Spot"),
    "Pepper__bell___healthy": ("Pepper", "Healthy"),
    "Potato___Early_blight": ("Potato", "Early Blight"),
    "Potato___Late_blight": ("Potato", "Late Blight"),
    "Potato___healthy": ("Potato", "Healthy"),
    "Tomato_Bacterial_spot": ("Tomato", "Bacterial Spot"),
    "Tomato_Early_blight": ("Tomato", "Early Blight"),
    "Tomato_Late_blight": ("Tomato", "Late Blight"),
    "Tomato_Leaf_Mold": ("Tomato", "Leaf Mold"),
    "Tomato_Septoria_leaf_spot": ("Tomato", "Septoria Leaf Spot"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("Tomato", "Spider Mites"),
    "Tomato__Target_Spot": ("Tomato", "Target Spot"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("Tomato", "Yellow Leaf Curl Virus"),
    "Tomato__Tomato_mosaic_virus": ("Tomato", "Mosaic Virus"),
    "Tomato_healthy": ("Tomato", "Healthy"),
}
```

### Severity Classification

Based on confidence score:
- **Low/Early:** 0.0 - 0.70 confidence → `#FFD700` (amber)
- **Moderate:** 0.70 - 0.85 confidence → `#FFA500` (orange)
- **Severe:** 0.85 - 1.0 confidence → `#E74C3C` (red)

### Remedy Database

**Structure:**
```json
{
  "<Crop>_<Disease>": {
    "english": {
      "cause": "...",
      "symptoms": ["...", "..."],
      "treatment_steps": ["...", "..."],
      "prevention": ["...", "..."],
      "fertilizer_recommendation": "..."
    },
    "kannada": { ... }
  }
}
```

**Key:** Compound key avoids cross-crop ambiguity (e.g., "Tomato_Early_Blight" vs "Potato_Early_Blight")

### External Services

#### Twilio SMS (OTP)
- **Config:** TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
- **On send-otp:** Generate random 6-digit code, send via SMS
- **On verify-otp:** Validate code, create/update user

#### Sarvam AI (Kannada TTS)
- **Endpoint:** https://api.sarvam.ai/text-to-speech/stream
- **Voice:** anushka (Kannada female)
- **Model:** bulbul:v3
- **Sample Rate:** 22050 Hz
- **Output:** Base64 WAV or binary stream

---

## 8. Database Design

### SQLite Schema

#### users table
```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  phone TEXT UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_login TIMESTAMP
);
```

#### predictions table
```sql
CREATE TABLE predictions (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  disease TEXT NOT NULL,
  confidence REAL NOT NULL,
  crop TEXT NOT NULL,
  image_path TEXT,
  remedy_en TEXT,  -- JSON string
  remedy_kn TEXT,  -- JSON string
  FOREIGN KEY (user_id) REFERENCES users(id)
);
```

#### otp_sessions table
```sql
CREATE TABLE otp_sessions (
  id INTEGER PRIMARY KEY,
  phone TEXT NOT NULL,
  otp_code TEXT NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Query Patterns

**Recent history for user:**
```sql
SELECT disease, confidence, crop, timestamp 
FROM predictions 
WHERE user_id = ? 
ORDER BY timestamp DESC 
LIMIT 100;
```

---

## 9. Authentication Flow

### Demo/Dev Mode

**Bypass Credentials:**
- **Phone:** `1234567890`
- **OTP:** `123456`

**Flow:**
1. Frontend detects phone === "1234567890"
2. Skip /auth/send-otp call
3. Pre-fill OTP field with "123456"
4. User sees "Demo login enabled for 1234567890"
5. On verify, skip /auth/verify-otp call
6. Set isLoggedIn = true directly

**Use Case:** Development, testing, demos, easy onboarding

### Production Mode

**Flow:**
1. User enters real phone number
2. Frontend calls POST /auth/send-otp
3. Backend generates random 6-digit OTP
4. Twilio sends SMS to user
5. User enters OTP in frontend
6. Frontend calls POST /auth/verify-otp
7. Backend validates OTP against DB
8. On success: return user_id, set session
9. Frontend stores session, renders main app

---

## 10. Key Features

### 1. **Disease Prediction**
- Fine-tuned ResNet50 on 15 crop diseases
- Real-time inference (GPU accelerated)
- Confidence score + severity classification
- Inline remedy lookup (no extra API call)

### 2. **Multilingual Support**
- English (en) & Kannada (kn) UI
- Language toggle in header
- Kannada remedies from database
- Kannada text-to-speech playback

### 3. **Text-to-Speech**
- Sarvam AI integration
- Kannada female voice (anushka)
- 22050 Hz WAV output
- Base64 encoded in response
- HTML5 audio player with controls

### 4. **Prediction History**
- Last 100 predictions per user
- Recharts line chart (confidence trend)
- Compact list with timestamp
- Manual refresh button

### 5. **Sharing & Copy**
- Copy prediction text to clipboard (with toast)
- WhatsApp share link generation
- Pre-filled message template

### 6. **Responsive Design**
- Mobile-first layout
- Tailwind CSS breakpoints
- Grid layout (2 columns on desktop, 1 on mobile)
- Sticky sidebar on desktop
- Drag-and-drop upload on desktop

### 7. **Error Handling**
- User-friendly error messages
- Retry buttons
- Loading states for all async operations
- Network error detection
- Graceful fallbacks (ImageNet model if fine-tuned not available)

### 8. **CORS & Security**
- Backend allows CORS from all origins (`*`)
- Production: restrict to frontend domain
- HTTPS required in production
- OTP expiry (typically 10 minutes)
- Rate limiting recommended

---

## 11. Configuration

### Frontend (.env)

```env
VITE_API_URL=http://localhost:8000
```

### Backend (.env)

```env
# Sarvam AI TTS
SARVAM_API_KEY=your_api_key
SARVAM_TTS_URL=https://api.sarvam.ai/text-to-speech/stream
SARVAM_TTS_VOICE=anushka
SARVAM_TTS_MODEL=bulbul:v3
SARVAM_TTS_SAMPLE_RATE=22050
SARVAM_TIMEOUT_SECONDS=30

# Twilio OTP
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+1234567890
```

---

## 12. Deployment Considerations

### Frontend
- Build: `npm run build`
- Output: `dist/` folder
- Host on: Vercel, Netlify, S3 + CloudFront, etc.
- Environment: Set `VITE_API_URL` to backend domain

### Backend
- Deploy: Docker, Heroku, EC2, GCP Run, etc.
- Start: `python3 -m uvicorn backend.app_backend:app --host 0.0.0.0 --port 8000`
- Model: Place `poc/best_model.pth` and `poc/class_names.json` in repo
- Database: SQLite fine for MVP; migrate to PostgreSQL for scale
- Environment: Set TWILIO and SARVAM credentials

### CORS Production
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Replace with frontend domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## 13. Performance & Optimization

### Frontend
- **Image compression:** Resize before upload (max 2MB)
- **Lazy loading:** History chart rendered on-demand
- **Code splitting:** Components lazy-loaded via React.lazy()
- **Caching:** API responses cached where appropriate

### Backend
- **Model caching:** Loaded once at startup
- **Batch inference:** Support batch predictions (future enhancement)
- **Image optimization:** PIL auto-rotates EXIF, resizes to 224×224
- **DB indexing:** Index on `user_id` and `timestamp` for history queries
- **Rate limiting:** Implement token bucket for OTP requests

---

## 14. Testing Strategy

### Frontend
- Unit tests: Jest + React Testing Library
- E2E tests: Cypress or Playwright
- Components: Upload, Results, History, Auth
- Integration: API mocking with MSW (Mock Service Worker)

### Backend
- Unit tests: pytest
- Endpoints: /health, /predict, /auth/send-otp, /auth/verify-otp
- Model inference: Accuracy benchmarks on test set
- External services: Mock Twilio, Sarvam responses

---

## 15. Future Enhancements

- [ ] Multi-image batch prediction
- [ ] Real-time WebSocket updates for history
- [ ] Admin dashboard (prediction analytics, user management)
- [ ] Farmer profile & subscription tiers
- [ ] Push notifications for weather alerts
- [ ] Offline support (cached model + service worker)
- [ ] Video/live camera feed prediction
- [ ] Integration with agricultural extension services
- [ ] Crop & market price data
- [ ] Recommendation engine (crop rotation, fertilizer market rates)

---

## 16. Glossary

| Term               | Definition |
|--------------------|-----------|
| **ResNet50**       | 50-layer Residual Network; backbone for image classification |
| **Fine-tune**      | Transfer learning: freeze backbone, train new FC layer |
| **Confidence**     | Max probability from model output (0.0–1.0) |
| **Severity**       | Disease intensity classification (Low/Moderate/Severe) |
| **OTP**            | One-Time Password; 6-digit SMS-delivered auth code |
| **TTS**            | Text-to-Speech; Kannada audio generation |
| **Remedy**         | Treatment advice (cause, symptoms, steps, prevention, fertilizer) |
| **CORS**           | Cross-Origin Resource Sharing; browser security policy |
| **Uvicorn**        | ASGI server for FastAPI |
| **Vite**           | JavaScript build tool & dev server (fast cold start) |

---

## 17. Links & References

- **Frontend Code:** `frontend/src/`
- **Backend Code:** `backend/app_backend.py`
- **ML Pipeline:** `ml_pipeline/`
- **Data:** `data/karnataka_curated/`
- **Docs:** `docs/README.md`, `docs/POC_CHANGES_README.md`
- **Archive:** `archive/poc/` (training scripts, baseline models)

---

**Document Version:** 2.0.0  
**Last Modified:** 29 April 2026  
**Maintainer:** Abhinav
