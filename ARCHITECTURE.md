# Crop Disease Detection System - Architecture & Project Overview

## 🎯 Project Overview

**AgriVision Edge** is a multilingual crop disease detection system built for Indian farmers. The application uses AI-powered image analysis to detect crop diseases and provides instant treatment recommendations with audio support in Kannada and English.

**Key Features:**
- Real-time crop disease detection using EfficientNet-V2-S deep learning model
- Bilingual support (English & Kannada)
- AI-powered remedy generation using Gemini LLM
- Text-to-Speech (TTS) support via Sarvam AI
- User authentication with OTP verification
- Prediction history tracking
- Mobile-responsive design using Tailwind CSS

---

## 📁 Project Structure

```
crop_disease_detetction/
├── frontend/                    # React + Vite frontend
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── hooks/              # Custom React hooks
│   │   ├── utils/              # Utilities (API, i18n, colors)
│   │   ├── App.jsx             # Main application component
│   │   └── main.jsx            # Vite entry point
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── backend/                     # FastAPI Python backend
│   ├── app_backend.py          # Main FastAPI application
│   ├── database.py             # SQLite database operations
│   └── users.db                # User database
│
├── ml_pipeline/                # ML model utilities
│   ├── data_preprocessing.py
│   ├── validate_karnataka_datasets.py
│   └── test_data_pipeline.py
│
├── data/                        # Dataset storage
│   ├── karnataka_curated/      # Curated plant disease images
│   └── raw/                    # Raw dataset
│
├── best_crop_model.pth         # Trained EfficientNet-V2-S model
├── classes.txt                 # Disease class labels
├── requirements.txt            # Python dependencies
└── package.json               # Node dependencies
```

---

## 🏗️ Architecture Overview

### High-Level Flow

```
User Browser
    ↓
[React Frontend] ← → [FastAPI Backend] ← → [SQLite DB]
    ↓
[Vite Dev Server]
    ↓
[Tailwind CSS Styling]
    
API Integrations:
- Gemini API (LLM for remedy generation)
- Sarvam AI (Text-to-Speech)
- Twilio (SMS notifications)
```

---

## 💻 FRONTEND ARCHITECTURE

### Component Hierarchy

```
App.jsx (Root Component)
├── Login.jsx
├── Navbar.jsx
├── Hero.jsx
├── UploadSection.jsx        (Tab 01: Image Upload)
├── ResultsSection.jsx       (Tab 02: Results Display)
├── AIAdviceSection.jsx      (Expandable remedy details)
├── AudioPlayer.jsx          (TTS playback controls)
├── HowItWorks.jsx          (Tutorial section)
├── Features.jsx            (Feature highlights)
├── DiseaseHistory.jsx      (User prediction history)
├── Footer.jsx
└── ResponsiveLayout.jsx    (Mobile layout adapter)
```

### Frontend Components Detailed

#### **1. App.jsx (Main Container)**
**Purpose:** Root component managing application state and orchestrating all child components.

**Key Responsibilities:**
- Manages authentication state (`isLoggedIn`)
- Handles image file selection and preview generation
- Orchestrates API calls through `useApi()` hook
- Manages prediction results and remedy data
- Handles language toggle (English/Kannada)
- Manages audio playback nonce for re-triggering
- Error handling and user feedback

**State Management:**
```javascript
- selectedFile: File | null              // Currently selected image
- previewUrl: string                     // Blob URL for image preview
- prediction: { disease, crop, confidence } | null
- remedy: { english, kannada } | null   // LLM-generated treatment
- language: 'en' | 'kn'                 // Current UI language
- loadingPrediction: boolean
- loadingAdvice: boolean
- loadingAudio: boolean
- error: string | null                  // Error messages
```

**Key Methods:**
- `handleFileSelect()` - Updates selected file and resets results
- `handleAnalyze()` - Calls API to analyze image
- `handleRequestAdvice()` - Fetches LLM remedy
- `handleRequestAudio()` - Generates TTS audio

---

#### **2. Login.jsx (Authentication Gate)**
**Purpose:** User authentication using phone number and OTP verification.

**Features:**
- Two-tab interface: Login & Sign Up
- Phone number validation (10-digit Indian format)
- OTP-based verification (dev default: "123456")
- Language selection (persists to app)
- Bypass mode for development (default phone: "1234567890")

**Dev Credentials:**
```
Phone: 1234567890
OTP: 123456
```

**Flow:**
1. User enters phone → Sends OTP
2. User enters OTP → Backend verification (hardcoded for dev)
3. Verified → Redirects to main app

---

#### **3. UploadSection.jsx (Tab 01: Image Input)**
**Purpose:** Handles image file selection with drag-and-drop support.

**Features:**
- File drag-and-drop interface
- Image preview after selection
- File validation (image/* types only)
- Analyze button (disabled until image selected)
- Current selected file display

**Props:**
```javascript
{
  language: 'en' | 'kn',
  labels: UIText,
  selectedFile: File | null,
  previewUrl: string,
  loading: boolean,
  error: string | null,
  onFileSelect: (file) => void,
  onInvalidFile: (message) => void,
  onAnalyze: () => void
}
```

**Key Elements:**
- Upload button with drag-and-drop zone
- Image preview (max 256px height)
- Analyze button (orange, bottom-right)
- Loading spinner during analysis

---

#### **4. ResultsSection.jsx (Tab 02: Results Display)**
**Purpose:** Displays disease detection results with confidence and severity indicators.

**Result Data Structure:**
```javascript
{
  disease: string,           // e.g., "Early Blight"
  crop: string,              // e.g., "Tomato"
  confidence: number,        // 0-100 percentage
  severity: string           // LOW, MEDIUM, HIGH
}
```

**Features:**
- Disease name display
- Confidence percentage with color coding
- Severity badge (icon + label + color)
- Copy result to clipboard
- Share via WhatsApp
- "Get AI Advice" button (triggers LLM call)
- "Hear in Kannada" button (generates TTS)

**Severity Mapping (by confidence):**
- 0-40%: LOW (🟢 Green)
- 40-75%: MEDIUM (🟡 Yellow)
- 75-100%: HIGH (🔴 Red)

---

#### **5. AIAdviceSection.jsx (Remedy Details)**
**Purpose:** Expandable section displaying LLM-generated treatment recommendations.

**Remedy Structure:**
```javascript
{
  english: {
    cause: string,
    treatment_steps: string,
    prevention: string,
    pest_control: string
  },
  kannada: {
    cause: string,
    treatment_steps: string,
    prevention: string,
    pest_control: string
  }
}
```

**Features:**
- Collapsible/expandable card
- Language switcher (English ↔️ Kannada)
- Displays cause, treatment, and prevention
- Uses markdown rendering for formatted text

---

#### **6. AudioPlayer.jsx (TTS Playback)**
**Purpose:** Controls audio playback for TTS-generated remedy content.

**Features:**
- Play/Pause button
- Progress bar with duration
- Loading spinner during generation
- Language indicator
- Auto-generated summary (max 78 words for EN, 60 for KN)

---

#### **7. DiseaseHistory.jsx (Prediction History)**
**Purpose:** Displays past predictions in a card grid or timeline.

**History Item:**
```javascript
{
  id: string,
  disease: string,
  crop: string,
  confidence: number,
  timestamp: datetime,
  image_path: string
}
```

**Features:**
- Loads last 5 predictions
- Refresh button to reload
- Shows "No history" when empty
- Each card shows: disease, crop, confidence, date

---

#### **8. Supporting Components**
- **Navbar.jsx** - Top navigation with language toggle and logout
- **Hero.jsx** - Landing section with title and badges
- **HowItWorks.jsx** - 4-step tutorial (Upload → AI Detects → Advice → Share)
- **Features.jsx** - Feature highlights grid
- **Footer.jsx** - Copyright and links
- **ResultCard.jsx** - Reusable card for history items

---

### Frontend Utilities

#### **useApi.js (Custom Hook)**
**Purpose:** Centralized API call interface for all backend communication.

**Methods:**
```javascript
{
  predictDisease(file: File): Promise<PredictionResult>
  fetchRemedy(disease: string, crop: string): Promise<RemedyData>
  fetchHistory(): Promise<HistoryItem[]>
  fetchTts(text: string, language: 'en'|'kn'): Promise<Blob>
}
```

---

#### **api.js (API Client)**
**Purpose:** Low-level HTTP client for backend communication.

**Key Functions:**

1. **predictDisease(file)**
   - Endpoint: `POST /predict`
   - Upload image → Receive disease detection
   - Returns: `{ disease, crop, confidence }`

2. **fetchRemedy(disease, crop)**
   - Endpoint: `GET /remedy-llm?disease=X&crop=Y`
   - Queries Gemini LLM for treatment advice
   - Returns: `{ english: {...}, kannada: {...} }`

3. **fetchHistory()**
   - Endpoint: `GET /history`
   - Retrieves user's prediction history
   - Returns: `HistoryItem[]`

4. **fetchTts(text, language)**
   - Endpoint: `POST /tts`
   - Converts text to speech using Sarvam AI
   - Returns: `Blob` (audio/wav)

**Base URL Resolution:**
- Uses `VITE_API_URL` env variable (if set)
- Fallback: `http://localhost:8000`

---

#### **i18n.js (Internationalization)**
**Purpose:** Bilingual UI text management.

**Supported Languages:**
- `en` - English
- `kn` - Kannada

**Text Categories:**
- Form labels & buttons
- Error messages
- Feature descriptions
- Tooltips & hints

**Usage:**
```javascript
const labels = uiText[language];
// e.g., labels.uploadTitle, labels.analyze
```

---

#### **colors.js (Severity Colors)**
**Purpose:** Maps confidence scores to severity indicators.

**Mapping:**
```javascript
0-40%:   { color: '#10b981', label: 'LOW', icon: '🟢' }
40-75%:  { color: '#f59e0b', label: 'MEDIUM', icon: '🟡' }
75-100%: { color: '#ef4444', label: 'HIGH', icon: '🔴' }
```

---

### Frontend Data Flow

```
1. LOGIN
   User enters phone → Backend OTP check → Login success

2. IMAGE UPLOAD
   User selects image → Preview generated → Stored in state

3. ANALYSIS
   Click "Analyze" → API call /predict
   ← Returns { disease, crop, confidence }

4. DISPLAY RESULTS
   Results shown in ResultsSection
   Severity badge calculated from confidence

5. GET ADVICE
   Click "Get AI Advice" → API call /remedy-llm
   ← Returns LLM-generated treatment in both languages

6. AUDIO GENERATION
   Click "Hear in Kannada" → API call /tts
   ← Returns audio blob → Play in AudioPlayer

7. HISTORY
   Page loads → API call /history
   ← Returns last 5 predictions → Display in DiseaseHistory
```

---

## 🔧 BACKEND ARCHITECTURE

### FastAPI Application Structure

**File:** `backend/app_backend.py`

#### **Core Endpoints**

##### **1. POST /predict**
**Purpose:** Image classification for disease detection.

**Input:**
```
file: Image (multipart/form-data)
```

**Processing:**
1. Load pre-trained EfficientNet-V2-S model (or ResNet50 fallback)
2. Preprocess image: resize to 224×224, normalize with ImageNet stats
3. Run inference through model
4. Get top class prediction
5. Map class to (crop, disease) pair

**Output:**
```json
{
  "disease": "Early Blight",
  "crop": "Tomato",
  "confidence": 0.89
}
```

**Confidence Range:** 0.76 - 0.89 (confidence thresholds)
- Below 0.76: Marked as uncertain
- Above 0.89: High confidence

---

##### **2. GET /remedy-llm**
**Purpose:** Generate AI-powered treatment recommendations using Gemini LLM.

**Query Parameters:**
```
disease: string (required)  e.g., "Early Blight"
crop: string (optional)     e.g., "Tomato"
```

**Processing:**
1. Build remedy lookup key: `{Crop}_{Disease}` (e.g., "Tomato_Early Blight")
2. Query Gemini API with prompt:
   - "Generate treatment for [disease] on [crop]"
   - Request: cause, treatment steps, prevention, pest control
3. Parse response into structured format
4. Translate to Kannada (if needed)

**Output:**
```json
{
  "english": {
    "cause": "Fungal infection caused by...",
    "treatment_steps": "1. Remove infected leaves\n2. Apply fungicide...",
    "prevention": "Ensure proper air circulation...",
    "pest_control": "Use approved pesticides..."
  },
  "kannada": {
    "cause": "...(Kannada text)...",
    "treatment_steps": "...",
    "prevention": "...",
    "pest_control": "..."
  }
}
```

---

##### **3. GET /history**
**Purpose:** Fetch user's prediction history.

**Processing:**
1. Query SQLite database for user's predictions
2. Sort by timestamp (most recent first)
3. Return last 5 records

**Output:**
```json
[
  {
    "id": "uuid1",
    "disease": "Early Blight",
    "crop": "Tomato",
    "confidence": 0.89,
    "timestamp": "2024-05-04T10:30:00Z",
    "image_path": "/path/to/image.jpg"
  },
  ...
]
```

---

##### **4. POST /tts**
**Purpose:** Text-to-speech generation using Sarvam AI.

**Input:**
```json
{
  "text": "Disease: Tomato Early Blight. Cause: Fungal infection...",
  "language": "kn"  // or "en"
}
```

**Processing:**
1. Validate text length (max 78 words for EN, 60 for KN)
2. Call Sarvam AI `/text-to-speech/stream` endpoint
3. Set voice: "anushka" (Kannada female voice)
4. Set model: "bulbul:v3"
5. Return audio stream

**Output:**
```json
{
  "audio_base64": "SUQzBAAAAAAAI1NT...",
  "mime_type": "audio/wav",
  "duration": 12.5
}
```

**Encoding:** Base64 (converted to Blob on frontend)

---

#### **Database Schema**

**File:** `backend/database.py`

**Table: users**
```sql
CREATE TABLE users (
  id TEXT PRIMARY KEY,           -- UUID or phone number
  phone TEXT UNIQUE NOT NULL,    -- 10-digit Indian phone
  name TEXT,                     -- User's full name
  created_at TIMESTAMP,          -- Account creation
  last_login TIMESTAMP
)
```

**Table: predictions**
```sql
CREATE TABLE predictions (
  id TEXT PRIMARY KEY,           -- UUID
  user_id TEXT NOT NULL,         -- Foreign key to users
  image_path TEXT,               -- Path to uploaded image
  disease TEXT NOT NULL,         -- Detected disease
  crop TEXT NOT NULL,            -- Detected crop
  confidence FLOAT,              -- 0-1 confidence score
  timestamp TIMESTAMP,           -- Prediction time
  FOREIGN KEY (user_id) REFERENCES users(id)
)
```

---

#### **Model Configuration**

**Model Architecture:**
- **Primary:** EfficientNet-V2-S (ImageNet pre-trained, then fine-tuned)
- **Fallback:** ResNet50 (if primary unavailable)
- **Input Size:** 224×224 pixels
- **Classes:** 15 crop-disease combinations
- **Training Framework:** PyTorch
- **Training Hardware:** Apple Silicon MacBook Pro (16GB, MPS backend)
- **Epochs:** 5 epochs on ~66k images
- **Batch Size:** 16
- **Optimizer:** AdamW (lr=0.0001)
- **Loss Function:** CrossEntropyLoss

**📚 See [MODEL_TRAINING.md](./MODEL_TRAINING.md) for full training details, retraining guide, and performance optimization.**

**Class Mapping:**
```python
CLASS_LABELS = {
    "Pepper__bell___Bacterial_spot": ("Pepper", "Bacterial Spot"),
    "Potato___Early_blight": ("Potato", "Early Blight"),
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
    "Potato___healthy": ("Potato", "Healthy"),
    "Pepper__bell___healthy": ("Pepper", "Healthy"),
    "Grape___Black_rot": ("Grape", "Black Rot"),
    ... (more classes)
}
```

---

#### **External API Integrations**

**1. Gemini API (Google)**
```
Base URL: https://generativelanguage.googleapis.com/v1beta/models
Model: gemini-1.5-flash
Purpose: LLM-based remedy generation
Auth: GEMINI_API_KEY environment variable
```

**2. Sarvam AI (Text-to-Speech)**
```
Base URL: https://api.sarvam.ai
Endpoint: /text-to-speech/stream
Voice: anushka
Model: bulbul:v3
Sample Rate: 22050 Hz
Auth: SARVAM_API_KEY environment variable
Languages: en (English), kn (Kannada)
```

**3. Twilio (SMS - Optional)**
```
Purpose: OTP delivery for authentication
Auth: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
```

---

#### **Environment Variables**

**Required (.env file):**
```bash
# API Keys
GEMINI_API_KEY=your_gemini_api_key
SARVAM_API_KEY=your_sarvam_api_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token

# URLs
SARVAM_TTS_URL=https://api.sarvam.ai/text-to-speech/stream
VITE_API_URL=http://localhost:8000

# Model Configuration
SARVAM_TTS_VOICE=anushka
SARVAM_TTS_MODEL=bulbul:v3
SARVAM_TTS_SAMPLE_RATE=22050
SARVAM_TIMEOUT_SECONDS=30

# Confidence Thresholds
CONFIDENCE_MIN=0.76
CONFIDENCE_MAX=0.89
```

---

#### **CORS Configuration**

**FastAPI CORS Middleware:**
```python
CORSMiddleware(
    app,
    allow_origins=["*"],  # TODO: Restrict to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🔗 Frontend ↔️ Backend Integration

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER BROWSER (FRONTEND)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Login Page]                                                     │
│      ↓ (onLogin)                                                 │
│  [App.jsx] ──state: isLoggedIn, language, selectedFile, etc      │
│      ↓                                                            │
│  [UploadSection] ←→ [ResultsSection]                             │
│      (Tab 01)        (Tab 02)                                    │
│      ↓               ↓                                            │
│      │       [AIAdviceSection]                                   │
│      │               ↓                                            │
│      │       [AudioPlayer]                                       │
│      │                                                            │
│      └─ useApi() hook ──────→ [API Client]                       │
│                                    ↓                             │
│                           ┌────────────────────────┐              │
│                           │   HTTP Requests        │              │
│                           ├────────────────────────┤              │
│                           │ POST /predict          │              │
│                           │ GET /remedy-llm        │              │
│                           │ GET /history           │              │
│                           │ POST /tts              │              │
│                           └────────────────────────┘              │
│                                    ↓ (to http://localhost:8000)  │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND SERVER (FASTAPI)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [app_backend.py]                                                │
│      ├─ Route: POST /predict                                     │
│      │    └─ Load model → Preprocess → Inference → Return JSON   │
│      │         ↓ (model: best_crop_model.pth)                    │
│      │    [EfficientNet-V2-S or ResNet50]                        │
│      │                                                            │
│      ├─ Route: GET /remedy-llm                                   │
│      │    └─ Build prompt → Call Gemini API → Parse → Return JSON│
│      │         ↓ (API: GEMINI_API_KEY)                           │
│      │    [Gemini LLM]                                           │
│      │                                                            │
│      ├─ Route: GET /history                                      │
│      │    └─ Query DB → Sort → Return JSON                       │
│      │         ↓ (database.py)                                   │
│      │    [SQLite: users.db]                                     │
│      │                                                            │
│      └─ Route: POST /tts                                         │
│           └─ Validate text → Call Sarvam AI → Return audio      │
│                ↓ (API: SARVAM_API_KEY)                           │
│           [Sarvam AI TTS]                                        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Running the Project

### Prerequisites
```bash
Python 3.9+
Node.js 16+
pip
npm or yarn
```

### Setup Backend

1. **Install Python dependencies:**
```bash
cd /Users/abhinav/crop_disease_detetction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Setup environment variables:**
```bash
# Create .env file in project root
GEMINI_API_KEY=your_key_here
SARVAM_API_KEY=your_key_here
VITE_API_URL=http://localhost:8000
```

3. **Run backend server:**
```bash
cd /Users/abhinav/crop_disease_detetction
source .venv/bin/activate
uvicorn backend.app_backend:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: `http://localhost:8000`

---

### Setup Frontend

1. **Install Node dependencies:**
```bash
cd frontend
npm install
```

2. **Create .env file:**
```bash
VITE_API_URL=http://localhost:8000
```

3. **Run development server:**
```bash
npm run dev
```

Frontend will be available at: `http://localhost:5173`

---

## 📊 Key Component Dependencies

### Frontend Component Tree
```
App (state management hub)
├── Uses: useApi() hook
├── Uses: uiText (i18n)
├── Uses: getSeverityInfo() (colors)
│
├─ Login
│  └─ Auth flow
│
├─ Navbar
│  └─ Language toggle, logout
│
├─ UploadSection
│  └─ File input & preview
│  └─ Triggers: handleAnalyze()
│
├─ ResultsSection
│  └─ Displays prediction
│  └─ Triggers: handleRequestAdvice()
│  └─ Triggers: handleRequestAudio()
│
├─ AIAdviceSection
│  └─ Displays remedy data
│  └─ Language switcher
│
├─ AudioPlayer
│  └─ Plays TTS audio
│  └─ Progress control
│
└─ DiseaseHistory
   └─ Displays past predictions
```

### Backend Dependency Flow
```
FastAPI App (app_backend.py)
├─ Models
│  ├─ EfficientNet-V2-S (best_crop_model.pth)
│  └─ ResNet50 (fallback)
│
├─ Database (database.py)
│  └─ SQLite (users.db)
│
├─ External APIs
│  ├─ Gemini (remedy generation)
│  ├─ Sarvam AI (text-to-speech)
│  └─ Twilio (SMS/OTP)
│
└─ Routes
   ├─ /predict
   ├─ /remedy-llm
   ├─ /history
   └─ /tts
```

---

## 🔐 Authentication Flow

```
Frontend (Login.jsx)
│
├─ User enters phone number
├─ Submits to backend
│
Backend (FastAPI)
│
├─ Verifies phone format
├─ Checks OTP (dev: hardcoded "123456")
├─ Creates or updates user record in DB
├─ Returns success
│
Frontend
│
└─ Sets isLoggedIn = true
   Redirects to main app
```

---

## ⚠️ Important Notes for Team Members

### 1. **API Base URL Configuration**
- Frontend looks for `VITE_API_URL` environment variable
- If not set, defaults to `http://localhost:8000`
- Make sure backend is running on the same port

### 2. **CORS Settings**
- Currently set to allow all origins (`["*"]`)
- **TODO:** Restrict to specific frontend URL in production

### 3. **Image Upload Size Limits**
- Max file size: Check FastAPI defaults (default 25MB)
- Supported formats: JPEG, PNG, WebP
- Resolution: Automatically resized to 224×224

### 4. **Confidence Score Interpretation**
- 0.76 - 0.89: Normal confidence range
- < 0.76: Low confidence (uncertain prediction)
- > 0.89: Very high confidence (extremely certain)

### 5. **Language Support**
- Frontend: English (en) and Kannada (kn)
- TTS: Generated separately for each language
- LLM Responses: Provided in both languages

### 6. **Remedy Data Source**
- Generated by Gemini LLM at runtime
- Not cached (new request each time)
- Format: Structured JSON with cause, treatment, prevention

### 7. **Model Fallback Mechanism**
- Tries to load `best_crop_model.pth` first
- If unavailable, falls back to ResNet50 (ImageNet trained)
- Fallback model may have lower accuracy

### 8. **User History Storage**
- Stores last 5 predictions per user
- Image stored at path in DB (not as BLOB)
- Indexed by user_id and timestamp for quick retrieval

---

## 🔄 Common Workflows

### Workflow 1: New User Login & First Prediction
```
1. User opens app → Login page
2. Enter phone (default: 1234567890)
3. Enter OTP (default: 123456)
4. Click "Verify & Login"
5. Redirected to main app
6. Upload image
7. Click "Analyze"
8. See results in 2-3 seconds
9. Click "Get AI Advice" for treatment
10. Click "Hear in Kannada" for audio
```

### Workflow 2: View History
```
1. Main app loads
2. DiseaseHistory component auto-fetches history
3. Shows last 5 predictions as cards
4. Click refresh to reload
```

### Workflow 3: Share Results
```
1. After prediction, click "Share via WhatsApp"
2. Generates shareable summary
3. Opens WhatsApp with pre-filled message
```

---

## 🐛 Debugging Tips

### Frontend Issues
- Check browser console (F12) for errors
- Verify backend is running: `curl http://localhost:8000/docs`
- Check network tab to see API calls
- Verify `VITE_API_URL` is set correctly

### Backend Issues
- Check terminal where uvicorn is running
- Verify model file exists at path
- Check API keys in `.env` file
- Test endpoints with curl:
  ```bash
  curl http://localhost:8000/docs  # Swagger UI
  ```

### Image Upload Issues
- Ensure file is valid image format
- Check browser console for FormData errors
- Verify file size is within limits

### TTS Issues
- Check Sarvam API key in `.env`
- Verify internet connection
- Check audio browser console errors
- Test with shorter text first

---

## 📝 Next Steps for Team

1. **Setup Development Environment**
   - Clone repo
   - Run `npm install` in frontend
   - Run `pip install -r requirements.txt` in backend
   - Set up `.env` files

2. **Understand the Flow**
   - Review this document
   - Trace through App.jsx component tree
   - Test the login flow with dev credentials

3. **API Testing**
   - Visit http://localhost:8000/docs for Swagger UI
   - Try uploading test images
   - Verify all endpoints working

4. **Set Team Preferences**
   - Default language (currently English)
   - Confidence thresholds
   - API timeouts

---

## 📞 Support & Questions

Refer to:
- Backend code comments in `app_backend.py`
- Component JSDoc in frontend files
- This architecture document
- Environment variables guide above

---

**Document Version:** 1.0  
**Last Updated:** May 4, 2026  
**Project:** AgriVision Edge - Crop Disease Detection System
