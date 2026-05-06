# AgriVision Edge Setup Guide

This guide walks through the full local setup for the crop disease detection system, including backend, frontend, Gemini, Sarvam, Twilio, and the bundled model files.

## What You Need

- macOS, Linux, or Windows
- Python 3.9 or newer
- Node.js 16 or newer
- A Google account for Gemini API access
- A Sarvam AI account for text-to-speech
- Optional: a Twilio account if you want SMS OTP delivery instead of mock OTP logging

## 1) Clone the repository

```bash
git clone <your-repo-url>
cd crop_disease_detetction
```

If you are already inside the repository, skip this step.

## 2) Create the Python environment

The backend is a Python FastAPI app. Create and activate a virtual environment from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If `python3` is not available, use your system's Python launcher that points to Python 3.9+.

## 3) Install backend dependencies

```bash
pip install -r requirements.txt
```

This installs the ML stack, FastAPI, Uvicorn, Pillow, Requests, Twilio, and the optional data tools used by the repo.

## 4) Get a Gemini API key

Gemini is used only for dynamic remedy generation in `/remedy-llm`. If you do not set a Gemini key, the backend falls back to the built-in local remedy dictionary.

### Create the key

1. Open Google AI Studio: https://aistudio.google.com/app/apikey
2. Sign in with your Google account.
3. If prompted, create or import a Google Cloud project.
4. Open the API keys page.
5. Click Create API key and copy the generated value.

### Keep it safe

- Do not commit the key to git.
- Treat it like a password.
- Restrict the key in Google Cloud Console if you plan to use it beyond local development.

### Add it to your environment

Create or update a `.env` file in the repository root:

```bash
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-1.5-flash
```

`GEMINI_MODEL` is optional. If omitted, the backend defaults to `gemini-1.5-flash`.

## 5) Get a Sarvam API key

Sarvam is used by the `/tts` endpoint to generate Kannada or English speech audio.

### Create the key

1. Open the Sarvam dashboard: https://dashboard.sarvam.ai/
2. Sign in or create an account.
3. Create a new API key from the dashboard.
4. Copy the key and store it securely.

The Sarvam quickstart confirms that the authentication header expected by the API is `api-subscription-key`, which is what the backend sends.

### Add it to your environment

Add these lines to the same root `.env` file:

```bash
SARVAM_API_KEY=your_sarvam_key_here
SARVAM_TTS_URL=https://api.sarvam.ai/text-to-speech/stream
SARVAM_TTS_VOICE=anushka
SARVAM_TTS_MODEL=bulbul:v3
SARVAM_TTS_SAMPLE_RATE=22050
SARVAM_TIMEOUT_SECONDS=30
```

### What those settings do

- `SARVAM_API_KEY` authenticates `/tts`
- `SARVAM_TTS_URL` points to the Sarvam streaming TTS endpoint
- `SARVAM_TTS_VOICE` chooses the voice
- `SARVAM_TTS_MODEL` selects the TTS model
- `SARVAM_TTS_SAMPLE_RATE` controls output sample rate
- `SARVAM_TIMEOUT_SECONDS` caps the request timeout

If `SARVAM_API_KEY` is missing, `/tts` returns a server error, so this key is required for audio playback.

## 6) Optional: configure Twilio for SMS OTP

The login flow supports SMS OTP delivery when Twilio credentials are present. If you do not configure Twilio, the backend prints the OTP to the terminal instead, which is fine for local development.

### Create the Twilio credentials

1. Create or sign in to a Twilio account.
2. Open the Twilio Console.
3. Copy your Account SID and Auth Token.
4. Buy or configure a Twilio phone number that can send SMS.

### Add them to `.env`

```bash
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
```

## 7) Configure the frontend environment

The frontend reads Vite environment variables from the `frontend/` project folder. Create a `.env` file inside `frontend/`:

```bash
VITE_API_URL=http://localhost:8000
```

This points the React app at the FastAPI backend.

If you leave `VITE_API_URL` unset, the frontend falls back to `http://localhost:8000`, but keeping it explicit is better.

## 8) Verify the model assets

The backend expects these files in the repository root:

- `best_crop_model.pth`
- `classes.txt`

What they do:

- `best_crop_model.pth` contains the fine-tuned EfficientNet-V2-S weights
- `classes.txt` contains the class labels used during inference

The backend now loads them using relative paths, so the repository can be moved to another folder without editing code.

At startup the backend tries to load the fine-tuned model. If that fails, it falls back to an ImageNet ResNet50 model. That fallback keeps the API running, but the predictions are much less useful than the fine-tuned model.

If you ever need to retrain the model, check the ML scripts under `archive/poc/` and `ml_pipeline/`. Retraining is not required to run the app locally.

## 9) Start the backend

From the repository root, with the virtual environment activated:

```bash
uvicorn backend.app_backend:app --reload --host 0.0.0.0 --port 8000
```

You should then be able to open:

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

The health endpoint reports whether the fine-tuned model loaded successfully.

## 10) Start the frontend

You can start the frontend either from the repository root or from the `frontend/` directory.

### Option A: from the repository root

```bash
npm install
npm run dev
```

### Option B: from the frontend folder

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server should be available at http://localhost:5173.

## 11) First-run checklist

1. Backend terminal shows the model load message.
2. `http://localhost:8000/health` returns `status: ok`.
3. `http://localhost:8000/docs` opens the FastAPI Swagger UI.
4. Frontend opens at `http://localhost:5173`.
5. Uploading an image returns a prediction.
6. `/remedy-llm` works when `GEMINI_API_KEY` is configured, or uses the local fallback when it is not.
7. `/tts` works when `SARVAM_API_KEY` is configured.

## 12) Test the main endpoints

### Predict an image

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"
```

### Get remedy advice

```bash
curl "http://localhost:8000/remedy-llm?disease=Early%20Blight&crop=Tomato"
```

### Generate TTS audio

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Disease detected","language":"kn"}'
```

### Check history

```bash
curl http://localhost:8000/history
```

## 13) Authentication flow

The app uses phone number + OTP login.

- If Twilio is configured, OTPs are sent by SMS.
- If Twilio is not configured, the backend prints the OTP to the terminal.
- OTP records are stored temporarily in memory, while verified users are stored in SQLite.

The SQLite database file is created automatically at `backend/users.db`.

## 14) Environment variable summary

| Variable | Required | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | No | Enables dynamic remedy generation |
| `GEMINI_MODEL` | No | Gemini model name, defaults to `gemini-1.5-flash` |
| `SARVAM_API_KEY` | Yes for `/tts` | Authenticates Sarvam TTS requests |
| `SARVAM_TTS_URL` | No | Sarvam TTS endpoint |
| `SARVAM_TTS_VOICE` | No | Voice name used for TTS |
| `SARVAM_TTS_MODEL` | No | TTS model name, defaults to `bulbul:v3` |
| `SARVAM_TTS_SAMPLE_RATE` | No | Output sample rate |
| `SARVAM_TIMEOUT_SECONDS` | No | Request timeout for Sarvam |
| `TWILIO_ACCOUNT_SID` | No | Enables SMS OTP |
| `TWILIO_AUTH_TOKEN` | No | Enables SMS OTP |
| `TWILIO_PHONE_NUMBER` | No | Sender number for SMS OTP |
| `VITE_API_URL` | Recommended | Frontend backend URL |

## 15) Common problems

### Backend starts, but predictions look generic

Check whether the backend reports `model: imagenet-fallback` on `/health`. If it does, the fine-tuned weights or class labels were not loaded.

### `/tts` returns 500

Confirm that `SARVAM_API_KEY` is set and that the Sarvam endpoint is reachable.

### `/remedy-llm` does not use Gemini

If `GEMINI_API_KEY` is missing, the backend intentionally falls back to the static remedy dictionary.

### Frontend cannot reach the backend

Make sure `VITE_API_URL=http://localhost:8000` is set in `frontend/.env`, and that the backend is running on port 8000.

### Login OTP never arrives

That usually means Twilio is not configured. Use the backend terminal output in local development or add the Twilio credentials.

## 16) What the model currently supports

The committed model is a fine-tuned EfficientNet-V2-S classifier trained on the crop-disease label set defined in `classes.txt`.

The backend uses those labels to map predictions into crop, disease, confidence, severity, and history entries. The remedy layer then looks up treatment text for the predicted crop/disease pair.

For deeper model notes and retraining work, the relevant code lives in:

- `backend/app_backend.py`
- `classes.txt`
- `best_crop_model.pth`
- `archive/poc/train_model.py`
- `ml_pipeline/`
