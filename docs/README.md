# Crop Disease Detection Frontend

A responsive React + Vite frontend for crop disease detection. It supports English/Kannada UI toggling, image upload, disease prediction, AI-generated remedies, Kannada text-to-speech playback, and prediction history with a confidence trend chart.

## PoC Implementation Log

For a detailed log of all PoC changes implemented on 10 April 2026, see:

- [POC_CHANGES_README.md](POC_CHANGES_README.md)

## New Mock Result Card

- Added a Tailwind-only `ResultCard` component for static UI prototyping.
- It currently renders mock data in the app (no API call required).
- The card includes:
  - bold disease name
  - confidence bar (0-100%)
  - severity badge with fixed colors:
    - Early: `#FFD700`
    - Moderate: `#FFA500`
    - Severe: `#E74C3C`
  - crop type

## Highlights

- Mobile-first layout that adapts to desktop
- Image upload with drag-and-drop support
- Prediction results with disease, confidence, severity, and crop type
- Color-coded severity badges: mild, moderate, severe
- AI advice panel with English and Kannada content
- Kannada audio playback for the recommendation text
- Prediction history graph using Recharts
- Copy result and WhatsApp share actions
- Friendly empty states and loading indicators

## Tech Stack

- React 18
- Vite
- Tailwind CSS
- Recharts

## Backend API

Set `VITE_API_URL` to your backend base URL.

Expected endpoints:

- `POST /predict`
  - Accepts an image file
  - Returns:
    - `disease`
    - `confidence`
    - `severity`
    - `crop`

- `GET /remedy-llm?disease=X`
  - Returns AI advice in both English and Kannada
  - Expected shape:
    - `english: { cause, symptoms, treatment_steps, prevention, fertilizer_recommendation }`
    - `kannada: { cause, symptoms, treatment_steps, prevention, fertilizer_recommendation }`

- `POST /tts`
  - Accepts `{ text, language }`
  - Returns audio as either:
    - a binary audio file, or
    - JSON containing a base64 audio payload

- `GET /history`
  - Returns the latest prediction history

## Local Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment

Create a `.env` file in the project root:

```bash
VITE_API_URL=http://localhost:8000
```

Update the URL to match your backend if it runs elsewhere.

### 3. Start the app

```bash
npm run dev
```

### 4. Build for production

```bash
npm run build
```

### 5. Preview the production build

```bash
npm run preview
```

## Project Structure

```text
src/
├── components/
│   ├── Header.jsx
│   ├── UploadSection.jsx
│   ├── ResultsSection.jsx
│   ├── AIAdviceSection.jsx
│   ├── DiseaseHistory.jsx
│   ├── AudioPlayer.jsx
│   ├── ResultCard.jsx
│   └── ResponsiveLayout.jsx
├── hooks/
│   ├── useApi.js
│   └── useResponsive.js
├── utils/
│   ├── api.js
│   ├── colors.js
│   └── i18n.js
├── App.jsx
├── App.css
├── index.css
└── main.jsx
```

## UI Behavior

### Mock Result Card

- A mock result card is displayed in the left panel via `ResultCard`.
- Current mock payload in `App.jsx`:
  - `disease: Apple_Black_rot`
  - `confidence: 92`
  - `severity: Moderate`
  - `crop: Apple`
- Designed to be mobile responsive with Tailwind utility classes.

### Upload

- Accepts image files only
- Shows image preview after selection
- Supports drag-and-drop on desktop
- Displays a friendly validation message for non-image files

### Results

- Shown after a prediction is returned
- Includes disease name, confidence bar, severity badge, and crop type
- Severity colors:
  - Mild/Early: amber
  - Moderate: orange
  - Severe: red

### AI Advice

- Expandable section
- Language tabs let you switch between English and Kannada content
- Displays cause, symptoms, treatment steps, prevention, and fertilizer recommendation

### Audio

- Requests a Kannada TTS version of the advice text
- Shows loading state while generating audio
- Displays an inline audio player after the response is ready

### History

- Loads the last 10 predictions on mount
- Shows a Recharts line graph for confidence over time
- Includes a compact list of the latest predictions

## Environment Notes

- The frontend expects CORS to be enabled on the backend
- If the `/tts` endpoint returns base64 audio, the client will convert it to a blob automatically
- Kannada text rendering is supported through the loaded font stack

## Scripts

- `npm run dev` - start the Vite dev server
- `npm run build` - build the app for production
- `npm run preview` - preview the production build locally

## Sharing With Teammates

A teammate only needs to:

1. Clone the repo
2. Run `npm install`
3. Add `VITE_API_URL` to `.env`
4. Run `npm run dev`

## Notes

- The app is ready for backend integration, but it still depends on the API contract above.
- The production build currently emits a bundle-size warning because of the charting library. It does not block usage.
- If needed, the history chart can be split into a lazy-loaded chunk later to reduce bundle size.


## Model Evaluation Results

- Total Images: 12,104  
- Accuracy: 96.42%  
- Precision: 96.64%  
- Recall: 96.42%  
- F1 Score: 96.46%  
- Average Inference Time: 137.65 ms  

### Observations
- The model shows high accuracy and balanced performance.
- Inference time is efficient for near real-time usage.
- Minor misclassifications exist in edge cases (from confusion matrix).

### Detailed Output (for reference)
========== FINAL RESULTS ==========
Total images: 12104
Total processed: 12104

Accuracy: 96.42%
Precision: 96.64%
Recall: 96.42%
F1 Score: 96.46%

Average Inference Time per image: 137.65 ms

Confusion Matrix:
[[121   0   0 ...   0   0   0]
 [  0 378   0 ...   0   0   0]
 [  0   0 285 ...   0   0   0]
 ...
 [  0   0   0 ... 593   5  20]
 [  0   0   0 ...  13 591  11]
 [  0   0   0 ...  37  15 517]]


 ## TTS Evaluation (Sarvam AI)

### Summary
- The `/tts` endpoint works correctly and returns valid audio responses.
- Kannada speech generation is functional and understandable.
- End-to-end flow (Frontend → Backend → Sarvam API → Audio) is working.

---

### Observations
- Latency observed: ~12–15 seconds per request.
- Backend TTS integration is stable and reliable.
- Audio quality is acceptable but slightly robotic.
- Frontend audio playback is inconsistent due to blob handling issues.

---

### Issues Identified
- Audio playback interruption due to multiple blob requests (frontend bug).
- Missing Twilio dependency in `requirements.txt` (fixed).

---

### Evidence
- Network logs confirm `/tts` endpoint returns **200 OK**.
- Audio payload (~600–700 KB) confirms successful TTS generation.
- Blob media requests indicate playback inconsistency in frontend.

---

### Conclusion
Backend TTS functionality is working correctly; however, frontend audio playback handling is inconsistent and requires improvement. Performance optimization (latency reduction) would further enhance user experience.