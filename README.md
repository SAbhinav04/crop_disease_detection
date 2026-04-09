# Crop Disease Detection Frontend

A responsive React + Vite frontend for crop disease detection. It supports English/Kannada UI toggling, image upload, disease prediction, AI-generated remedies, Kannada text-to-speech playback, and prediction history with a confidence trend chart.

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
