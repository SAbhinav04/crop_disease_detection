# Frontend PoC: ResultCard

This folder documents the UI-only Proof of Concept for displaying a disease prediction card with mock data.

## Implemented Files

- `src/components/ResultCard.jsx`
- `src/App.jsx`

## Requirements Covered

1. Hardcoded disease result (no API call)
2. Card content:
   - Disease name (bold)
   - Confidence bar (0-100%)
   - Severity badge (color-coded)
   - Crop type
3. Tailwind CSS-only styling
4. Mobile responsive layout

## Severity Color Mapping

- Early: `#FFD700`
- Moderate: `#FFA500`
- Severe: `#E74C3C`

## Mock Data in App

- disease: `Apple_Black_rot`
- confidence: `92`
- severity: `Moderate`
- crop: `Apple`

## Run

```bash
npm install
npm run dev
```

## Build Check

```bash
npm run build
```

Build succeeds with the current implementation.
