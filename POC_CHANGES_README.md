# PoC Changes Implemented on 10 April 2026

This document summarizes all Proof of Concept work completed today across model evaluation, backend API, and frontend integration.

## Scope Summary

Three PoCs were implemented:

1. PoC 1: Baseline model evaluation script (PyTorch + torchvision, no training)
2. PoC 2: FastAPI backend with pretrained ResNet50 inference endpoint
3. PoC 3: Frontend ResultCard component with Tailwind, now connected to real prediction state

All changes were added inside the existing workspace and validated with local execution/build checks.

## PoC 1: Baseline Evaluation Script

### Files

- [poc/poc1_baseline_model.py](poc/poc1_baseline_model.py)
- [poc/poc1_results.txt](poc/poc1_results.txt)

### Goal

Evaluate a pretrained ResNet50 on the Karnataka curated dataset test split without training.

### What It Does

1. Loads pretrained ResNet50 using torchvision weights
2. Loads images from Karnataka curated test set
3. Runs inference only
4. Computes and prints accuracy percentage
5. Saves metrics to [poc/poc1_results.txt](poc/poc1_results.txt)

### Dataset Path Used

- data/karnataka_curated/dataset/test

### Runtime Controls

The script supports:

1. batch size
2. worker count
3. max samples (default set for quick execution)

### Validation Performed

Smoke test was run on 256 samples and completed successfully. Output file generation was confirmed.

## PoC 2: FastAPI Backend API

### Files

- [poc_api/app_backend.py](poc_api/app_backend.py)

### Goal

Serve image prediction via HTTP using pretrained ResNet50.

### Implemented Endpoints

1. GET /health
   - Response: {"status": "ok"}
2. POST /predict
   - Input: image file upload
   - Response JSON includes:
     - disease
     - confidence
     - severity
     - crop

### Backend Features

1. Loads pretrained ResNet50 once at app startup
2. Applies torchvision transforms for inference
3. Returns confidence score from softmax top class
4. Maps predicted ImageNet class to domain-like crop and disease labels
5. CORS enabled for all origins/methods/headers

### Run Command

python3 -m uvicorn app_backend:app --reload

Run from:

- poc_api

### API Checks Performed

1. GET /health verified OK
2. POST /predict verified with a real image from curated dataset

## PoC 3: Frontend ResultCard (Tailwind)

### Files

- [src/components/ResultCard.jsx](src/components/ResultCard.jsx)
- [src/App.jsx](src/App.jsx)
- [poc_frontend/README.md](poc_frontend/README.md)

### Goal

Add a responsive card UI for prediction results and integrate it into the app.

### UI Requirements Covered

1. Displays disease name in bold
2. Displays confidence progress bar from 0 to 100
3. Displays severity badge with fixed colors:
   - Early: #FFD700
   - Moderate: #FFA500
   - Severe: #E74C3C
4. Displays crop type
5. Uses Tailwind CSS only
6. Works on mobile and desktop

### Integration Status

1. Initially rendered with mock data for UI development
2. Updated to use real prediction state when available
3. Keeps mock fallback before first prediction
4. Confidence normalization handles both ranges:
   - 0 to 1
   - 0 to 100

### Build Validation

Frontend production build completed successfully after integration.

## Dependency Updates

### Python

Updated in [requirements.txt](requirements.txt):

1. torch==2.11.0
2. torchvision==0.26.0
3. fastapi>=0.135.3
4. uvicorn>=0.44.0
5. python-multipart>=0.0.24

### Frontend

No new npm packages were required for ResultCard.

## End-to-End Run Guide

### 1. Python Environment

Create and activate virtual environment if needed, then install dependencies from [requirements.txt](requirements.txt).

### 2. Run Backend

From project root:

1. cd poc_api
2. python3 -m uvicorn app_backend:app --reload

### 3. Configure Frontend API URL

Set VITE_API_URL in .env to backend base URL:

VITE_API_URL=http://localhost:8000

### 4. Run Frontend

From project root:

1. npm install
2. npm run dev

### 5. Test Backend Quickly

Health:

curl -X GET http://localhost:8000/health

Predict:

curl -X POST http://localhost:8000/predict -F "file=@data/karnataka_curated/dataset/test/Grape___Black_rot/ds1_24ec0660f5a8.jpg"

## Known Limitations

1. ResNet50 is pretrained on ImageNet, not fine-tuned on the crop disease dataset
2. Disease labels are currently heuristic mappings from ImageNet classes
3. Accuracy and disease naming are baseline/PoC quality, not production quality
4. For production quality, a domain-trained classifier is required

## Recommended Next Steps

1. Replace heuristic mapping with a model trained on curated disease classes
2. Add robust label schema shared by backend and frontend
3. Add endpoint tests for /predict with representative images
4. Add frontend loading and empty-state variants for ResultCard
5. Remove mock fallback once backend reliability is finalized
