# NeuroScan AI – Brain Tumor Batch Analysis

NeuroScan AI is a web application for batch classification of brain MRI images (128x128) into tumor/no-tumor categories using two CNN models with Grad-CAM heatmaps for explainable AI visualizations.

## Features
- Batch processing: Upload multiple MRI scans simultaneously for efficient analysis.
- Dual CNN models: Choose between Baseline ConvNet or Deep ConvNet (recommended).
- Grad-CAM visualization: Side-by-side original image and AI heatmap highlighting potential tumor regions.
- Responsive UI: Modern upload interface with loading spinner, file count badge, and error handling.

## Quick Start

1. **Prerequisites**  
   - Python 3.9+ installed.  
   - Modern web browser (Firefox/Chrome).  

2. **Project Structure (example)**  
```
deploy
├── frontend/
│ └── index.html # Batch upload UI with results grid
├── backend/
│ ├── main.py # FastAPI server, model loading, prediction + Grad-CAM
│ └── modelbrain.py # CNN model definitions (BrainTumorConvNet, BrainTumorDeepConvNet)
├── models/ # Pre-trained .pth model weights
├── requirements.txt # Dependencies
└── README.md # This file
```
Backend loads model weights from the `models/` directory via `../models` relative path.

3. **Install Dependencies**  
From the deploy folder where `requirements.txt` is located, run:
```
pip install -r requirements.txt
```
Core dependencies include:
fastapi
uvicorn
python-multipart
torch
torchvision
grad-cam
pillow
numpy
lightning

4. **Start the Backend API**  
From the backend folder"
```
python main.py
```

5. **Open the Frontend**  
- Open `frontend/index.html` (or `index.html` if flat) directly in your browser.  
- The frontend is configured to send requests to `http://127.0.0.1:8000/predict`.

6. **Usage Flow**  
- Click the upload box and select multiple MRI scans (JPG/PNG).  
- Select a model from the dropdown:
  - Deep ConvNet (recommended)  
  - Baseline ConvNet  
- Click the "Analyze Batch" button.  
- Inspect the results grid showing:
  - Filename  
  - Original MRI image  
  - AI heatmap (Grad-CAM)  
  - Prediction label