from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import os
from typing import List

# --- IMPORT YOUR SPECIFIC MODELS ---
from model_brain import BrainTumorConvNet, BrainTumorDeepConvNet

app = FastAPI(title="NeuroScan AI API", description="Brain Tumor Detection API with Grad-CAM")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "../models/"

# Global dictionary to hold loaded models
loaded_models = {}

# --- LOAD MODELS ---
@app.on_event("startup")
def load_models():
    print("Loading models...")
    
    # 1. Load Baseline Model
    try:
        path = os.path.join(MODELS_DIR, "BrainTumorConvNet.pth")
        if os.path.exists(path):
            baseline = BrainTumorConvNet(num_classes=NUM_CLASSES)
            baseline.load_state_dict(torch.load(path, map_location=DEVICE))
            baseline.to(DEVICE)
            baseline.eval()
            loaded_models["baseline"] = baseline
            print("Baseline (BrainTumorConvNet) loaded.")
        else:
            print(f"File not found: {path}")
    except Exception as e:
        print(f"Failed to load Baseline: {e}")

    # 2. Load Deep Model
    try:
        path = os.path.join(MODELS_DIR, "BrainTumorDeepConvNet.pth")
        if os.path.exists(path):
            deep = BrainTumorDeepConvNet(num_classes=NUM_CLASSES)
            deep.load_state_dict(torch.load(path, map_location=DEVICE))
            deep.to(DEVICE)
            deep.eval()
            loaded_models["deep"] = deep
            print("✅ Deep (BrainTumorDeepConvNet) loaded.")
        else:
            print(f"File not found: {path}")
    except Exception as e:
        print(f"Failed to load Deep Model: {e}")

# --- HELPER: Process Image ---
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = my_transforms(image).unsqueeze(0).to(DEVICE)
    return tensor, image

# --- HELPER: Generate Heatmap (Grad-CAM) ---
def get_gradcam_overlay(model, model_type, input_tensor, original_image):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    if model_type == "deep":
        target_layers = [model.model[6]]
    else:
        target_layers = [model.model[7]]

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        rgb_img = np.float32(original_image.resize((128, 128))) / 255
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        pil_img = Image.fromarray(visualization)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        return None

# --- UPDATED ENDPOINT FOR MULTIPLE FILES ---
@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...), 
    model_name: str = Form("deep")
):
    if model_name not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available.")
    
    selected_model = loaded_models[model_name]
    results = []

    for file in files:
        try:
            # 1. Read Image
            image_bytes = await file.read()
            tensor, original_pil = transform_image(image_bytes)
            
            # 2. Get Prediction
            with torch.no_grad():
                outputs = selected_model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted_class = torch.max(probabilities, 0)
            
            class_index = predicted_class.item()
            conf_score = confidence.item() * 100
            label = "TUMOR DETECTED" if class_index == 1 else "NO TUMOR"
            
            # 3. Get Visualization (Heatmap)
            heatmap_b64 = get_gradcam_overlay(selected_model, model_name, tensor, original_pil)
            
            # 4. Get Original Image (Resized for display consistency)
            resized_orig = original_pil.resize((128, 128))
            buff = BytesIO()
            resized_orig.save(buff, format="JPEG")
            original_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
            
            results.append({
                "filename": file.filename,
                "label": label,
                "heatmap_image": heatmap_b64,
                "original_image": original_b64, # Sending this back now
                "model_used": model_name
            })
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)