# deploy_brain_mri.py

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import gradio as gr

from model_brain import BrainTumorDeepConvNet  # or BrainTumorConvNet

# Same data_dir and transform as in training
data_dir = "data/brain_mri"

transform = T.Compose([
    T.Resize((128, 128)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
])

# Get class names (so we can map indices back to labels)
full_dataset = ImageFolder(root=data_dir, transform=transform)
class_idx_to_name = {v: k for k, v in full_dataset.class_to_idx.items()}
num_classes = len(class_idx_to_name)

# Load model and checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"

model = BrainTumorDeepConvNet(num_classes=num_classes)
model_path = "best_brain_tumor_model.ckpt"  # or use lightning_logs checkpoint

# If you saved a Lightning checkpoint through Trainer, you can do:
# from lightning.pytorch import Trainer
# model = BrainTumorDeepConvNet.load_from_checkpoint(model_path, num_classes=num_classes)

# For simplicity, if you just used state_dict:
# model.load_state_dict(torch.load(model_path, map_location=device))

model.to(device)
model.eval()

def predict(image: Image.Image):
    # apply same transform as training
    x = transform(image).unsqueeze(0).to(device)  # [1,1,128,128]
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()

    out = {class_idx_to_name[i]: float(probs[i]) for i in range(num_classes)}
    return out


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=num_classes),
    title="Brain MRI Tumor Classifier",
    description="Upload a brain MRI slice and get a prediction: tumor vs no-tumor."
)

if __name__ == "__main__":
    demo.launch()
