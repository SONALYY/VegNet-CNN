import gradio as gr
import json, torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

LABELS_JSON = "artifacts/labels.json"
WEIGHTS = "artifacts/vegnet_resnet18.pth"
IMG_SIZE = 224  # if you trained with a different size, you can change it

with open(LABELS_JSON, "r") as f:
    CLASS_NAMES = json.load(f)["class_names"]

def load_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))
    state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

MODEL = load_model()
TFMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = TFMS(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(MODEL(x), dim=1).squeeze(0).tolist()
    return {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Veg Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="VegNet Stage Classifier",
    description="Predict Unripe · Ripe · Old · Dried · Damaged"
)

if __name__ == "__main__":
    demo.launch()
