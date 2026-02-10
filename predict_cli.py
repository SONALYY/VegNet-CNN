import argparse, json, torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

def load_model(weights_path, num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="artifacts/vegnet_resnet18.pth")
    p.add_argument("--labels", required=True, help="artifacts/labels.json")
    p.add_argument("--image", required=True, help="Path to an image to classify")
    p.add_argument("--img_size", type=int, default=224)
    args = p.parse_args()

    with open(args.labels, "r") as f:
        class_names = json.load(f)["class_names"]

    tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfms(img).unsqueeze(0)

    model = load_model(args.weights, len(class_names))
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0)

    top = probs.argmax().item()
    print(f"Predicted: {class_names[top]} (confidence={probs[top].item():.3f})\n")
    for i, c in enumerate(class_names):
        print(f"{c:7s}: {probs[i].item():.3f}")

if __name__ == "__main__":
    main()
