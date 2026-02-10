# VegNet – Vegetable Stage Classifier 🌱

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Status](https://img.shields.io/badge/Status-Working-success)

VegNet is a CNN-based image classification system that predicts the **stage and condition of vegetables** from images.  
It classifies vegetables into **five categories** and displays **confidence scores** using an interactive **Gradio web interface**.

---

## ✨ Features

- Classifies vegetables into 5 stages:
  - Unripe
  - Ripe
  - Old
  - Dried
  - Damaged
- Shows prediction confidence (%) for each class
- Simple drag-and-drop web interface
- Command-line prediction support
- Built using PyTorch (ResNet-18)

---

## 🎥 Demo

![VegNet Demo](demo.gif)

> Upload a vegetable image → click **Submit** → get predictions with confidence scores.


## 🧠 Model Overview

- Architecture: **ResNet-18**
- Framework: **PyTorch**
- Input size: **224 × 224**
- Output: **Softmax probabilities**
- Number of classes: **5**

Trained weights:

---

## 📂 Project Structure

VegNet-CNN/
├── vegnet_gradio_app.py # Gradio web app
├── train_pytorch.py # Model training script
├── predict_cli.py # CLI prediction tool
├── requirements.txt
├── artifacts/
│ ├── vegnet_resnet18.pth
│ ├── vegnet_resnet18_last.pth
│ └── labels.json
├── Data/
│ ├── Damaged/
│ ├── Dried/
│ ├── Old/
│ ├── Ripe/
│ └── Unripe/
└── README.md

---

## ⚙️ Installation

Install dependencies:
```bash
pip install -r requirements.txt
▶️ Run the Web App
python vegnet_gradio_app.py


Open in browser:

http://127.0.0.1:7860

🧪 CLI Prediction
python predict_cli.py \
  --weights artifacts/vegnet_resnet18.pth \
  --labels artifacts/labels.json \
  --image path/to/image.jpg


Example output:

Predicted: Damaged (confidence=0.96)

Damaged : 0.96
Old     : 0.03
Dried   : 0.02
Ripe    : 0.00
Unripe  : 0.00

📦 Requirements

Python 3.9+

torch

torchvision

pillow

gradio

## 📜 License

This project is licensed under the **Apache License 2.0**.

You are free to:
- Use the software for any purpose
- Modify and distribute the code
- Use it in private or commercial projects

Provided that you:
- Include a copy of the license
- State any significant changes made

See the [LICENSE](LICENSE) file for details.

🤝 Contributions

Pull requests and suggestions are welcome.
Feel free to fork the repository and experiment.

⭐ Acknowledgements

PyTorch

Gradio

torchvision



