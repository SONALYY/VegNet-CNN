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

---

## 🧠 Model Overview

- Architecture: **ResNet-18**
- Framework: **PyTorch**
- Input size: **224 × 224**
- Output: **Softmax probabilities**
- Number of classes: **5**

Trained weights:
artifacts/vegnet_resnet18.pth


---
 
📁 VegNet-CNN
├── vegnet_gradio_app.py        # Gradio web application
├── train_pytorch.py            # Model training script
├── predict_cli.py              # CLI-based prediction tool
├── requirements.txt            # Project dependencies
├── artifacts/
│   ├── vegnet_resnet18.pth     # Trained model weights
│   ├── vegnet_resnet18_last.pth
│   └── labels.json             # Class labels
├── Data/
│   ├── Damaged/
│   ├── Dried/
│   ├── Old/
│   ├── Ripe/
│   └── Unripe/
└── README.md


---

## ⚙️ Installation

bash
pip install -r requirements.txt

---

▶️ Run the Web App 🚀

Start the Gradio-based web application:

python vegnet_gradio_app.py


Once the server starts, open your browser and visit:

http://127.0.0.1:7860


📷 Upload a vegetable image

📊 View predicted class with confidence percentages

---

🧪 CLI Prediction 🖥️

Use the command-line tool for quick predictions:

python predict_cli.py \
  --weights artifacts/vegnet_resnet18.pth \
  --labels artifacts/labels.json \
  --image path/to/image.jpg

📈 Sample Output
Predicted: Damaged (confidence=0.96)

Damaged : 0.96
Old     : 0.03
Dried   : 0.02
Ripe    : 0.00
Unripe  : 0.00

---

📦 Requirements 🧰

🐍 Python 3.9 or higher

🔥 torch

🖼️ torchvision

🖌️ pillow

🌐 gradio

---

📜 License ⚖️

📄 Licensed under the Apache License 2.0

✅ You are allowed to:

Use the software for any purpose

Modify and redistribute the code

Use it in private or commercial projects

⚠️ Conditions:

Include a copy of the license

Clearly mention any significant changes made

🔗 See the LICENSE
 file for complete details.

 ---

🤝 Contributing 💡

🌱 Contributions, issues, and feature requests are welcome

🔀 Fork the repository and submit a pull request

💬 Suggestions and improvements are always appreciated

---

⭐ Acknowledgements 🙌

🔥 PyTorch – Deep learning framework

🖼️ torchvision – Image models & transformations

🌐 Gradio – Interactive ML web interface


