# VegNet – Vegetable Stage Classification System 🌱

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Status](https://img.shields.io/badge/Status-Stable-success)

VegNet is a convolutional neural network (CNN)–based image classification system designed to identify the **stage and condition of vegetables** from images.  
The system provides both a **web-based interface** and a **command-line interface** for flexible usage in research, experimentation, and deployment scenarios.

---

## Features

- Classifies vegetables into five stages:
  - Unripe
  - Ripe
  - Old
  - Dried
  - Damaged
- Displays confidence scores for each predicted class
- Interactive web interface built using Gradio
- Command-line interface for scripted or batch predictions
- Built using PyTorch with a ResNet-18 backbone

---

## Demo

The Gradio web interface allows users to upload a vegetable image and receive predictions along with confidence scores.

*(A demo GIF can be added as `demo.gif` in the repository root.)*

---

## Model Overview

- **Architecture:** ResNet-18  
- **Framework:** PyTorch  
- **Input Resolution:** 224 × 224  
- **Output:** Softmax probability distribution  
- **Number of Classes:** 5  

**Trained Model Weights:**
artifacts/vegnet_resnet18.pth


---

## Model Card

### Model Name
VegNet (Vegetable Stage Classifier)

### Intended Use
The model is intended for:
- Automated classification of vegetable stages
- Computer vision research and experimentation
- Educational demonstrations of CNN-based systems
- Decision-support applications

The model should not be used as the sole decision-making system in critical or safety-sensitive environments.

### Supported Classes
- Unripe
- Ripe
- Old
- Dried
- Damaged

### Training Data
- Image dataset organized into five class-specific directories
- Images represent different visual stages of vegetables
- Dataset size and diversity are limited and may not cover all real-world scenarios

### Limitations
- Performance may degrade on unseen vegetable types
- Sensitive to lighting, image quality, and camera conditions
- Predictions are probabilistic and not guaranteed to be correct

### Ethical Considerations
- Predictions are based solely on visual appearance
- Dataset bias may affect output reliability
- Users are responsible for validating predictions before real-world use

---

## Project Structure

VegNet-CNN/
├── vegnet_gradio_app.py # Gradio-based web application
├── train_pytorch.py # Model training script
├── predict_cli.py # Command-line prediction tool
├── requirements.txt # Python dependencies
├── artifacts/ # Model artifacts
│ ├── vegnet_resnet18.pth
│ ├── vegnet_resnet18_last.pth
│ └── labels.json
├── Data/ # Dataset organized by class
│ ├── Damaged/
│ ├── Dried/
│ ├── Old/
│ ├── Ripe/
│ └── Unripe/
└── README.md


---

## Installation

Install all required dependencies using:

bash
pip install -r requirements.txt

## Running the Web Application

Start the Gradio web interface:
python vegnet_gradio_app.py
Once the server starts, open the following URL in your browser:
http://127.0.0.1:7860

# Upload a vegetable image to view predictions and confidence scores.


---

## Command-Line Prediction
For prediction without the web interface:

python predict_cli.py \
  --weights artifacts/vegnet_resnet18.pth \
  --labels artifacts/labels.json \
  --image path/to/image.jpg

# Sample Output:

Predicted: Damaged (confidence=0.96)

Damaged : 0.96
Old     : 0.03
Dried   : 0.02
Ripe    : 0.00
Unripe  : 0.00

## Requirements
- Python 3.9 or higher

- torch

- torchvision

- pillow

- gradio


--- 

## License
This project is licensed under the Apache License 2.0.

You are permitted to:

- Use the software for any purpose

- Modify and redistribute the source code

- Use the project in private or commercial environments

Provided that:

- A copy of the license is included

- Any significant modifications are clearly stated

- See the LICENSE file for full details.


--- 

## Contributing
Contributions are welcome.

To contribute:

- Fork the repository

- Create a feature branch

- Submit a pull request with a clear description of changes


--- 

## Acknowledgements
- PyTorch – Deep learning framework

- torchvision – Image models and transformations

- Gradio – Interactive web interface for machine learning applications
