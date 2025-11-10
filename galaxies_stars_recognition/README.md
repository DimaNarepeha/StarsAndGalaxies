# Stellar & Galactic Image Recognition (PyTorch Lightning)

**“Advanced Deep Learning Models for Stellar and Galactic Recognition from Telescope Imagery”**  
This repository contains the implementation of a convolutional neural network (CNN) model for recognizing galaxies 
and stars in astronomical images. 
The model is built using PyTorch Lightning and achieves a high classification accuracy through various 
techniques like data augmentation, normalization, and dropout regularization.
 
---
## 📑 Table of Contents
- [Highlights](#-highlights)
- [Repository Structure](#-repository-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup](#-setup)
- [Installation](#installation)
- [Quick Demo (Visual Prediction)](#-quick-demo-visual-prediction)
- [Figures](#-figures)
- [Evaluate Model](#-evaluate-model)
- [Model Overview](#-model-overview)
- [Hyperparameters](#hyperparameters)
- [Checkpoint](#-checkpoint)
- [Reproducibility](#-reproducibility)
- [Results](#results)
- [Citation](#-citation)
- [License](#-license)

## 🚀 Highlights
- Validated accuracy: **88.7%**
- Binary classification: **galaxies vs stars**
- Fully reproducible pipeline (train → evaluate → visualization)
- Works on CPU and GPU
- Stand-alone prediction script with visual output

---

## 🗂 Repository Structure

.
├─ Model.py
├─ main.py
├─ make_prediction.py           # show Pred/Actual/Conf with images
├─ make_more_figs.py             # confusion, ROC/PR, grids → thesis_figs/
├─ CustomDataset.py              # optional CSV-based dataset loader
├─ lightning_logs/               # contains checkpoints (*.ckpt)
├─ data/
│   ├─ train/
│   ├─ validate/
│   └─ test/
├─ requirements.txt
└─ README.md

Expected **ImageFolder** layout:

## Dataset

The dataset used for training and testing contains astronomical images of galaxies and stars. The data is divided into the following directories:

- `data/train`: Contains the training images.
- `data/validate`: Contains the validation images.
- `data/test`: Contains the testing images.  
- 
data/
 ├─ train/
 │   ├─ galaxies/*.jpg
 │   └─ stars/*.jpg
 ├─ validate/
 │   ├─ galaxies/*.jpg
 │   └─ stars/*.jpg
 └─ test/
     ├─ galaxies/*.jpg
     └─ stars/*.jpg
## Model Architecture

The model consists of several convolutional layers followed by max pooling, and fully connected layers. The following key layers are used:

- **Convolution Layers**: Feature extraction from input images.
- **MaxPooling Layers**: Downsampling the feature maps.
- **Fully Connected Layers**: Classification layers.
- **Dropout**: Regularization to avoid overfitting.
- **Batch Normalization**: Improving training speed and stability.

You can check the full architecture of the model in the [Model.py](./Model.py) file.

---

## ✅ Setup

## Installation

To get started with the project, clone the repository and install the required packages.

```bash
git clone https://github.com/DimaNarepeha/galaxies-stars-recognition.git
cd galaxies-stars-recognition
```
Make sure you have installed PyTorch, torchvision, and PyTorch Lightning.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python ≥ 3.10  
Torch auto-detects CUDA (GPU) if available.

---

## ▶️ Quick Demo (Visual Prediction)

Show predictions with image preview:

```bash
python make_prediction.py   --ckpt "lightning_logs/version_2/checkpoints/88.7%epoch=32-step=8250.ckpt"   --images "data/test"
```

A window will display:

```
Actual: galaxies
Predicted: galaxies
Conf: 0.97
```

---

## 📊 Figures

Produce confusion matrix, ROC/PR, grids of correct/misclassified samples:

```bash
python make_more_figs.py
```

Results saved into `thesis_figs/`.

---

## 🧪 Evaluate Model

Minimal evaluation using the trained checkpoint:

```bash
python main.py
```

---

## 🧠 Model Overview

- Input: **128 × 128 RGB**
- Backbone: 3 convolutional blocks (Conv×2 → MaxPool)
- FC head: 2048 → 2048 → 2
- Regularization: Dropout, data augmentation
- Normalization: ImageNet mean/std
- Metrics: Macro accuracy, confusion matrix, ROC/PR

Model source: `Model.py`
## Hyperparameters
You can adjust the following hyperparameters for training:

learning_rate: 0.001
batch_size: 64
num_epochs: 200
These are defined at the start of the script and can be tuned as needed.

---

## 📦 Checkpoint

Default location:

```
lightning_logs/version_2/checkpoints/88.7%epoch=32-step=8250.ckpt
```

---

## 🔁 Reproducibility

- Deterministic training
- Fixed transforms
- Fixed image size 128×128
- requirements.txt contains exact package versions

---

## Results
The model achieved high accuracy on the validation dataset. Below are some of the performance metrics:

**Validation Accuracy: 88.7%**
To visualize the weight distribution of the model, the function plot_large_weights_distribution() can be used to check the distribution of large weights.
## 📜 Citation

If using this code or results:

```
@misc{Narepekha2025StellarGalaxy,
  title  = {Advanced Deep Learning Models for Stellar and Galactic Recognition from Telescope Imagery},
  author = {Dmytro Narepekha},
  year   = {2025}
}
```

---

## 🛡 License

Academic/research use only. Check dataset licenses before redistribution.
