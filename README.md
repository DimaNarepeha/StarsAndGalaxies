# **Galaxies and Stars Recognition**

This repository contains the implementation of a convolutional neural network (CNN) model for recognizing galaxies and stars in astronomical images. The model is built using PyTorch Lightning and achieves a high classification accuracy through various techniques like data augmentation, normalization, and dropout regularization.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Results](#results)

## Overview

The goal of this project is to classify astronomical images into two categories: **galaxies** and **stars**. The model has been trained using a custom convolutional neural network with multiple layers, batch normalization, and dropout to prevent overfitting. Additionally, the model utilizes PyTorch Lightning to streamline training and evaluation.

## Model Architecture

The model consists of several convolutional layers followed by max pooling, and fully connected layers. The following key layers are used:

- **Convolution Layers**: Feature extraction from input images.
- **MaxPooling Layers**: Downsampling the feature maps.
- **Fully Connected Layers**: Classification layers.
- **Dropout**: Regularization to avoid overfitting.
- **Batch Normalization**: Improving training speed and stability.

You can check the full architecture of the model in the [Model.py](./Model.py) file.

## Dataset

The dataset used for training and testing contains astronomical images of galaxies and stars. The data is divided into the following directories:

- `data/train`: Contains the training images.
- `data/validate`: Contains the validation images.
- `data/test`: Contains the testing images.
- `data/hand_test`: Additional test images for custom evaluation.

Each directory contains two subdirectories representing the classes:
- `galaxies/`
- `stars/`

## Installation

To get started with the project, clone the repository and install the required packages.

```bash
git clone https://github.com/your-username/galaxies-stars-recognition.git
cd galaxies-stars-recognition
```
Make sure you have installed PyTorch, torchvision, and PyTorch Lightning.

## Training the Model

To train the model, use the `main.py` file. You can train the model from scratch or load a pre-trained checkpoint from the `lightning_logs/version_*` directory.

```bash
python main.py
```
## Hyperparameters
You can adjust the following hyperparameters for training:

learning_rate: 0.001
batch_size: 64
num_epochs: 200
These are defined at the start of the script and can be tuned as needed.

## Evaluating the Model
Once the model is trained, you can evaluate its performance on the test dataset. The script will load the best model checkpoint and run the evaluation.

Command:

python main.py --evaluate

You can also visualize the predictions and compare real vs. predicted labels using the built-in function display_predictions().

## Results
The model achieved high accuracy on the validation dataset. Below are some of the performance metrics:

**Validation Accuracy: 88.7%**
Test Accuracy: (add test accuracy here)
Confusion Matrix: (you can add a confusion matrix plot here)
To visualize the weight distribution of the model, the function plot_large_weights_distribution() can be used to check the distribution of large weights.
