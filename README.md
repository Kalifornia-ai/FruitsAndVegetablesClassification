# 🥦🍎 Fruits and Vegetables Classification using CNN

This mini-project implements a **Convolutional Neural Network (CNN)** using **PyTorch Lightning** to classify images of fruits and vegetables. The dataset is explored, visualized, and used to train a deep learning model capable of distinguishing between multiple food classes.

## 🧠 Objective
To build a robust image classification pipeline that can:
- Load and preprocess a labeled dataset of fruits and vegetables
- Train a CNN model for multi-class classification
- Evaluate performance with confusion matrix and accuracy metrics
- Log training progress using **Weights & Biases (W&B)**

---

## 📁 Dataset
- Dataset used: **[Fruits and Vegetables Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)**
- Total classes: Varies (based on training folder)
- Format: Folder-per-class structure
- Used transformations: normalization, resizing, augmentation (if applied)

---

## 🧰 Tools & Libraries
- **PyTorch**
- **PyTorch Lightning**
- **TorchVision**
- **Matplotlib**
- **Scikit-learn**
- **W&B (Weights & Biases)**

---

## 🏗️ Model Architecture
- CNN implemented using PyTorch Lightning
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Metrics: Accuracy, Confusion Matrix

---

