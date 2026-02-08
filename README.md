# 🩺 Breast Cancer Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98.5%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

A **deep learning-based medical imaging system** that uses **VGG16 Transfer Learning** to classify **breast ultrasound images** as **Benign** or **Malignant** with **98.5% accuracy**.  
The project includes a **Gradio-based web interface** and **automated PDF report generation**, enabling real-time clinical decision support.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Model Performance](#-model-performance)
- [System Workflow](#-system-workflow)

---

## 🔍 Project Overview

Breast cancer is one of the most common and life-threatening diseases among women worldwide.  
Early and accurate diagnosis significantly increases survival rates.

This project leverages **transfer learning** to build a **highly accurate, fast, and reliable** breast cancer classification system using **ultrasound images**, making it suitable for **clinical assistance and academic research**.

---

## ✨ Key Features

### 🔁 Transfer Learning
- Uses **VGG16 pre-trained on ImageNet**
- Training completes in **under 5 minutes**

### 📊 High Accuracy
- Achieves **98.5% accuracy**
- Outperforms **ResNet50** and **DenseNet121**

### 🧑‍⚕️ Clinical Support Tools
- **Gradio Web Interface** for easy image upload
- **Automated PDF Diagnostic Report** using ReportLab

### ⚡ Fast Inference
- Predicts results in **~0.5 seconds per image**

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|------|---------|-----------|--------|----------|
| **VGG16** | **98.5%** | 0.91 | 0.89 | 0.90 |
| ResNet50 | 96.1% | 0.95 | 0.94 | 0.94 |
| DenseNet121 | 97.3% | 0.97 | 0.96 | 0.96 |

> 📝 **Note:**  
> All models were trained for **10 epochs** using an **80:20 train-test split**.

---

## 🧠 System Workflow

```text
Image Upload
     ↓
Image Preprocessing
     ↓
VGG16 Feature Extraction
     ↓
Custom Classification Head
     ↓
Prediction & Probability Score
     ↓
PDF Diagnostic Report Generation

---

🚀 How to Run (Quick Steps)

1. Install dependencies
2. Train model using `train.py`
3. Run app using `app.py`
4. Upload image and get result
