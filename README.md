# Breast-Cancer-Classification
A deep learning framework using VGG16 Transfer Learning to classify breast ultrasound images as benign or malignant with 98.5% accuracy. Features an intuitive Gradio web interface and automated PDF report generation for real-time clinical decision support. Designed for high efficiency and reliability.

# Key Features
# Transfer Learning: 
Uses pre-trained ImageNet weights to reduce training time to under 5 minutes.
# High Performance: 
Achieves superior results compared to ResNet50 and DenseNet121.
# Clinical Tools: 
Integrated Gradio UI for easy image uploads and ReportLab for generating expert diagnostic reports.
# Fast Inference: 
Processes individual images in approximately 0.5 seconds.
# Model Performance
VGG16 (Accuracy - 98.5%, Precision - 0.91, Recall - 0.89, F1-Score - 0.90)
ResNet50 (Accuracy - 96.1%, Precision - 0.95, Recall - 0.94, F1-Score - 0.94)
DenseNet121 (Accuracy - 97.3%, Precision - 0.97, Recall - 0.96, F1-Score - 0.96)
Note: (Results based on 10 epochs of training with an 80-20 dataset split )

# System Workflow
# Preprocessing: 
Resizes images to 224 X 224 and normalizes pixel values to [0, 1].
# Feature Extraction: 
Leverages 13 frozen convolutional layers from the VGG16 architecture.
# Classification: 
Processes features through a custom head with 256 neurons and a 0.4 dropout rate to prevent overfitting.
# Output: 
Generates a malignancy probability score and an automated PDF report.
# Requirements:
TensorFlow / Keras 
OpenCV
Gradio
ReportLab
NumPy 

# How to Run
1. Install dependencies
2. Train model using `train.py`
3. Run app using `app.py`
4. Upload image and get result

# Dataset
Breast Ultrasound Images Dataset (Kaggle)

