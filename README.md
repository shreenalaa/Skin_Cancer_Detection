:

🩺 Skin Cancer (Melanoma) Detection System using Deep Learning (CNN)

An end-to-end Deep Learning project for detecting skin cancer (melanoma) from medical images using a Convolutional Neural Network (CNN) built with TensorFlow / Keras.
The system classifies skin lesion images into Benign and Malignant categories and provides a complete AI pipeline from data preprocessing to deployment-ready prediction.

📌 Project Overview

This project aims to:

Perform binary classification of skin lesion images:

✅ Benign (Non-Cancer)

❌ Malignant (Cancer)

Build a CNN model from scratch

Apply data augmentation for better generalization

Train and validate the model

Evaluate performance using metrics and visualizations

Save and reload trained models

Implement a prediction system for new images

🧠 AI Pipeline

Dataset loading

Directory-based data generation

Data augmentation

Image normalization

Train/Test data preparation

CNN model architecture design

Model compilation

Training with callbacks

Performance evaluation

Visualization (accuracy, confusion matrix)

Model saving

Model loading

Inference system (prediction pipeline)

📂 Dataset Structure
dataset/
│
├── train/
│   ├── benign/
│   ├── malignant/
│
└── test/
    ├── benign/
    ├── malignant/

🏷️ Classification Labels
Class	Label
Benign	0
Malignant	1
⚙️ Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Seaborn

Scikit-learn

OpenCV

Kaggle Environment

⚙️ Dependencies
pip install tensorflow numpy matplotlib seaborn scikit-learn opencv-python

🖼️ Image Processing

Resize images to 224 × 224

Normalize pixel values (1./255)

Apply data augmentation:

Rotation

Zoom

Width/Height shift

Shear

Horizontal flip

Batch loading using ImageDataGenerator

🧠 CNN Architecture

Input: (224, 224, 3)

Feature Extraction

Conv2D(32) + ReLU

MaxPooling2D(2×2)

Conv2D(64) + ReLU

MaxPooling2D(2×2)

Conv2D(128) + ReLU

MaxPooling2D(2×2)

Classification Head

Flatten

Dense(512) + ReLU

Dropout(0.5)

Dense(1) + Sigmoid

⚡ Model Configuration

Optimizer: Adam

Loss Function: Binary Crossentropy

Metric: Accuracy

📈 Training Strategy

Epochs: 30

Batch Size: 32

Callbacks:

EarlyStopping (patience=5)

ReduceLROnPlateau (adaptive learning rate)

📊 Evaluation Methods

Training Accuracy

Validation Accuracy

Accuracy vs Epoch plots

Classification Report

Confusion Matrix (Seaborn Heatmap)

🔮 Prediction System
Features:

Load trained model (.h5)

Load image from path

Resize and normalize image

Predict class probability

Binary classification output

Visual prediction display

Output:
Predicted: Malignant


OR

Predicted: Benign

💾 Model Saving
skin_cancer_cnn.h5


The trained model can be reloaded for inference without retraining.

🚀 How to Run
Training:
python train_model.py

Prediction:
predict_skin_cancer("image_path.jpg", model)

🎯 Applications

Medical diagnosis assistance

Clinical decision support systems

Healthcare AI platforms

Dermatology screening tools

Smart hospitals

Telemedicine systems

Medical research

AI-powered diagnostic tools

🔮 Future Enhancements

Transfer Learning (ResNet, EfficientNet, MobileNet)

Multi-class classification (multiple skin diseases)

Real-time detection

Web deployment (FastAPI / Flask)

Mobile app integration

API services

Model quantization

Edge AI deployment

Explainable AI (Grad-CAM visualization)

Federated learning for medical privacy

👩‍💻 Author

Shereen Alaa
Machine Learning Engineer

GitHub: https://github.com/shreenalaa

LinkedIn: https://www.linkedin.com/in/shreen-alaa/
