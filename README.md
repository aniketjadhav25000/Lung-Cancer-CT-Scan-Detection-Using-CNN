# 🫁 Lung-Cancer-CT-Scan-Detection-Using-CNN
This project is a deep learning-based application for detecting lung cancer from CT scan images using a Convolutional Neural Network (CNN). It includes a full GUI built with Tkinter, allowing users to upload CT scan images and receive predictions in real time.

## 📁 Project Structure
LungCancerDetection/
├── preprocess.py
├── cnn_model.py
├── train_cnn_model.py
├── lung_cancer_cnn_model.keras
├── app_gui.py
├── README.md
└── LungcancerDataSet/
    ├── train/
    ├── valid/
    └── test/

## ✅ Features
- Image preprocessing with real-time augmentation

- CNN model architecture for binary classification (Cancerous / Non-Cancerous)

- GUI for image upload and prediction

- Displays patient details, uploaded image, prediction result, and confidence

## 🔧 Requirements
###Make sure the following packages are installed:
- pip install tensorflow pillow numpy

## 🧠 How to Train the Model
### Place your dataset under LungcancerDataSet/Data/ with the following structure:

Data/
├── train/
│   ├── Cancerous/
│   └── Non-Cancerous/
├── valid/
│   ├── Cancerous/
│   └── Non-Cancerous/
└── test/
    ├── Cancerous/
    └── Non-Cancerous/
