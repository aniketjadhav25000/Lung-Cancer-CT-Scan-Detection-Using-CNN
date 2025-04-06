# 🫁 Lung-Cancer-CT-Scan-Detection-Using-CNN
This project is a deep learning-based application for detecting lung cancer from CT scan images using a Convolutional Neural Network (CNN). It includes a full GUI built with Tkinter, allowing users to upload CT scan images and receive predictions in real time.

## 📁 Project Structure

![image](https://github.com/user-attachments/assets/c3b51d5c-a4e0-4f58-a901-05bd2edefaa3)


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
