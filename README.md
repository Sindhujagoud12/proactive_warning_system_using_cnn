#  Proactive Malware Detection System Using CNN

A deep learning-based **Proactive Malware Detection System** that uses a **Convolutional Neural Network (CNN)** to analyze uploaded images and identify potential malware classes.

The application provides a simple **Streamlit web interface** where users can upload an image, receive a predicted malware class, and get a warning based on the model's confidence.

---

##  Project Overview

Malware can pose serious security risks to computer systems and users.

This project demonstrates how **Deep Learning and Computer Vision techniques** can be used to build a proactive malware detection system.

The trained CNN model analyzes an uploaded image and predicts its corresponding class. The application then displays the prediction and confidence score and provides a warning when the model's confidence exceeds a predefined threshold.

###  System Workflow

```text
User Uploads Image
        ↓
Image Preprocessing
        ↓
Resize Image to 224 × 224
        ↓
Normalize Pixel Values
        ↓
CNN Model
        ↓
Class Prediction
        ↓
Confidence Score
        ↓
Proactive Warning
