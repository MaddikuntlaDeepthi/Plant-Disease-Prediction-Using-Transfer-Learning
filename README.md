# 🌱 Plant Disease Prediction Using Transfer Learning

## 📌 Project Overview
This project uses Transfer Learning (MobileNetV2) to classify plant leaf diseases from images.

The model was trained using the PlantVillage dataset from Kaggle.

---

## 🚀 Features
- Transfer Learning (MobileNetV2)
- Fine-Tuning
- Data Augmentation
- Early Stopping
- ReduceLROnPlateau
- Confusion Matrix
- Classification Report
- Real Image Prediction

---

## 🗂 Dataset
Dataset: PlantVillage  
Source: https://www.kaggle.com/datasets/emmarex/plantdisease

---

## 🏗 Model Architecture
- MobileNetV2 (Pretrained on ImageNet)
- Global Average Pooling
- Dense Layer (256 units)
- Dropout (0.5)
- Output Softmax Layer

---

## 📊 Results
- Validation Accuracy: ~95%
- Successfully detects diseases such as:
  - Bacterial Spot
  - Early Blight
  - Late Blight
  - Healthy Leaves

---

## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- Matplotlib
- Scikit-Learn

---

## ▶ How to Run
1. Clone repository
2. Install requirements:
   pip install -r requirements.txt
3. Run notebook in Jupyter/Colab

---

## 📌 Future Improvements
- Deploy using Streamlit
- Add Grad-CAM visualization
- Convert to Mobile App

---

## 👩‍💻 Author
Deepthi Maddikuntla
