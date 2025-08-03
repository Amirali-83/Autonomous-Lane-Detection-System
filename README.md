# Autonomous Lane Detection System 🚗🛣️

## 📌 Overview
This project implements an **Autonomous Lane Detection System** using the [TuSimple Dataset](https://www.kaggle.com/datasets/manideep1108/tusimple) and a **UNet segmentation model**.

The goal is to detect lane lines from highway driving footage — a core component in self-driving car technology.

---

## 📂 Dataset
- **Dataset**: TuSimple lane detection dataset  
- **Images**: ~6,400 training images, 2,782 test images  
- **Resolution**: 1280×720 (resized to 256×512 for training)

---

## ⚙️ Model
- **Architecture**: UNet with ResNet34 backbone  
- **Loss Function**: Dice Loss (binary)  
- **Optimizer**: Adam (lr=1e-4)  
- **Training**: 30 epochs on Kaggle GPU (Tesla T4/P100)

---

## 📊 Results
- **Mean IoU**: ~0.53 on validation set  
- Visual predictions show correct lane detection on both train and test sets.

---


