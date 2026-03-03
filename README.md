# Land Cover Segmentation & Web App

## 🌍 Overview
This project implements a semantic segmentation model to classify and map different types of land cover from satellite and aerial imagery. It includes a robust deep learning pipeline for training and a Flask-based web application for easy inference and visualization of terrain distribution.

## 📊 Dataset
* **Source:** Open-source satellite imagery sourced from Kaggle.
* **Size:** ~2.8 GB of image data.
* **Classes:** The dataset classifies pixels into 7 distinct categories:
  * Urban Land (Cyan)
  * Agriculture Land (Yellow)
  * Rangeland (Purple)
  * Forest Land (Green)
  * Water (Blue)
  * Barren Land (White)
  * Unknown (Black)
* **Augmentation:** Extensive data augmentation was applied using the `albumentations` library (including CLAHE, geometric transformations, and color jitter) to improve model robustness across different lighting and texture conditions.

## 🧠 Model Architecture

This project utilizes a **U-Net++** architecture built with **PyTorch** and the `segmentation_models_pytorch` library. U-Net++ improves upon the standard U-Net by utilizing redesigned skip pathways to bridge the semantic gap between the encoder and decoder.

* **Framework:** PyTorch
* **Encoder:** EfficientNet-B3 (Pre-trained on ImageNet)
* **Loss Function:** Dice Loss (Multi-label)
* **Optimizer:** Adam (Learning Rate: 0.0001)
* **Input Resolution:** 256x256 pixels

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pateldhruvil0622/LandCover_Segmentation.git
   cd landcover_segmentation
