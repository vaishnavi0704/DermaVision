# 🧬 DermAI-Vision

**DermAI-Vision** is a powerful deep learning system for **automated skin lesion analysis** using a dual-architecture approach — combining **Vision Transformer (ViT)** and **U-Net** models. Built to perform **multi-task learning**, this system simultaneously classifies skin lesion types and segments affected regions with high precision.

Optimized using advanced image preprocessing techniques and trained with **5-fold stratified cross-validation**, DermAI-Vision aims to assist in early diagnosis and medical decision-making in dermatology.

---

## 🩺 Key Features

### 🧠 Dual-Architecture Design
- **ViT (Vision Transformer):** Used for **lesion classification** — captures global context with self-attention.
- **U-Net:** Used for **segmentation** — excels at capturing spatial details for lesion localization.

### ⚙️ Multi-Task Learning
- Jointly optimizes both classification and segmentation tasks.
- Reduces training time and increases generalization by sharing backbone features.

### 🧪 Advanced Image Processing
- **Hair removal, contrast enhancement, resizing**
- **Extensive data augmentation** using the `Albumentations` library:
  - Random brightness/contrast
  - Elastic transform
  - Horizontal/vertical flips
  - Rotation, shifting, cropping

### 🔁 Robust Training Strategy
- **5-Fold Stratified Cross-Validation** to ensure fair and comprehensive performance evaluation.
- Balanced learning across lesion classes like Melanoma, Nevus, and Keratosis.

---

## 📊 Use Case

DermAI-Vision can be applied in:

- **Clinical Decision Support:** Assists dermatologists in identifying and segmenting lesions.
- **Teledermatology Apps:** On-device classification and segmentation for patient self-checks.
- **Medical Imaging Research:** Transferable pipeline for other biomedical image segmentation tasks.

---

## 🛠️ Tech Stack

| Component        | Library/Tool         | Purpose                                |
|------------------|----------------------|----------------------------------------|
| Deep Learning    | TensorFlow & Keras   | Model training and evaluation          |
| Image Processing | OpenCV, Albumentations | Preprocessing and augmentation        |
| Data Handling    | NumPy, Pandas        | Loading and managing image metadata    |
| Evaluation       | Scikit-learn         | Metrics like accuracy, F1, AUC         |
| Model Types      | ViT, U-Net           | Classification and segmentation        |
| Cloud/Scale      | Google Cloud Platform (GCP) | GPU training & cloud experimentation |

---

## 📁 Folder Structure

DermAI-Vision/
├── data/ # Dataset (e.g., HAM10000)
├── preprocessing/ # Hair removal, resizing scripts
├── augmentation/ # Albumentations pipeline
├── classification/ # ViT training and evaluation
├── segmentation/ # U-Net training and mask generation
├── models/ # Saved model checkpoints
├── cross_validation/ # Stratified fold logic
├── utils/ # Metrics, plotting, callbacks
├── training_pipeline.py # Unified multi-task training script
├── config.yaml # Training parameters and paths
├── README.md
└── requirements.txt
---

## 🧪 Training Strategy

### 🌀 5-Fold Stratified Cross-Validation
- Maintains class distribution across all folds
- Results averaged across folds for stable evaluation
- Prevents overfitting and ensures generalization

### 📈 Metrics Tracked
- **Classification:** Accuracy, Precision, Recall, F1-score, AUC
- **Segmentation:** Dice Score, IoU, Pixel Accuracy

---

## 📦 Installation

```bash
git clone https://github.com/your-username/DermAI-Vision.git
cd DermAI-Vision

pip install -r requirements.txt
