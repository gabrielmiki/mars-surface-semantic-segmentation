# 🔴 Mars Surface Semantic Segmentation

> **AN2DL Homework 2 — 2024/2025**  
> Pixel-wise classification of Mars surface images using deep learning, developed as part of the Artificial Neural Networks and Deep Learning course at Politecnico di Milano.

---

## 📋 Overview

This project tackles **semantic segmentation** of Mars surface imagery. Given grayscale images of the Martian terrain, the goal is to classify every pixel into one of **5 semantic classes** (labels 0–4). Models were evaluated using **Mean Intersection over Union (MeanIoU)** on a held-out Kaggle leaderboard, with the background class (0) excluded from the metric.

The project follows an iterative experimental workflow, progressively building from a minimal baseline to increasingly complex architectures.

---

## 📁 Repository Structure

```
.
├── 00_-_Data_Analysis.ipynb                          # EDA, class imbalance, oversampling strategies
├── 01_-_Minimal_Working_Example.ipynb                # Baseline 1×1 Conv model + submission pipeline
├── 02_-_Addition_of_Basic_Autoencoder.ipynb          # Basic U-Net encoder–decoder
├── 03_-_Double_Basic_Autoencoder.ipynb               # Dual-path U-Net with data augmentation
├── 04_Basic_Unet_Model_plus_Inception_Senet_Residual.ipynb  # U-Net with advanced blocks
├── 05_Previous_Models_Dense_Layers.json              # Dense layer experiments and ablation variants
└── README.md
```

---

## 🗃️ Dataset

- **File:** `mars_for_students.npz`
- **Training set:** Grayscale images with corresponding pixel-level label masks
- **Test set:** Unlabelled images for Kaggle submission
- **Classes:** 5 (labels 0–4), where class 0 is background
- **Preprocessing:** pixel values normalised to [0, 1]; a channel dimension added for compatibility with Conv2D layers

### Class Imbalance

A key challenge was severe **class imbalance**, particularly for class 4, which appeared far less frequently than others. Three oversampling strategies were explored:

1. **Pure reuse** — repeating minority-class samples within each batch
2. **Image generation** — using `ImageDataGenerator` (rotations, flips, shifts) to synthesise new samples
3. **Mixed approach** — combining generated and reused samples, saved as `oversampled_normalized_train_mars_for_students.npz`

---

## 🧪 Experimental Notebooks

### `00` — Data Analysis
Exploratory analysis of the dataset: class distribution plots, per-class example visualisation, and evaluation of different oversampling strategies. This informed all subsequent modelling decisions.

### `01` — Minimal Working Example
A **1×1 Convolution** baseline model (single Conv2D layer with softmax) trained for 1 epoch. Its purpose was to validate the end-to-end pipeline — from data loading to Kaggle CSV submission — before committing to more complex architectures.

### `02` — Basic U-Net (Autoencoder)
A standard **U-Net encoder–decoder** with skip connections. The encoder progressively halves spatial resolution using MaxPooling, while the decoder upsamples back to the original resolution using `Conv2DTranspose`. Trained with early stopping (patience = 30) and a 300-sample validation split.

Key elements:
- Custom `unet_block` with stacked Conv2D + BatchNorm + ReLU layers
- No augmentation in this version
- Adam optimiser, learning rate 1e-3

### `03` — Double U-Net (Double Autoencoder)
An extension of the basic U-Net using **two parallel encoder paths** whose features are concatenated before the bottleneck and decoder. Data augmentation (horizontal and vertical flips) was introduced via a custom `tf.data` pipeline to improve generalisation.

### `04` — U-Net + Inception / SENet / Residual Blocks
The most architecturally complex model, integrating three advanced building blocks into the U-Net backbone:

- **Inception block** — parallel convolutions with 1×1, 3×3, and 5×5 kernels, concatenated before BatchNorm
- **SENet (Squeeze-and-Excitation) block** — channel-wise attention applied after the inception step to recalibrate feature map importance
- **Residual block** — skip connections around the inception+SENet composite to ease gradient flow

The oversampled dataset was used for training, along with a class-4-aware mixed batch sampling strategy to ensure the minority class appeared consistently within each batch.

### `05` — Dense Layer Experiments
Further architecture variants adding dense (fully-connected) layers at various points, as well as ablation studies across different filter counts and residual path depths. Includes both the double U-Net and the full Inception–SENet–Residual U-Net with modified configurations.

---

## ⚙️ Technical Stack

| Component | Details |
|-----------|---------|
| Framework | TensorFlow / Keras |
| Environment | Google Colab (GPU) |
| Data pipeline | `tf.data.Dataset` with custom augmentation |
| Augmentation | Random horizontal & vertical flips |
| Optimiser | Adam (`lr = 1e-3`) |
| Loss | Sparse Categorical Crossentropy |
| Metric | MeanIoU (excluding background class) |
| Callbacks | EarlyStopping (`patience = 30`, monitor = `val_accuracy`) |
| Batch size | 64 |
| Max epochs | 1000 |

---

## 📊 Model Progression Summary

| Notebook | Architecture | Dataset | Augmentation |
|----------|-------------|---------|--------------|
| 01 | 1×1 Conv baseline | Original | None |
| 02 | Basic U-Net | Original | None |
| 03 | Dual-path U-Net | Original | Flips |
| 04 | U-Net + Inception + SENet + Residual | Oversampled | Flips + mixed batch |
| 05 | Dense variants & ablations | Oversampled | Flips + mixed batch |

---

## 🚀 Getting Started

1. **Mount Google Drive** and navigate to the project directory (all notebooks include this step for Colab).
2. **Place the dataset files** (`mars_for_students.npz` and optionally `oversampled_normalized_train_mars_for_students.npz`) in the working directory.
3. **Run notebooks in order** — starting from `00` for EDA, then proceeding sequentially through the model experiments.
4. **Generate submissions** using the `y_to_df` utility in notebook `01`, which flattens predictions into the Kaggle-compatible CSV format.

### Dependencies

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

---

## 📌 Notes

- The custom `DeepLearningLib` (`dll`) module is used across all notebooks for utility functions such as `plotImages`, `calculateClassDistribution`, and `plotDifferentClassesExamples`.
- Models are saved with timestamped filenames (`model_YYMMDD_HHMMSS.keras`) to avoid overwriting.
- The validation split is fixed at **300 samples** with `random_state=42` for reproducibility across all experiments.
