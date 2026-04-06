# Anomaly Detection in Industrial Images Using Neural Networks

U-Net-based autoencoder for detecting pixel-level surface anomalies in industrial product images. Trained on the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) — a standard benchmark for industrial inspection.

## Problem
Manual visual inspection on manufacturing lines is costly, slow, and inconsistent. This project automates defect detection using unsupervised deep learning — training only on defect-free images, then flagging deviations at inference time.

## Approach
- **Architecture:** U-Net convolutional autoencoder (encoder → bottleneck → decoder)
- **Training:** Defect-free images only — the model learns what "normal" looks like
- **Detection:** Anomalies flagged via reconstruction error thresholding
- **Categories:** Carpet and Leather (MVTec AD)
- **Pre-processing:** Resizing, normalization, grayscale conversion

## Results

| Category | Val Loss | Overfitting Gap | Defect Accuracy |
|----------|----------|-----------------|-----------------|
| Leather  | ~0.05    | < 5%            | 90%+            |
| Carpet   | ~0.07    | < 5%            | 90%+            |

- Fast convergence within 5 epochs
- Stable generalization across 50 epochs
- Successfully segments cuts, holes, scratches, and contamination

## Stack
`Python` `TensorFlow/Keras` `OpenCV` `NumPy` `Matplotlib` `Google Cloud Storage`

## Limitations & Future Work
- Thresholding is static — doesn't adapt to defect variability
- Model detects anomalous regions but doesn't classify defect types yet
- Future: Variational Autoencoders (VAE), SHAP/LIME explainability, adaptive thresholding

## Team
Boston University — BA865: Neural Network Modelling (May 2025)  
Aparna Kalla · Nilay Jaini · Sai Nruthya Vaka
