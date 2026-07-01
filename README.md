# Isobah2026
"SpectraVision: Deep learning pipeline for multispectral satellite imagery synthesis and temporal gap-filling. Developed for ISRO Hackathon 2026."
SpectraVision: AI-Driven Multispectral Satellite Imagery Synthesis

SpectraVision is a custom deep-learning pipeline designed for Bharaitya Antariksh Hackathon 2026. This project solves the persistent issue of satellite data gaps (due to cloud cover or revisit rates) by generating high-fidelity predicted spectral frames using a time-series triplet approach.

🚀 Architecture Overview

Our model, SpectraFlow-Net, utilizes a specialized architecture for multispectral reconstruction.

graph LR
    subgraph Encoder
        B1[RGB Residual]
        B2[Red-Edge Residual]
        B3[NIR Residual]
        B4[SWIR Residual]
    end
    subgraph Attention
        MHA[Multi-Head Attention]
    end
    subgraph Decoder
        D1[Conv2D 128]
        D2[Conv2D 64]
    end
    



🛠 Key Features

10-Band Reconstruction: Processes Sentinel-2 harmonized SR data, maintaining spectral consistency across NIR, Red-Edge, and SWIR bands.

Temporal Cross-Attention: Uses a custom Multi-Head Attention layer to fuse spatial features from past and future frames to predict the temporal gap.

Physics-Aware Validation: We don't just rely on visual intuition. Our training pipeline utilizes scientific metrics to ensure the generated imagery is suitable for environmental indices like NDVI and NDWI.

📊 Validation Metrics

Our model performance is evaluated using three industry-standard quantitative metrics:

Metric

Purpose

Importance

SSIM

Structural Similarity Index

Ensures farm boundaries and road networks retain their structure.

PSNR

Peak Signal-to-Noise Ratio

Measures pixel-level reconstruction accuracy to reduce noise.

FSIM

Feature Similarity Index

Preserves edge information for precise agricultural mapping.

🚀 Quick Start

Prerequisites:
Ensure you have Python 3.10+ and TensorFlow installed:

pip install tensorflow numpy matplotlib scikit-image



Data Acquisition:
Use our GEE script to download triplets:

python download_sentinel2_triplet.py



Inference:
Run the visualization script to generate a prediction and calculate metrics:

python visualize_prediction.py



Developed for ISRO Hackathon 2026 | Team SpectraVision
