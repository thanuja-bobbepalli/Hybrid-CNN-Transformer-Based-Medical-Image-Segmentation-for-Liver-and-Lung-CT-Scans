# Hybrid CNN–Transformer Based Medical Image Segmentation  
### Liver and Lung CT Scan Segmentation

## Overview

This repository presents a research implementation of a Hybrid CNN–Transformer architecture for medical image segmentation, evaluated on liver and lung CT datasets.

The work is based on a faithful reimplementation and adaptation of:

SSNet: A Novel Transformer and CNN Hybrid Network for Remote Sensing Semantic Segmentation  
IEEE Access, 2024  
https://ieeexplore.ieee.org/document/10412345

The architecture integrates:
- Convolutional Neural Networks (CNNs) for local feature extraction
- Transformer-based encoders (MiT backbones) for global contextual modeling

The model was trained from scratch and evaluated under limited medical data conditions.

## Key Contributions

- Faithful reimplementation of an SSNet-style hybrid architecture
- Adaptation from remote sensing to medical CT segmentation
- Training from scratch without pretrained weights
- Evaluation across MiT backbone variants (B0, B3, B5)
- Comparative benchmarking with state-of-the-art medical segmentation models
- Achieved Dice score up to 96.03%

## Architecture Overview

The framework consists of:

Hybrid Encoder
- Transformer branch (Mix Vision Transformer – MiT)
- CNN branch for local boundary preservation
- Feature Fusion Module (FFM)
- Feature Injection Module (FIM)

Decoder
- CNN-based progressive upsampling
- Skip connections
- Atrous Spatial Pyramid Pooling (ASPP)
- Final binary segmentation output
![Model _arthitecture](https://github.com/user-attachments/assets/7be40f5b-6b03-4d67-b893-6bd562817792)
## Datasets

Liver Segmentation  
Medical Segmentation Decathlon – Task 03  
https://medicaldecathlon.com/

Lung Segmentation  
COVID-19 CT Lung Dataset (Zenodo)  
https://zenodo.org/record/3757476

Preprocessing steps:
- 2D axial slice extraction
- Resized to 256×256
- Intensity normalization
- Volume-level data split (70% / 15% / 15%)

## Training Strategy

- Optimizer: Adam
- Learning rate: 1e-4
- Loss: Dice + Binary Cross Entropy
- Maximum 50 epochs
- Early stopping (patience = 5)
- Best validation model retained

## Results

### Liver Segmentation (MSD)

| Backbone | Dice (%) | HD95 (mm) |
|----------|----------|-----------|
| MiT-B0   | 95.94    | 7.54      |
| MiT-B3   | **96.03** | **7.49**  |
| MiT-B5   | 95.13    | 10.50     |

###  Lung Segmentation (COVID-19 CT)

| Model | Dice (%) | HD95 (mm) |
|-------|----------|-----------|
| SSNet (Ours) | **96.03** | **3.65** |

### Test Image Results 
1. For Liver Dataset
   <img width="2950" height="5984" alt="test_samples_visualization (1)" src="https://github.com/user-attachments/assets/a87096c3-7a3c-4389-bce1-87dfc652ebed" />
   
2. For Lung Dataset
   
   <img width="432" height="562" alt="image" src="https://github.com/user-attachments/assets/88157a9a-11fb-491c-af08-b7fa055b9803" />

MiT-B3 provided the best balance between accuracy and computational efficiency.

The proposed hybrid model achieved competitive or superior performance compared to nnUNet, Swin UNETR, TransUNet, UNetFormer, and 3D U-Net.

## Conference Paper

**Hybrid CNN–Transformer Based Medical Image Segmentation for Liver and Lung CT Scans**

📄 Full Paper (PDF):  
<p>
<strong>Hybrid CNN–Transformer Based Medical Image Segmentation for Liver and Lung CT Scans</strong><br>
<a href="https://raw.githubusercontent.com/thanuja-bobbepalli/Hybrid-CNN-Transformer-Based-Medical-Image-Segmentation-for-Liver-and-Lung-CT-Scans/main/Hybrid_CNN_Transformer_Based_Medical_Image_Segmentation_for_Liver_and_Lung_CT_Scans.pdf" target="_blank">
📄 View Full Paper (PDF)
</a>
</p>

This repository documents the methodology, architectural design, and experimental findings presented at the conference. The implementation is a faithful adaptation of an SSNet-style hybrid CNN–Transformer architecture for medical CT image segmentation.

## Future Work

- Multi-organ segmentation
- 3D volumetric modeling
- Self-supervised or pretrained initialization
- Extension to additional anatomical structures


## Contributors

- [Thanuja Bobbepalli](https://github.com/thanuja-bobbepalli)
- [Bala Shankar Tataji Ommi](linkedin.com/in/bala-shankar-tataji)
  
Note: Source code is not publicly shared. This repository documents the research methodology and experimental results.
