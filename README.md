# Interpretable Solar Panel Defect Detection via Neuro-Fuzzy XAI

[![Dataset](https://img.shields.io/badge/Dataset-ELPV-green.svg)](#)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **Official repository for the paper:** *"Interpretable Solar Panel Defect Detection via Fuzzy Rule Extraction from Deep Learning Architectures"*

## Overview
Deep learning excels at detecting photovoltaic (PV) defects, but solar farm operators cannot trust "black-box" models for critical infrastructure maintenance. This repository introduces a novel **Hybrid Neuro-Fuzzy Architecture** that bridges high-performance Deep Learning (Swin Transformers, ConvNeXt) with human-readable **Explainable AI (XAI)**.

Instead of relying on uncalibrated probabilities, this framework extracts the latent representations from state-of-the-art vision models and translates them into **explicit Fuzzy IF-THEN rules** to classify the severity of solar cell defects (Healthy, Mild, Severe).

###Key Features
* **State-of-the-Art Backbones:** Comparative analysis of 5 architectures: ResNet50, EfficientNet-B0, ViT-Tiny, ConvNeXt-Tiny, and Swin-Tiny.
* **Fuzzy Rule Extraction:** Translates complex mathematical latent vectors into transparent, linguistic diagnostic rules.
* **Explainable AI (XAI):** Validated using Grad-CAM attention heatmaps and t-SNE feature space clustering.
* **Optimized for ELPV:** Trained and evaluated on the standard Electroluminescence Photovoltaic (ELPV) benchmark dataset.

---

## Proposed Architecture

![Architecture Pipeline](docs/Figure_1.png)
> **Macro-Architecture Pipeline:** Input images (224x224) pass through a dual-path backbone (Swin-Tiny + ConvNeXt-Tiny). Extracted latent vectors undergo Pearson Correlation Analysis to generate the Fuzzy Inference System, yielding transparent severity rules and Grad-CAM localization.

---

## Experimental Results

Our experiments prove that modern hierarchical transformers and modern CNNs learn significantly better representations for fuzzy rule extraction than traditional CNNs.

### 1. Classification Performance & Rule Quality
| Model Architecture | Accuracy | Key Feature Extracted | Feature Correlation | Extracted Fuzzy Rule (Severity Logic) |
| :--- | :---: | :---: | :---: | :--- |
| **Swin-Tiny (Winner)** | **80.96%** | `f101` | **0.78** | **IF** `f101` is LOW (< 1.44) **THEN** MILD. **ELSE** SEVERE. |
| **ConvNeXt-Tiny** | 80.71% | `f487` | 0.82 | **IF** `f487` is HIGH (> -19.18) **THEN** MILD. **ELSE** SEVERE. |
| **ViT-Tiny** | 79.44% | - | - | *(Global grid attention, lacking local crack precision)* |
| **EfficientNet-B0** | 74.62% | `f199` | 0.42 | **IF** `f199` is LOW (< 0.18) **THEN** MILD. **ELSE** SEVERE. |
| **ResNet50 (Baseline)** | 73.60% | `f1555` | 0.64 | **IF** `f1555` is LOW (< 0.24) **THEN** MILD. **ELSE** SEVERE. |

### 2. Visual Explainability (Grad-CAM)
* **ResNet50:** Hyper-focused on localized cracks, failing to contextualize mild defects.
* **Swin-Tiny:** Successfully highlights both the specific defect and the surrounding solar cell grid context, justifying its superior classification of ambiguous "Mild" defects.

---

## Installation and Usage

### Prerequisites
Clone the repository and install the required dependencies:
```bash
- git clone [https://github.com/YourUsername/NeuroFuzzy-Solar-Defect.git](https://github.com/YourUsername/NeuroFuzzy-Solar-Defect.git)
- download the dataset
- run the code given one by one 
```

## Citation
If you find this code or our fuzzy rule extraction methodology useful for your research, please consider citing our paper:
```
@inproceedings{chammar2026interpretable,
  title={Interpretable Solar Panel Defect Detection via Fuzzy Rule Extraction from Deep Learning Architectures},
  author={Chammar, Aman and God, Latchan},
  booktitle={2026 International Conference on Sustainable AI for Social Impact and Global Development (SASIGD)},
  year={2026},
  organization={IEEE}
}
```
Contact
For technical questions, please open an issue or reach out via email [latchanchhetri19@gmail.com]
