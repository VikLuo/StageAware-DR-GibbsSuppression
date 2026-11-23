# Pathology-Stage-Aware Selective Enhancement for DR

This repository contains the inference code for a pathology-stage-aware diabetic retinopathy (DR) grading pipeline, which selectively applies Pyramid+CLAHE enhancement to early-stage fundus images based on a gating network to improve grading performance.

## Repository Structure
├── src/# Core inference code, model architecture and detailed documentation

├── demo/# Usage demo and quick start examples

└── README.md# Project overview (this file)

## Getting Started

### 1. Environment Setup
Install PyTorch and torchvision matching your GPU first, then install dependencies:
```bash`
pip install -r src/requirements.txt

### 2. Model Weights

Model checkpoints (`.pth` files) are **not included** in this repository. They will be available for download on Zenodo after publication.
After downloading, place the weight files in the `src/checkpoints/` directory.

### 3. Run Inference

Navigate to the `src` directory and run the inference script:
```bash`
cd src
python predict.py \
  --mode two_stage \
  --original-dir ../data/original \
  --enhanced-dir ../data/pyramid \
  --stage1-weights checkpoints/stage1_gate.pth \
  --stage2-weights checkpoints/stage2_grade.pth \
  --output ../outputs/predictions.csv


For detailed parameter descriptions and baseline modes, please refer to **[src/README.md](./src/README.md)**.

## Citation

Pathology Stage Aware Selective Enhancement and Gibbs Artifact Control for Diabetic Retinopathy Fundus Images. Manuscript in preparation.

## Disclaimer

This repository is a reconstructed inference implementation for research reference. It does not guarantee reproduction of the metrics reported in the manuscript.
