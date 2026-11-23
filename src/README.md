```markdown
# Pathology-Stage-Aware Selective Enhancement for DR: Inference Code

## Introduction
This repository provides a reconstructed, inspectable inference implementation for a pathology-stage-aware diabetic retinopathy (DR) grading pipeline. It uses a ConvNeXt-Small stage-gating model and a separate ConvNeXt-Small five-class grader. Early-stage images can be routed through a precomputed Pyramid+CLAHE enhancement branch, while other images are kept unchanged.

**Research provenance**: The original complete five-fold prediction traces and original training script are unavailable. This repository is a reconstruction based on the manuscript, available checkpoints, and preprocessing descriptions. It is **not** evidence that the reported 96.37% accuracy or 0.9497 quadratic weighted kappa has been independently reproduced. No missing per-image predictions have been synthesized.

## Model architecture
1  Original fundus RGB image
2      |  Resize to 224×224 and ImageNet normalization
3      v
4  ConvNeXt-Small [stage 1: binary routing head]
5      |  early/late routing decision, mapping confirmed via training config
6      +-- enhance --> precomputed Pyramid+CLAHE image ---+
7      +-- retain  --> original RGB image ----------------+
```

Both networks use torchvision's `convnext_small` feature extractor. For checkpoints with historical naming, the classification head is `nn.Sequential(nn.Dropout(0.3), nn.Linear(768, n_classes))`; this results in `classifier.2.1.weight` and `classifier.2.1.bias` keys. The exact dropout probability is not identifiable from stored weights. Set `--head-style linear` for a direct `nn.Linear` head if that was used in your checkpoint.

## Install
Install PyTorch and torchvision appropriate to your GPU first. Then:
```bash
python -m pip install -r requirements.txt
```

## Inputs
- Original images in one folder, for example `data/original/DR_001_R.jpg`.
- Paired, already-computed Pyramid+CLAHE images in another folder with names like `data/pyramid/DR_001_R_enhanced.png`.
- Trusted stage-1 weights from the same routing definition and stage-2 grading weights.
- The `--enhance-class` index identifying which stage-1 softmax output means *use enhanced image*. In the reconstructed training code it is class 1. **Check the original checkpoint training labels before relying on this for historical weights.**

> Do not upload patient-identifying images, non-public datasets, or model weights to a public GitHub repository without permission and applicable ethics review.

## Inference
Use only known model checkpoints and preserve the run metadata.

### Two-stage main pipeline
> Model weights (`.pth` files) are not included in this repository. They will be available for download on Zenodo after publication. Please download the checkpoint files and place them in a local directory (e.g., `checkpoints/`) before running inference.

```bash
python predict.py \
  --mode two_stage \
  --original-dir data/original \
  --enhanced-dir data/pyramid \
  --stage1-weights checkpoints/stage1_gate.pth \
  --stage2-weights checkpoints/stage2_grade.pth \
  --stage-boundary grade01 \
  --enhance-class 1 \
  --output outputs/predictions.csv

### Baseline modes
- For the raw-image baseline, use `--mode raw` and omit stage-1 and enhanced-directory parameters.
- For the global-enhancement baseline use `--mode all_enhanced` and supply the enhanced directory.

> **Caution**: The manuscript selects grade 0–1 for enhancement, whereas a historical training log describes a grade 0–2 versus grade 3–4 gate. A checkpoint trained for one boundary is not interchangeable with the other. `--stage-boundary` records the chosen interpretation; it does not change the weights' learned labels.

## Test
Run unit tests to verify checkpoint structure and input specifications:
```bash
python -m unittest test_components.py -v
```

## File Description
| File | Purpose |
|------|---------|
| `__init__.py` | Package initialization file |
| `models.py` | ConvNeXt-Small gate and grader architecture, strict weight loading |
| `preprocessing.py` | Input standardization and enhanced-image pairing logic |
| `predict.py` | End-to-end two-stage or baseline inference script |
| `test_components.py` | Checkpoint structure and input specification unit tests |

## Limitations and provenance
- Data splitting and original patient identifiers are not verifiable from this repository. A seed of 42 alone does not guarantee recreation of the original five folds.
- Inference code by itself cannot validate reported metrics without actual out-of-fold predictions.
- The saved first-fold checkpoint tested against original images is not a trained Pyramid+CLAHE checkpoint. Do not swap its inputs and report the resulting metric as a retraining experiment.
- The manuscript's five-fold method, ten-fold artifact-control experiment, and RingingScore calculations must be distinguished. The no-reference RingingScore implementation is not included here because the exact executable formula and thresholds cannot be verified from the supplied files.
- No dataset, clinical images, or private checkpoints are distributed in this public-code scaffold.

## Citation
Pathology Stage Aware Selective Enhancement and Gibbs Artifact Control for Diabetic Retinopathy Fundus Images. Manuscript in preparation. A permanent citation and version tag can be added after publication.
```
