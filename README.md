# RadClinNet — Multimodal CT + EHR Survival Prediction in NSCLC

**AMS 563 · Medical Image Analysis · Stony Brook University**  
Presented to Prof. Chenyu You | Author: Shruti Jagdale

---

## Overview

RadClinNet is an end-to-end multimodal deep learning pipeline that fuses raw chest CT DICOM images with EHR-derived structured clinical variables to predict binary survival outcome (Dead / Alive) in Non-Small Cell Lung Cancer (NSCLC) patients.

Three model architectures are trained and compared:

| Model | Input | Architecture |
|---|---|---|
| **Tabular MLP** (clinical-only baseline) | EHR-derived clinical features | 3-layer MLP |
| **CT-Only** | 8 axial CT slices @ 96×96 | ImageNet-pretrained ResNet18 |
| **Fusion Model** | CT + clinical features | ResNet18 + MLP → concatenation → classifier head |

---

## Motivation

NSCLC carries an ~85% 5-year mortality rate for advanced disease and is the leading cause of cancer mortality worldwide. Current clinical risk scores rely almost exclusively on tabular EHR variables (demographics, cancer stage, histology, lab values). CT scans are routinely acquired but rarely integrated into risk models.

This project investigates whether fusing CT imaging with structured clinical data yields stronger survival prediction compared to either modality alone.

---

## Dataset

**NSCLC-Radiogenomics** (The Cancer Imaging Archive, TCIA)

- 211 patients total; 211 with labeled survival status (`Dead` / `Alive`)
- 24 patients with matched DICOM CT studies (used for CT and fusion experiments)
- **Target:** `Survival Status` → binary label (`Dead = 1`, `Alive = 0`)
- **Class distribution (full cohort):** Alive 148 (70%) · Dead 63 (30%)

**Tabular features used:**

| Category | Features |
|---|---|
| Demographics | Age at diagnosis, gender, ethnicity, weight |
| Smoking | Smoking status, pack years, quit year |
| Tumor | Location (RUL/RML/RLL/LUL/LLL), histology, %GG |
| Pathology | T/N/M stage, histopathological grade, lymphovascular invasion, pleural invasion |
| Genomics | EGFR mutation status, KRAS mutation status, ALK translocation status |
| Treatment | Adjuvant treatment, chemotherapy, radiation |
| Outcome | Recurrence, recurrence location, time to death (days) |

> **Why this dataset?** It is the only public dataset pairing raw CT DICOMs with matched EHR-derived clinical labels suitable for multimodal learning.

---

## Pipeline

```
DICOM Loading → CT Preprocessing → Tabular Prep → Model Training → Evaluation + Grad-CAM
```

### CT Preprocessing
1. Load DICOM series with `pydicom`; apply RescaleSlope / RescaleIntercept for Hounsfield Unit (HU) conversion
2. Select the series with the most slices per patient
3. Sample `n_slices = 8` evenly spaced axial slices
4. HU windowing: clip to `[-1000, 400]` and normalize to `[0, 1]`
5. Bilinear resize to `96 × 96` pixels per slice
6. Replicate grayscale to 3 channels for ResNet18 compatibility

### Tabular Preprocessing
- Drop date columns and direct label leakage columns
- One-hot encode all categorical features (with `dummy_na=True`)
- Impute missing values with `most_frequent` strategy
- Standardize with `StandardScaler`

### Data Augmentation
- Random horizontal flipping applied during training to reduce overfitting on the small 36-patient training set

---

## Model Architectures

### Tabular MLP (clinical-only baseline)
- Input: EHR-derived features (one-hot encoded + standardized)
- Architecture: Linear(d→128) → ReLU → Dropout(0.4) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1)
- Trained on the full 211-patient cohort

### CT-Only (ResNet18)
- Input: 8 slices × 3 channels × 96 × 96
- Encoder: ImageNet-pretrained ResNet18 (512-dim output)
- Slice pooling: mean over 8 slices → 512-dim vector
- Head: Linear(512→1)
- Trained on 36-patient matched subset

### Fusion Model
- CT encoder: same ResNet18 as above (512-dim)
- Tabular encoder: Linear(d→128) → ReLU → Dropout(0.2) → Linear(128→64) (64-dim)
- Fusion: concatenation → 576-dim → Linear(576→64) → ReLU → Linear(64→1)
- Trained on 36-patient matched subset

**All models:** Adam optimizer · `weight_decay=1e-3` · `BCEWithLogitsLoss` with class weights · 6 epochs · `lr=1e-3`

---

## Iterative Development

| Iteration | Change | CT-Only AUROC | Fusion AUROC | Outcome |
|---|---|---|---|---|
| 1 | Custom 3-layer CNN from scratch | 0.722 | 1.000 | Overfit |
| 2 | + Dropout, weight decay, augmentation, class weights | 0.722 | 1.000 | No improvement |
| 3 | Replace CNN with ImageNet-pretrained ResNet18 | **0.917** | **0.861** | Best results |

Transfer learning was essential — a custom CNN cannot learn meaningful imaging features from only 36 patients.

---

## Results

| Model | AUROC | AUPRC | Brier Score | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Tabular MLP | 1.000* | 1.000* | 0.034 | 1.000 | 1.000 |
| CT-Only (ResNet18) | 0.889 | 0.917 | 0.162 | 1.000 | 0.500 |
| Fusion Model | 0.861 | 0.856 | 0.214 | 1.000 | 0.500 |

*Tabular model achieves perfect test separation on n=12 test patients — consistent with strong clinical predictors (stage, histology) but should be interpreted cautiously given the small test set.*

**Key finding:** Transfer learning boosted CT-only AUROC from 0.722 → 0.889. The fusion model (0.861) reflects genuine multimodal combination rather than tabular dominance — the CT branch contributes real signal that modulates the overconfident tabular predictions.

---

## Interpretability — Grad-CAM

Grad-CAM visualizations were generated for the CT-Only ResNet18 model. Attention currently concentrates on image background rather than lung tissue — a known small-data artifact. With more training patients, the model would learn to focus on clinically relevant anatomy (tumor region, lymph nodes).

---

## Limitations

- **Small dataset:** 112 total patients, only 12 in the CT test set. Results are not statistically stable.
- **Tabular ceiling effect:** Structured clinical variables (stage, histology, age) are very strong NSCLC survival predictors. This makes it difficult for imaging to demonstrate measurable added value on this cohort.
- **Grad-CAM background attention:** The CT model lacks spatial supervision and does not focus on clinically relevant anatomy.

---

## Future Work

- **Scale to larger cohorts:** TCGA-LUAD or NLST for statistically stable evaluation and cross-validation
- **Cross-attention fusion:** Replace concatenation with a Transformer cross-attention mechanism for dynamic modality weighting
- **3D volumetric models:** 3D CNNs or Vision Transformer-based models to capture full volumetric CT context rather than 2D slices
- **Lung segmentation priors:** Use segmentation masks to restrict CT attention to clinically relevant anatomy

---

## Repository Structure

```
CT-EHR-multimodal-transformer/
├── AMS563_PROJECT.ipynb       # Full pipeline notebook (Google Colab)
├── AMS 563 PROJECT.pdf        # Project slides (RadClinNet presentation)
└── README.md
```

---

## Setup

The notebook is designed to run on **Google Colab** with data stored in Google Drive.

```python
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn pydicom torch torchvision tqdm
```

**Data paths to configure in the notebook (Section 1):**

```python
PROJECT_ROOT   = Path('/content/drive/MyDrive/AMS 563/project')
CLINICAL_FILE  = PROJECT_ROOT / 'NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv'
IMAGE_ROOT     = PROJECT_ROOT / 'NLCSC-Radiogenomics'   # folder of per-patient DICOM directories
```

Download the NSCLC-Radiogenomics dataset from [TCIA](https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/).

---

## Configuration

Key hyperparameters are controlled via `CFG` in the notebook:

```python
CFG = {
    'random_state': 42,
    'max_patients_for_ct': 24,  # CT/fusion cohort cap
    'img_size': 96,             # spatial resolution per slice
    'n_slices': 8,              # axial slices sampled per patient
    'batch_size': 4,
    'num_workers': 2,
    'epochs': 6,
    'lr': 1e-3,
}
```

---

## Citation

If you use this code or pipeline, please cite the NSCLC-Radiogenomics dataset:

> Bakr, S. et al. (2018). *A radiogenomic dataset of non-small cell lung cancer.* Scientific Data, 5, 180202.
