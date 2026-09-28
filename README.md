# Skin Lesion Image Classification (HAM10000)

Classifying dermatoscopic images of pigmented skin lesions into 7 diagnostic categories, comparing a logistic-regression baseline with transfer learning on EfficientNetB0.

*CSC 411 – Interdisciplinary Machine Learning for Data Scientists, San Francisco State University, Fall 2024.*
Full write-up: [`reports/skin_lesion_classification_report.pdf`](reports/skin_lesion_classification_report.pdf)

## Problem

Skin cancers arise from skin lesions. Non-melanoma cancers are highly treatable when caught early, and melanoma is less common but far more likely to spread. The goal was to classify lesion images into the seven HAM10000 categories so that dangerous lesions can be flagged early:

| Code | Lesion type |
|---|---|
| `akiec` | Actinic keratoses and intraepithelial carcinoma / Bowen's disease |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

Because this is a medical task where missing a positive case is costly, **recall** was treated as the primary metric.

## Data

**Source:** [Skin Cancer MNIST: HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (`kmader/skin-cancer-mnist-ham10000`)

- 10,015 dermatoscopic images at 600 × 450 px, in two image folders, plus `HAM10000_metadata.csv` (`lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, `localization`). The target is `dx`.
- `hmnist_28_28_RGB.csv` holds the same images downsampled to 28 × 28 × 3 and flattened: 10,015 rows × 2,353 columns (2,352 pixel columns plus `label`).
- **Class imbalance is severe:** `nv` makes up 66.9% of images, `mel` 11.1% and `bkl` 11.0%. The remaining four classes share 11.2%, and `df` alone is only 1.1%.
- There are 57 missing `age` values (age was not used as a feature) and no duplicate `image_id`s.

## Approach

**Preprocessing**
- 70 / 10 / 20 train / validation / test split (7,210 / 802 / 2,003 images).
- Class balancing to **850 samples per class**: minority classes upsampled (Keras `ImageDataGenerator` augmentation with flips, rotation and zoom for images; `sklearn.utils.resample` for the tabular data) and `nv` downsampled. This gave a final training set of 5,864 images.
- Images resized to 224 × 224 and rescaled to [0, 1]. Tabular pixels scaled with `MinMaxScaler`.

**Models**
1. **Multinomial logistic regression (baseline):** trained on the 2,352 flattened 28 × 28 RGB pixels with `solver='lbfgs'`, `C=1` and `max_iter=1000`, tuned by hand because `GridSearchCV` was too slow on this data.
2. **EfficientNetB0 (transfer learning):** ImageNet weights, then GlobalAveragePooling, Dense(128, ReLU) and Dense(7, softmax), trained with categorical cross-entropy.
   - *Iteration 1:* train the head with the base frozen (Adam, lr 1e-3), then unfreeze and fine-tune the whole network (lr 1e-5) with early stopping on validation loss. It trained for 25 epochs.
   - *Iteration 2:* add L2 regularization on the dense layer and Dropout(0.5), with max epochs raised to 50. Early stopping halted it at 10 epochs.

## Key results

Test-set metrics from the report (§7.2), rounded to three decimals:

| Model | Accuracy | Recall | Precision | F1 | Log loss |
|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.563 | 0.563 | 0.698 | 0.606 | 1.346 |
| **EfficientNetB0 – iteration 1** | **0.781** | **0.754** | **0.816** | **0.668** | **0.599** |
| EfficientNetB0 – iteration 2 (L2 + dropout) | 0.773 | 0.754 | 0.796 | 0.661 | 0.872 |

- **EfficientNetB0 iteration 1 was the best model.** It had the highest accuracy, recall and precision and the lowest log loss.
- The baseline suffers because flattening 28 × 28 pixels throws away spatial structure.
- The extra regularization in iteration 2 over-constrained the model rather than helping it.

> **Note:** the report's §7.3 prose describes iteration 2 with slightly different figures (recall 76.1%, F1 66.6%, precision 80.2%, log loss 0.854) than its own results table. The table above uses the §7.2 table values.

**Future work (report §9.2):** unfreeze EfficientNet layers gradually instead of all at once, try VGG16 or ResNet50 and a custom CNN, and use feature selection to reduce the logistic-regression input size.

## About the notebook

[`notebooks/skin_lesion_classification_ham10000.ipynb`](notebooks/skin_lesion_classification_ham10000.ipynb) is an **earlier, partial version** of the project pipeline. It covers:

- dataset download
- merging the image folders
- sample images
- EDA (class, age, sex, localization and `dx_type` distributions; missing values; duplicates)
- the image train/test split and generator setup
- the augmentation helper
- logistic-regression preprocessing

It does **not** contain the final logistic-regression or EfficientNetB0 training runs. Its split and balancing outputs (for example a 7,214 / 798 train/validation split and a target of 421 per class) come from an earlier configuration and differ from the report. The results above come from the report.

## How to run

```bash
pip install -r requirements.txt
```

The data location is set by `DATA_DIR` in the first code cell, which reads the `HAM10000_DATA_DIR` environment variable. There are two ways to provide the data:

- **Automatic download:** leave `HAM10000_DATA_DIR` unset and configure [Kaggle API credentials](https://www.kaggle.com/docs/api). `kagglehub` then downloads the dataset (about 5.2 GB).
- **Local copy:** download and unzip the dataset yourself, then point the notebook at it:
  ```bash
  export HAM10000_DATA_DIR=/path/to/skin-cancer-mnist-ham10000
  ```

Then run:

```bash
jupyter notebook notebooks/skin_lesion_classification_ham10000.ipynb
```

A GPU is recommended; the original run used Google Colab with a Tesla T4. The notebook writes its train/test folders and augmented images inside the data directory.

## Repository structure

```
├── notebooks/
│   └── skin_lesion_classification_ham10000.ipynb   # data prep, EDA, preprocessing
├── reports/
│   └── skin_lesion_classification_report.pdf       # final project report
├── requirements.txt
└── README.md
```

## Team & my role

This was a team project by Dona Inayyah and **[TODO: teammate names]**.

My contributions (report §10):
- Found the dataset we used and planned the project workflow.
- Led the coding: dataset setup, organizing images into class directories, and handling class imbalance through augmentation.
- Trained and evaluated both models.
- Cleaned up teammates' code so it was consistent.
- Wrote the Introduction, Data Preprocessing and Modeling Approach sections of the proposal.
