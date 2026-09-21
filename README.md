# Leakage-Safe Block-Disjoint Benchmark and Diversity-Driven Stacking Ensemble for N-BaIoT

A reproducible research pipeline for **multi-class IoT botnet traffic classification** on the **N-BaIoT dataset**, with particular emphasis on preventing temporal/block-level data leakage.

The project implements the methodology described in:

> **A Leakage-Safe Block-Disjoint Benchmark and Diversity-Driven Stacking Ensemble for Multi-Class IoT Botnet Traffic Classification: An Analytical and Empirical Study on the N-BaIoT Dataset**

---

## 1. Overview

The N-BaIoT dataset contains network-flow statistics generated from IoT devices under benign traffic and Mirai/Gafgyt (BASHLITE) attacks.

A major issue addressed by this project is **evaluation leakage caused by random row-level train/test splitting**. Consecutive N-BaIoT rows can be strongly correlated because Kitsune statistics are computed using decayed aggregation windows. Therefore, randomly distributing individual rows across training and testing sets can place highly related observations in both partitions.

This repository implements a **block-disjoint evaluation protocol**:

- Raw rows are grouped into contiguous **500-row blocks**.
- Blocks, rather than individual rows, are used as the splitting unit.
- A held-out test partition is separated at block level.
- Five-fold cross-validation is also grouped by block.
- Preprocessing, feature selection, model fitting and hyperparameter-related computations are restricted to the appropriate training partition.
- Leakage assertions are explicitly implemented in the pipeline.

The study uses an **11-class classification task**:

1. Benign
2. Mirai ACK
3. Mirai Scan
4. Mirai SYN
5. Mirai UDP
6. Mirai UDPPlain
7. Gafgyt Combo
8. Gafgyt Junk
9. Gafgyt Scan
10. Gafgyt TCP
11. Gafgyt UDP

---

## 2. Dataset

The experiments use the **N-BaIoT dataset** from the UCI Machine Learning Repository.

### Full dataset

- 9 IoT devices
- 7,062,606 network-flow rows
- 115 Kitsune statistical features
- Benign, Mirai and Gafgyt/BASHLITE traffic
- Five decay windows are represented in the Kitsune statistics

### Experimental sample

To make the complete benchmark computationally manageable, the pipeline constructs a stratified sample of:

- **267,000 rows**
- **534 blocks**
- **500 rows per block**
- **11 classes**

The sample contains:

| Family | Rows |
|---|---:|
| Benign | 27,000 |
| Gafgyt/BASHLITE | 135,000 |
| Mirai | 105,000 |
| **Total** | **267,000** |

The experimental partition is:

- Training + validation: **213,500 rows / 427 blocks**
- Held-out test: **53,500 rows / 107 blocks**
- Training/validation: **5 grouped folds**

> The raw N-BaIoT dataset is not included in this repository. Download it separately from the UCI Machine Learning Repository and arrange the files according to the directory structure described below.

---

## 3. Project Structure

```text
.
├── code/
│   ├── harness.py
│   ├── stage1_build_sample.py
│   ├── stage2_eda_and_folds.py
│   ├── stage3_feature_selection.py
│   ├── stage4_classical_ml.py
│   ├── stage5_deep_learning.py
│   ├── stage6_proposed_ensemble.py
│   ├── stage6b_ensemble_refit_test.py
│   ├── stage7_ablation.py
│   ├── stage8_robustness.py
│   ├── stage9_explainability.py
│   ├── stage10_statistics.py
│   └── stage11_consolidate_and_figures.py
│
├── data/
│   ├── sampled_raw.pkl
│   ├── sampled_with_folds.pkl
│   ├── classical_oof.npz
│   └── ...
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── models/
│
├── README.md
└── requirements.txt
```

The supplied scripts currently use:

```text
/home/claude/nbaiot_raw
/home/claude/iot_botnet_study/data
/home/claude/iot_botnet_study/results
```

If you run the repository on another machine, update these paths in the Python scripts or replace them with configurable project-relative paths.

---

## 4. Raw Dataset Directory Structure

The sampling script expects the N-BaIoT CSV files to be arranged approximately as follows:

```text
nbaiot_raw/
├── Danmini Doorbell/
│   ├── benign_traffic.csv
│   ├── mirai/
│   │   ├── ack.csv
│   │   ├── scan.csv
│   │   ├── syn.csv
│   │   ├── udp.csv
│   │   └── udpplain.csv
│   └── gafgyt/
│       ├── combo.csv
│       ├── junk.csv
│       ├── scan.csv
│       ├── tcp.csv
│       └── udp.csv
│
├── Philips B120N10 Baby Monitor/
│   ├── benign_traffic.csv
│   ├── mirai/
│   └── gafgyt/
│
└── ... other N-BaIoT devices ...
```

The exact device directory names should match the directories present in the downloaded dataset.

---

## 5. Installation

Recommended environment:

- Python 3.11
- NumPy
- Pandas
- SciPy
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- PyTorch
- SHAP
- statsmodels
- matplotlib
- joblib

Install the dependencies:

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm catboost torch shap statsmodels matplotlib joblib
```

For reproducible research, it is recommended to pin the package versions used in your execution environment.

---

## 6. Complete Pipeline

Run the stages in the following order:

```bash
python code/stage1_build_sample.py
python code/stage2_eda_and_folds.py
python code/stage3_feature_selection.py
python code/stage4_classical_ml.py
python code/stage5_deep_learning.py
python code/stage6_proposed_ensemble.py
python code/stage6b_ensemble_refit_test.py
python code/stage7_ablation.py
python code/stage8_robustness.py
python code/stage9_explainability.py
python code/stage10_statistics.py
python code/stage11_consolidate_and_figures.py
```

### Stage 1 — Build the experimental sample

`stage1_build_sample.py`

Performs:

- N-BaIoT file discovery
- Device/file inventory
- MD5 recording of source files
- 500-row block construction
- Stratified block sampling
- Label construction
- Duplicate and missing-value checks
- Creation of `sampled_raw.pkl`

The script samples up to 3,000 rows per source file using complete 500-row blocks.

---

### Stage 2 — EDA and block-disjoint folds

`stage2_eda_and_folds.py`

Performs:

- Feature identification
- Dataset statistics
- Duplicate analysis
- Zero-variance feature detection
- Correlation analysis
- High-correlation pair detection
- 80/20 block-disjoint train/test split
- Five-fold `StratifiedGroupKFold`
- Leakage assertions

The critical checks are:

```python
assert (check["n_roles"] == 1).all()
assert (check["n_folds"] == 1).all()
```

These ensure that a block does not cross train/test or CV-fold boundaries.

---

### Stage 3 — Feature Selection

`stage3_feature_selection.py`

The feature-selection pipeline consists of two stages:

1. Mutual Information ranking
2. Random Forest importance ranking

The rankings are combined and followed by correlation-based redundancy pruning.

A feature is considered redundant when:

```text
|Pearson correlation| >= 0.98
```

A feature-count sweep is then performed to identify a compact operating point.

The reported study retained:

```text
40 features
```

The selected feature list is saved to:

```text
results/tables/selected_features.json
```

---

## 7. Classical Machine Learning Benchmark

`stage4_classical_ml.py`

The benchmark evaluates ten classical models:

1. Logistic Regression
2. Linear SVM
3. RBF SVM
4. Random Forest
5. Extra Trees
6. XGBoost
7. LightGBM
8. CatBoost
9. HistGradientBoosting
10. k-NN

Evaluation uses grouped five-fold cross-validation.

### Metrics

The pipeline calculates:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)
- Macro OVR AUROC
- Macro OVR AUPRC
- Log Loss
- Multiclass Brier Score

**MCC is used as the primary classification metric** because it provides a useful summary for multiclass classification under class imbalance.

---

## 8. Deep Learning Benchmark

`stage5_deep_learning.py`

Four PyTorch architectures are evaluated:

### MLP

```text
40 → 256 → 128 → 11
```

with batch normalization and dropout.

### 1D-CNN

A 1D convolutional model treats the 40 selected features as a single-channel sequence.

### Bidirectional GRU

The 40-dimensional feature vector is arranged as a pseudo-sequence of four-feature chunks.

### Lightweight Transformer

A compact two-layer Transformer encoder operates on the same pseudo-sequence representation.

Common training components include:

- Class-weighted cross entropy
- Adam optimizer
- Weight decay
- ReduceLROnPlateau scheduler
- Gradient clipping
- Early stopping

The neural pipeline uses a numerical-stability safeguard:

```python
np.clip(X, -20.0, 20.0)
```

after RobustScaler transformation and before neural-network training.

---

## 9. Diversity-Driven Stacking Ensemble

The proposed ensemble is implemented in:

```text
stage6_proposed_ensemble.py
stage6b_ensemble_refit_test.py
```

The ensemble is not selected simply by taking the top-performing models.

Instead, it combines:

- Standalone model competence
- Pairwise prediction disagreement

The candidate pool contains the ten classical models.

The greedy selection uses:

```text
score(c) = α × diversity(c) + (1 − α) × MCC(c)
```

with:

```text
α = 0.6
ensemble size = 4
```

The reported selected base learners are:

```text
HistGradientBoosting
Logistic Regression
CatBoost
Extra Trees
```

Their probability outputs are concatenated and passed to a multinomial Logistic Regression meta-learner.

### Nested evaluation

The stacking model is evaluated using nested grouped cross-validation so that the meta-learner does not train on the fold being evaluated.

---

## 10. Held-Out Test Evaluation

`stage6b_ensemble_refit_test.py`

After model selection:

1. Base learners are refit on the complete training/validation set.
2. The scaler is refit on training/validation data.
3. The meta-learner is fitted using the leakage-safe OOF probability matrix.
4. The final ensemble is evaluated once on the block-disjoint held-out test set.

Saved model artifacts include:

```text
results/models/scaler.joblib
results/models/base_<model>.joblib
results/models/meta_learner.joblib
```

---

## 11. Ablation Study

`stage7_ablation.py`

The ablation pipeline investigates the contribution of different processing stages, including:

- Raw 115-feature Logistic Regression
- Scaled 115-feature Logistic Regression
- Full-feature Random Forest
- Random-Forest-only feature selection
- Proposed MI + RF ranking and redundancy-pruned feature selection
- Ensemble-related configurations

Output:

```text
results/tables/T_ablation_study.csv
```

---

## 12. Robustness and Generalization

`stage8_robustness.py`

The robustness suite evaluates:

### Noise robustness

Gaussian perturbations at:

```text
0%, 5%, 10%, 25%, 50%, 100%
```

of the training-feature standard deviation.

### Missing-feature robustness

Missing-feature simulation at:

```text
10%, 25%, 50%, 75%
```

### Training-size sensitivity

Training fractions:

```text
10%, 25%, 50%, 75%, 100%
```

### Cross-device generalization

Leave-one-device-out evaluation.

### Zero-day / cross-scenario evaluation

Leave-one-attack-subtype-out evaluation using the rate at which an unseen subtype is still detected as non-benign.

Outputs include:

```text
T_robustness.csv
T_cross_device_generalization.csv
T_cross_scenario_generalization.csv
```

---

## 13. Explainable AI

`stage9_explainability.py`

Three feature-importance methods are used:

1. SHAP TreeExplainer
2. Permutation importance
3. Random Forest Mean Decrease in Impurity (MDI)

The pipeline also calculates pairwise Spearman rank correlations between importance rankings.

Generated outputs include:

```text
results/tables/T_shap_global_importance.csv
results/tables/T_shap_classwise_importance.csv
results/tables/T_permutation_importance.csv
results/tables/T_mdi_importance.csv
```

Figures include:

```text
results/figures/F_SHAP_global.png
results/figures/F_SHAP_classwise.png
```

---

## 14. Statistical Validation

`stage10_statistics.py`

The statistical validation includes:

### Pairwise Wilcoxon signed-rank tests

Applied across the five grouped CV folds.

### Holm correction

Applied within each metric family.

### Friedman omnibus test

Used for multi-model comparison across folds.

### Average ranks

Calculated for the evaluated models.

### Paired bootstrap

A 2,000-resample bootstrap is used on the held-out test set for comparison between the proposed ensemble and the best individual base model identified from the test metrics.

The code explicitly notes that only five paired CV folds are available, which limits the minimum attainable exact two-sided Wilcoxon p-value.

---

## 15. Final Consolidation

`stage11_consolidate_and_figures.py`

This stage creates consolidated tables and publication figures.

Important final files include:

```text
results/tables/FINAL_METRICS.csv
results/tables/FINAL_ABLATION.csv
results/tables/FINAL_ROBUSTNESS.csv
results/tables/FINAL_STATISTICS_pairwise.csv
results/tables/FINAL_COMPLEXITY.csv
```

Main figures include:

```text
F_dataset_distribution.png
F_class_distribution.png
F_feature_count_sweep.png
F_MCC_comparison.png
F_DL_MCC_comparison.png
F_ablation_performance.png
F_noise_robustness.png
F_cross_device_generalization.png
F_cross_scenario_generalization.png
```

---

## 16. Key Reported Results

Under the block-disjoint evaluation protocol, the study reports a clear separation between linear models and tree-based ensembles.

Selected reported five-fold CV results:

| Model | MCC | Macro-F1 | Balanced Accuracy | AUROC |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 0.9808 ± 0.0221 | 0.9812 ± 0.0206 | 0.9787 ± 0.0251 | 0.9994 ± 0.0008 |
| XGBoost | 0.9793 ± 0.0160 | 0.9795 ± 0.0150 | 0.9792 ± 0.0186 | 0.9994 ± 0.0008 |
| LightGBM | 0.9777 ± 0.0176 | 0.9771 ± 0.0162 | 0.9745 ± 0.0191 | 0.9998 ± 0.0002 |
| CatBoost | 0.9745 ± 0.0188 | 0.9766 ± 0.0159 | 0.9792 ± 0.0154 | 0.9981 ± 0.0022 |
| Extra Trees | 0.9580 ± 0.0550 | 0.9617 ± 0.0492 | 0.9648 ± 0.0390 | 0.9970 ± 0.0046 |
| Random Forest | 0.8578 ± 0.0184 | 0.8554 ± 0.0122 | 0.8937 ± 0.0109 | 0.9878 ± 0.0025 |
| k-NN | 0.8179 ± 0.0459 | 0.8173 ± 0.0383 | 0.8244 ± 0.0379 | 0.9266 ± 0.0237 |
| Linear SVM | 0.1815 ± 0.0190 | 0.0825 ± 0.0062 | 0.1436 ± 0.0106 | 0.5467 ± 0.0255 |
| RBF SVM | 0.1328 ± 0.0121 | 0.0603 ± 0.0103 | 0.1214 ± 0.0111 | 0.5496 ± 0.0089 |
| Logistic Regression | 0.0695 ± 0.0193 | 0.0534 ± 0.0054 | 0.1269 ± 0.0136 | 0.5531 ± 0.0081 |

For the deep-learning benchmark, the reported held-out test MCC values are:

| Architecture | MCC |
|---|---:|
| 1D-CNN | 0.8851 |
| MLP | 0.7753 |
| Transformer | 0.7639 |
| Bi-GRU | 0.7590 |

The diversity-driven ensemble achieved:

```text
Nested CV MCC: 0.9887 ± 0.0154
Held-out test MCC: 0.9952
Held-out test Macro-F1: 0.9961
```

The reported final held-out comparison is intentionally preserved as an empirical result of this particular evaluation protocol rather than interpreted as a universal ranking of algorithms.

---

## 17. Leakage-Safety Design

The central methodological principle is:

```text
Raw N-BaIoT data
       │
       ▼
Contiguous 500-row blocks
       │
       ▼
Block-level stratified sampling
       │
       ├───────────────┐
       ▼               ▼
Train/Validation    Held-out Test
       │
       ▼
5 Grouped CV Folds
       │
       ▼
Training-only preprocessing
       │
       ▼
Training-only feature selection
       │
       ▼
Model training
       │
       ▼
OOF predictions
       │
       ▼
Diversity-driven model selection
       │
       ▼
Nested stacking
       │
       ▼
Final block-disjoint test
```

The important rule is:

> **No row from a block assigned to a validation/test role is used to fit preprocessing, feature-selection statistics, or model parameters for another role.**

---

## 18. Reproducibility

The pipeline uses:

```text
Random seed = 42
Block size = 500
Held-out test fraction = 20%
CV folds = 5
Feature correlation threshold = 0.98
Feature-count operating point = 40
Ensemble size = 4
Diversity weight α = 0.6
Bootstrap repetitions = 2,000
```

For exact reproduction, use the same:

- N-BaIoT source files
- Python version
- package versions
- operating-system/library configurations
- random seed
- data directory structure

---

## 19. Computational Notes

The benchmark contains relatively large tabular datasets. Some models have explicit training caps for computational tractability:

```text
RBF-SVM: 12,000 training rows per fold
k-NN:    40,000 training rows per fold
```

Other classical models are trained on the full training portion of each fold.

The deep-learning models can run on CPU or CUDA-enabled PyTorch. The code automatically selects:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 20. Citation

If you use this implementation or methodology in academic work, cite the associated manuscript:

```bibtex
@article{
  arumugam_nagarajan_nbaiot,
  title={A Leakage-Safe Block-Disjoint Benchmark and Diversity-Driven Stacking Ensemble for Multi-Class IoT Botnet Traffic Classification: An Analytical and Empirical Study on the N-BaIoT Dataset},
  author={Arumugam, Jeeva and Ramalingam, Nagarajan},
  journal={},
  year={}
}
```

Update the BibTeX metadata after the manuscript receives its final publication information.

---

## 21. Dataset Citation

The project uses the N-BaIoT dataset from the UCI Machine Learning Repository.

Please follow the dataset provider's citation and usage requirements when redistributing results.

---

## 22. License

Add the appropriate repository license before publishing the project, for example:

```text
MIT License
```

Do not add a license unless you have confirmed that the code and included materials can legally be distributed under that license.

---

## 23. Disclaimer

This repository is intended for **research and educational use** in IoT network-security and machine-learning experimentation.

The reported results are specific to the N-BaIoT dataset, sampling strategy, block construction, preprocessing, models, and evaluation protocol implemented in this repository. They should not be interpreted as proof of equivalent performance on other datasets, network environments, or real-time production traffic.

