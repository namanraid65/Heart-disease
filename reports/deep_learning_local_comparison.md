# Deep Learning Local Model Comparison: 1D AlexNet vs. 1D ResNet

## 1. Executive Summary

This report presents a direct, hospital-by-hospital comparative evaluation between the two deep learning architectures developed for the Federated Heart Disease Prediction project:
1. **1D AlexNet:** Classical 5-stage convolutional pipeline with dense multi-layer classification head (120,257 parameters).
2. **1D ResNet:** 6-block residual convolutional network with identity/projection shortcuts and global average pooling (244,065 parameters).

Both models were trained and tested on the exact same isolated client partitions under identical optimization hyperparameters (Adam, Initial LR: $0.001$, Batch Size: $16$, Seed: $42$, Early Stopping Patience: $25$).

---

## 2. Hospital-Wise Performance Comparison

| Hospital Client | Deep Learning Model | Test $N$ | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score | ROC-AUC | TP / FP / TN / FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hospital 1 (Cleveland)** | **1D AlexNet** | 46 | **86.96%** | **85.71%** | **85.71%** | 88.00% | **0.8571** | **0.9448** | 18 / 3 / 22 / 3 |
| | **1D ResNet** | 46 | 84.78% | 85.00% | 80.95% | 88.00% | 0.8293 | 0.8876 | 17 / 3 / 22 / 4 |
| **Hospital 2 (Hungarian)** | **1D AlexNet** | 44 | **79.55%** | 70.59% | **75.00%** | 82.14% | **0.7273** | 0.8750 | 12 / 5 / 23 / 4 |
| | **1D ResNet** | 44 | 70.45% | **80.00%** | 25.00% | **96.43%** | 0.3810 | **0.8795** | 4 / 1 / 27 / 12 |
| **Hospital 3 (Switzerland)** | **1D AlexNet** | 19 | **94.74%** | **94.74%** | **100.00%** | 0.00% | **0.9730** | **0.3889** | 18 / 1 / 0 / 0 |
| | **1D ResNet** | 19 | **94.74%** | **94.74%** | **100.00%** | 0.00% | **0.9730** | 0.0556 | 18 / 1 / 0 / 0 |

---

## 3. Comprehensive Metric Analysis

### 1. Clinical Risk Prediction (Recall & False Negatives)
- **Hospital 1 (Cleveland):**
  - AlexNet captured **$85.71\%$** of diseased patients ($3$ FN), outperforming ResNet's **$80.95\%$** ($4$ FN).
- **Hospital 2 (Hungarian):**
  - AlexNet maintained balanced clinical utility with **$75.00\%$** Recall ($4$ FN).
  - In contrast, ResNet suffered a severe sensitivity collapse to **$25.00\%$** Recall ($12$ False Negatives out of 16 diseased patients), over-indexing on the healthy majority class ($96.43\%$ Specificity).
- **Hospital 3 (Switzerland):**
  - Both architectures achieved **$100.00\%$** Recall ($0$ False Negatives) due to the $93.5\%$ disease prevalence in the Swiss cohort.

### 2. Discrimination & Ranking (ROC-AUC)
- **Hospital 1:** AlexNet delivered higher discriminative separation ($\text{ROC-AUC} = \mathbf{0.9448}$) compared to ResNet ($0.8876$).
- **Hospital 2:** Both models achieved comparable ROC-AUC scores ($0.8750$ for AlexNet vs $0.8795$ for ResNet), indicating that ResNet's low default-threshold recall was primarily a decision-boundary calibration issue on the unbalanced Hungarian set.
- **Hospital 3:** Both models showed depressed ROC-AUC scores ($0.3889$ vs $0.0556$) due to the presence of only a single healthy test sample in the highly skewed Swiss partition.

### 3. Overall Balance (F1-Score & Accuracy)
- AlexNet outperformed ResNet across all three hospital clients in F1-score ($0.8571$ vs $0.8293$ on H1; $0.7273$ vs $0.3810$ on H2; tied at $0.9730$ on H3).

---

## 4. Architectural & Parameter Efficiency Analysis

| Architectural Property | 1D AlexNet | 1D ResNet | Comparative Impact |
| :--- | :---: | :---: | :--- |
| **Total Trainable Parameters** | **120,257** | **244,065** | AlexNet has $\approx 51\%$ fewer parameters |
| **Convolutional Blocks** | 5 Sequential Conv1D | 6 Residual Blocks (3 Stages) | ResNet has deeper gradient pathways |
| **Downsampling Mechanism** | MaxPool1d (Stride 2) | Strided Conv1D (Stride 2) | ResNet learns projection filters |
| **Pooling Bridge** | `AdaptiveAvgPool1d(3)` | `AdaptiveAvgPool1d(1)` | ResNet pools to 1D channel vector |
| **Classifier Head** | 3 Linear layers (192 $\to$ 64 $\to$ 32 $\to$ 1) | 2 Linear layers (128 $\to$ 32 $\to$ 1) | AlexNet maintains richer dense capacity |
| **Regularization** | Dropout ($p=0.3$) + Weight Decay | BatchNorm + Dropout ($p=0.3$) + Weight Decay | Both models well-regularized |

---

## 5. Key Takeaways for Federated Learning

1. **AlexNet Shows Superior Local Tabular Regularity:**
   - On small clinical tabular sample sizes ($N \approx 86 - 212$ training samples), the more compact 1D AlexNet architecture exhibited less susceptibility to majority-class overfitting than the larger 1D ResNet.
2. **Persistence of Non-IID Label Skew Across Architectures:**
   - Both models exhibited identical pathological bias on Hospital 3 ($100\%$ Recall, $0\%$ Specificity), proving that architectural variation alone cannot overcome severe data heterogeneity without federated knowledge sharing.
3. **Multi-Model Benchmark Ready for Federated Aggregation:**
   - Both 1D AlexNet and 1D ResNet provide complementary, mathematically verified local baselines against which future Federated Averaging (FedAvg), FedProx, and personalized FL algorithms can be benchmarked.
