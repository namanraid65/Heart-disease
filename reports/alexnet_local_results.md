# Local 1D AlexNet Results

## 1. Model Architecture

The classical AlexNet architecture (originally designed for 2D images) was adapted for 1D tabular clinical data representations. The 1D adaptation preserves AlexNet's hierarchical design motif while appropriately scaling depth and parameters for tabular heart disease diagnostics.

### Architectural Layout
```
Input: Tensor (Batch_Size, 1, 25)
  │
  ├── [Conv Block 1]: Conv1d(1 -> 32, k=3, s=1, p=1) ──> BatchNorm1d(32) ──> ReLU ──> MaxPool1d(k=2, s=2)  [L_out: 12]
  ├── [Conv Block 2]: Conv1d(32 -> 64, k=3, s=1, p=1) ──> BatchNorm1d(64) ──> ReLU ──> MaxPool1d(k=2, s=2)  [L_out: 6]
  ├── [Conv Block 3]: Conv1d(64 -> 128, k=3, s=1, p=1) ──> BatchNorm1d(128) ──> ReLU                        [L_out: 6]
  ├── [Conv Block 4]: Conv1d(128 -> 128, k=3, s=1, p=1) ──> BatchNorm1d(128) ──> ReLU                      [L_out: 6]
  ├── [Conv Block 5]: Conv1d(128 -> 64, k=3, s=1, p=1) ──> BatchNorm1d(64) ──> ReLU ──> AdaptiveAvgPool1d(3) [L_out: 3]
  │
  ├── [Flatten]: Vector of shape (Batch_Size, 192)
  ├── [Dense Block 1]: Linear(192 -> 64) ──> ReLU ──> Dropout(p=0.3)
  ├── [Dense Block 2]: Linear(64 -> 32) ──> ReLU ──> Dropout(p=0.3)
  └── [Output Layer]: Linear(32 -> 1) ──> Raw Logit (BCEWithLogitsLoss)
```

- **Input Dimension:** 25 clinical features (dynamically loaded from schema).
- **Trainable Parameters:** Exactly **120,257 parameters**.

---

## 2. Training Configuration

Each hospital client was trained strictly as an isolated local model using only its own training and validation partitions. No federated aggregation was performed.

| Hyperparameter / Setting | Configured Value | Description / Purpose |
| :--- | :--- | :--- |
| **Loss Function** | Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) | Numerically stable sigmoid + cross-entropy |
| **Optimizer** | Adam | Adaptive moment estimation |
| **Base Learning Rate** | $0.001$ ($10^{-3}$) | Initial step size |
| **Weight Decay ($L_2$ Reg)** | $10^{-4}$ | Regularization against tabular overfitting |
| **Batch Size** | 16 | Suited for small decentralized medical cohorts |
| **Maximum Epochs** | 150 | Upper limit on gradient iterations |
| **Learning Rate Scheduler** | `ReduceLROnPlateau` | Halves LR (`factor=0.5`) on validation plateau (patience: 10) |
| **Early Stopping** | Patience = 25 epochs ($\Delta = 10^{-4}$) | Restores best model state based on validation loss |
| **Dropout Rate** | $0.30$ | Applied in fully connected dense layers |
| **Random Seed** | 42 | Guaranteed reproducible training and weight initialization |

---

## 3. Hospital 1 Results (Cleveland Clinic Foundation)

### Evaluation Summary on Held-Out Test Set ($N=46$)
- **Best Validation Epoch:** 3 (Validation Loss: $0.4192$)
- **Accuracy:** **$86.96\%$** ($40 / 46$ correct predictions)
- **Precision:** **$85.71\%$**
- **Recall (Sensitivity):** **$85.71\%$** ($18 / 21$ positive disease cases detected)
- **Specificity:** **$88.00\%$** ($22 / 25$ healthy patients correctly ruled out)
- **F1-Score:** **$0.8571$**
- **ROC-AUC:** **$0.9448$** (Outstanding discriminative separation)
- **Confusion Matrix Breakdown:**
  - True Positives (TP): 18
  - False Positives (FP): 3
  - True Negatives (TN): 22
  - False Negatives (FN): 3

---

## 4. Hospital 2 Results (Hungarian Institute of Cardiology)

### Evaluation Summary on Held-Out Test Set ($N=44$)
- **Best Validation Epoch:** 2 (Validation Loss: $0.4746$)
- **Accuracy:** **$79.55\%$** ($35 / 44$ correct predictions)
- **Precision:** **$70.59\%$**
- **Recall (Sensitivity):** **$75.00\%$** ($12 / 16$ positive disease cases detected)
- **Specificity:** **$82.14\%$** ($23 / 28$ healthy patients correctly ruled out)
- **F1-Score:** **$0.7273$**
- **ROC-AUC:** **$0.8750$** (Strong discriminative performance)
- **Confusion Matrix Breakdown:**
  - True Positives (TP): 12
  - False Positives (FP): 5
  - True Negatives (TN): 23
  - False Negatives (FN): 4

---

## 5. Hospital 3 Results (Switzerland University Hospital)

### Evaluation Summary on Held-Out Test Set ($N=19$)
- **Best Validation Epoch:** 5 (Validation Loss: $0.2372$)
- **Accuracy:** **$94.74\%$** ($18 / 19$ correct predictions)
- **Precision:** **$94.74\%$**
- **Recall (Sensitivity):** **$100.00\%$** ($18 / 18$ positive disease cases detected, **0 False Negatives**)
- **Specificity:** **$0.00\%$** ($0 / 1$ healthy patient correctly classified; 1 False Positive)
- **F1-Score:** **$0.9730$**
- **ROC-AUC:** **$0.3889$**
- **Confusion Matrix Breakdown:**
  - True Positives (TP): 18
  - False Positives (FP): 1
  - True Negatives (TN): 0
  - False Negatives (FN): 0

---

## 6. Performance Comparison

All metrics were computed strictly on the independent held-out test split of each hospital client:

| Hospital Client | Test Samples ($N$) | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score | ROC-AUC | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hospital 1 (Cleveland)** | **46** | **86.96%** | **85.71%** | **85.71%** | **88.00%** | **0.8571** | **0.9448** | 18 / 3 / 22 / 3 |
| **Hospital 2 (Hungarian)** | **44** | **79.55%** | **70.59%** | **75.00%** | **82.14%** | **0.7273** | **0.8750** | 12 / 5 / 23 / 4 |
| **Hospital 3 (Switzerland)** | **19** | **94.74%** | **94.74%** | **100.00%** | **0.00%** | **0.9730** | **0.3889** | 18 / 1 / 0 / 0 |

---

## 7. Confusion Matrix Analysis

### Clinical Risk & Error Analysis
1. **Critical Clinical Metric — False Negatives (FN):**
   - In cardiovascular diagnostic screening, a **False Negative** represents a patient with active coronary heart disease who is incorrectly sent home without treatment.
   - **Hospital 1:** 3 False Negatives out of 21 true positives ($14.3\%$ miss rate).
   - **Hospital 2:** 4 False Negatives out of 16 true positives ($25.0\%$ miss rate).
   - **Hospital 3:** **0 False Negatives** ($0.0\%$ miss rate). The model captured 100% of diseased patients.
2. **False Positives (FP) & Specificity:**
   - **Hospital 1:** 3 False Positives out of 25 true healthy individuals ($88.0\%$ Specificity).
   - **Hospital 2:** 5 False Positives out of 28 true healthy individuals ($82.1\%$ Specificity).
   - **Hospital 3:** 1 False Positive on the single healthy patient present in the test set.

---

## 8. Training Behavior & Convergence

1. **Hospital 1 (Cleveland):**
   - Rapid convergence within 3 epochs. Training loss dropped from $0.6353$ to $<0.01$ as the network fit local continuous interactions. Early stopping halted training at Epoch 28, successfully protecting the model from overfitting.
2. **Hospital 2 (Hungarian):**
   - Optimal validation performance achieved at Epoch 2 (Val Loss: $0.4746$, Val ROC-AUC: $0.846$). Early stopping triggered at Epoch 27.
3. **Hospital 3 (Switzerland):**
   - Reached minimum validation loss at Epoch 5 ($0.2372$). The model rapidly prioritized the dominant positive class ($93\%$ of training records), triggering early stopping at Epoch 30.

---

## 9. Key Observations for Future Federated Learning

1. **Baseline Local Benchmarks Established:**
   - Hospital 1 achieved an excellent local baseline ($86.96\%$ Accuracy, $0.9448$ ROC-AUC).
   - Hospital 2 achieved solid predictive ability ($79.55\%$ Accuracy, $0.8750$ ROC-AUC).
2. **The Non-IID Local Skew Effect (Hospital 3):**
   - Hospital 3’s high apparent accuracy ($94.74\%$) and F1-score ($0.9730$) is heavily influenced by the $93.5\%$ disease prevalence in its patient population. However, its local model suffers from zero specificity ($0.00\%$), as it learned a heavy positive-class bias.
3. **Motivation for Federated Learning:**
   - Isolated local training on Hospital 3 produces a one-sided classifier incapable of identifying healthy patients.
   - Federated collaboration will enable Hospital 3 to benefit from the rich healthy-patient distributions available at Hospital 1 and Hospital 2 without sharing raw patient records.
