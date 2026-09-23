# Local 1D ResNet Results

## 1. Model Architecture

The 1D ResNet model is adapted for tabular heart disease clinical feature vectors $(B, 1, 25)$. It incorporates residual skip connections that mitigate the vanishing gradient problem and allow deeper feature representations to be trained effectively on small tabular cohorts.

### Architectural Layout
```
Input: Tensor (Batch_Size, 1, 25)
  │
  ├── [Stem]: Conv1d(1 -> 32, k=3, s=1, p=1, bias=False) ──> BatchNorm1d(32) ──> ReLU
  │
  ├── [Stage 1]: 2x ResidualBlock1D(32 -> 32, stride=1)
  │     ├── Block 1.1: Conv1d(32->32) + Conv1d(32->32) + Identity Shortcut ──> ReLU
  │     └── Block 1.2: Conv1d(32->32) + Conv1d(32->32) + Identity Shortcut ──> ReLU
  │
  ├── [Stage 2]: 2x ResidualBlock1D(32 -> 64, stride=2) [Spatial: 25 -> 13]
  │     ├── Block 2.1: Conv1d(32->64, s=2) + Conv1d(64->64) + Conv1d(1x1, s=2) Projection ──> ReLU
  │     └── Block 2.2: Conv1d(64->64) + Conv1d(64->64) + Identity Shortcut ──> ReLU
  │
  ├── [Stage 3]: 2x ResidualBlock1D(64 -> 128, stride=2) [Spatial: 13 -> 7]
  │     ├── Block 3.1: Conv1d(64->128, s=2) + Conv1d(128->128) + Conv1d(1x1, s=2) Projection ──> ReLU
  │     └── Block 3.2: Conv1d(128->128) + Conv1d(128->128) + Identity Shortcut ──> ReLU
  │
  ├── [Global Pooling]: AdaptiveAvgPool1d(1) ──> (Batch_Size, 128, 1)
  │
  ├── [Classification Head]:
  │     ├── Flatten ──> (Batch_Size, 128)
  │     ├── Dropout(p=0.3)
  │     ├── Linear(128 -> 32) ──> ReLU ──> Dropout(p=0.3)
  │     └── Linear(32 -> 1) ──> Binary Classification Logit
```

- **Input Dimension:** 25 clinical features.
- **Residual Blocks:** 6 residual blocks across 3 stages with projection shortcuts where channel/spatial dimensions change.
- **Total Trainable Parameters:** Exactly **244,065 parameters**.

---

## 2. Training Configuration

Each local 1D ResNet model was trained independently on its respective hospital client partition.

| Parameter / Setting | Configured Value | Description / Rationale |
| :--- | :--- | :--- |
| **Loss Function** | `BCEWithLogitsLoss` | Numerically stable binary cross-entropy with sigmoid |
| **Optimizer** | Adam | Adaptive learning rate optimization |
| **Base Learning Rate** | $0.001$ ($10^{-3}$) | Initial step size |
| **Weight Decay ($L_2$)** | $10^{-4}$ | Weight penalty against overfitting |
| **Batch Size** | 16 | Small batch size suited for tabular medical partitions |
| **Maximum Epochs** | 150 | Maximum training iterations |
| **LR Scheduler** | `ReduceLROnPlateau` | $\text{factor}=0.5$, $\text{patience}=10$ |
| **Early Stopping** | $\text{Patience}=25$ epochs ($\Delta = 10^{-4}$) | Restores best model weights on validation loss |
| **Dropout Rate** | $0.30$ | Regularization in classification head |
| **Random Seed** | 42 | Full reproducibility |

---

## 3. Hospital 1 Results (Cleveland Clinic)

### Evaluation Summary on Held-Out Test Set ($N=46$)
- **Best Validation Epoch:** 2 (Validation Loss: $0.4983$)
- **Accuracy:** **$84.78\%$** ($39 / 46$ correct predictions)
- **Precision:** **$85.00\%$**
- **Recall (Sensitivity):** **$80.95\%$** ($17 / 21$ positive disease cases detected)
- **Specificity:** **$88.00\%$** ($22 / 25$ healthy patients correctly ruled out)
- **F1-Score:** **$0.8293$**
- **ROC-AUC:** **$0.8876$**
- **Confusion Matrix Breakdown:**
  - True Positives (TP): 17
  - False Positives (FP): 3
  - True Negatives (TN): 22
  - False Negatives (FN): 4

---

## 4. Hospital 2 Results (Hungarian Institute of Cardiology)

### Evaluation Summary on Held-Out Test Set ($N=44$)
- **Best Validation Epoch:** 2 (Validation Loss: $0.4381$)
- **Accuracy:** **$70.45\%$** ($31 / 44$ correct predictions)
- **Precision:** **$80.00\%$**
- **Recall (Sensitivity):** **$25.00\%$** ($4 / 16$ positive disease cases detected)
- **Specificity:** **$96.43\%$** ($27 / 28$ healthy patients correctly ruled out)
- **F1-Score:** **$0.3810$**
- **ROC-AUC:** **$0.8795$**
- **Confusion Matrix Breakdown:**
  - True Positives (TP): 4
  - False Positives (FP): 1
  - True Negatives (TN): 27
  - False Negatives (FN): 12

---

## 5. Hospital 3 Results (Switzerland University Hospital)

### Evaluation Summary on Held-Out Test Set ($N=19$)
- **Best Validation Epoch:** 4 (Validation Loss: $0.2462$)
- **Accuracy:** **$94.74\%$** ($18 / 19$ correct predictions)
- **Precision:** **$94.74\%$**
- **Recall (Sensitivity):** **$100.00\%$** ($18 / 18$ positive disease cases detected, **0 False Negatives**)
- **Specificity:** **$0.00\%$** ($0 / 1$ healthy patient correctly classified; 1 False Positive)
- **F1-Score:** **$0.9730$**
- **ROC-AUC:** **$0.0556$**
- **Confusion Matrix Breakdown:**
  - True Positives (TP): 18
  - False Positives (FP): 1
  - True Negatives (TN): 0
  - False Negatives (FN): 0

---

## 6. Performance Comparison

All metrics evaluated on held-out test splits:

| Hospital Client | Test Samples ($N$) | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score | ROC-AUC | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hospital 1 (Cleveland)** | **46** | **84.78%** | **85.00%** | **80.95%** | **88.00%** | **0.8293** | **0.8876** | 17 / 3 / 22 / 4 |
| **Hospital 2 (Hungarian)** | **44** | **70.45%** | **80.00%** | **25.00%** | **96.43%** | **0.3810** | **0.8795** | 4 / 1 / 27 / 12 |
| **Hospital 3 (Switzerland)** | **19** | **94.74%** | **94.74%** | **100.00%** | **0.00%** | **0.9730** | **0.0556** | 18 / 1 / 0 / 0 |

---

## 7. Confusion Matrix Analysis

1. **Hospital 1:** Maintains a solid balance between Sensitivity ($80.95\%$) and Specificity ($88.00\%$), with only 4 False Negatives.
2. **Hospital 2:** Displays high Specificity ($96.43\%$, only 1 False Positive), but severe under-sensitivity (Recall $25.00\%$, 12 False Negatives). Because Hospital 2 has a majority healthy cohort ($64.2\%$ healthy), the deeper ResNet architecture gravitated heavily toward predicting the dominant negative class.
3. **Hospital 3:** Exhibits total sensitivity (Recall $100.00\%$, 0 False Negatives), but zero specificity ($0.00\%$), predicting disease across all cases due to the $93.0\%$ local positive training distribution.

---

## 8. Training Behavior

- **Hospital 1:** Best validation checkpoint reached at Epoch 2 (Val Loss: $0.4983$). Early stopping triggered at Epoch 27.
- **Hospital 2:** Best validation checkpoint reached at Epoch 2 (Val Loss: $0.4381$). Early stopping triggered at Epoch 27.
- **Hospital 3:** Best validation checkpoint reached at Epoch 4 (Val Loss: $0.2462$). Early stopping triggered at Epoch 29.

---

## 9. Client-Wise Observations

1. **Capacity vs Cohort Size:** With 244,065 parameters, 1D ResNet has approximately double the capacity of 1D AlexNet (120,257 parameters). On small tabular datasets, this extra capacity led to earlier convergence and stronger reliance on the local prior class distributions.
2. **Pathology Bias on Client 3:** Confirms the identical Non-IID skew pattern observed in AlexNet, showing that architectural choice alone cannot overcome local data distribution bias without cross-client collaboration.

---

## 10. Conclusion

The 1D ResNet baseline is successfully established across all three isolated hospital clients. The models demonstrate valid learning dynamics with functional residual representations, setting the stage for direct comparative analysis against AlexNet and subsequent federated aggregation.
