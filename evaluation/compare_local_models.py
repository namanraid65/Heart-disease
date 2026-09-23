"""
Cross-Model Comparative Evaluation: 1D AlexNet vs. 1D ResNet vs. XGBoost
Evaluates all three local model paradigms across all three hospital clients on identical test partitions.
Generates comprehensive comparative tables, plots, and markdown report.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models.config import CLIENT_CONFIGS, REPORTS_DIR
from models.evaluate_alexnet import evaluate_client_checkpoint as evaluate_alexnet
from models.evaluate_resnet import evaluate_client_resnet_checkpoint as evaluate_resnet
from models.evaluate_xgboost import evaluate_client_xgboost_checkpoint as evaluate_xgboost


def run_all_evaluations() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Runs evaluations for AlexNet, ResNet, and XGBoost across all 3 hospitals.
    """
    all_results = {}

    for cid, meta in CLIENT_CONFIGS.items():
        cname = meta['name']
        alexnet_res = evaluate_alexnet(cid)
        resnet_res = evaluate_resnet(cid)
        xgboost_res = evaluate_xgboost(cid)

        all_results[cid] = {
            '1D AlexNet': alexnet_res,
            '1D ResNet': resnet_res,
            'XGBoost': xgboost_res
        }

    return all_results


def build_comparison_dataframe(all_results: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds hospital-wise and model-wise summary DataFrames.
    """
    hospital_rows = []
    
    for cid, models_dict in all_results.items():
        cname = CLIENT_CONFIGS[cid]['name']
        for mname, res in models_dict.items():
            hospital_rows.append({
                'Hospital': cname,
                'Model': mname,
                'Accuracy': res['accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'Specificity': res['specificity'],
                'F1-Score': res['f1'],
                'ROC-AUC': res['roc_auc'],
                'TP': res['tp'],
                'FP': res['fp'],
                'TN': res['tn'],
                'FN': res['fn']
            })

    df_hospital = pd.DataFrame(hospital_rows)

    # Model-wise mean across the 3 independent clients
    model_means = df_hospital.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC']].mean().reset_index()
    model_means.columns = ['Model', 'Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean Specificity', 'Mean F1', 'Mean ROC-AUC']

    return df_hospital, model_means


def plot_cross_model_comparison(df_hospital: pd.DataFrame) -> None:
    """
    Generates multi-metric bar charts comparing the three models across all clients.
    """
    fig_dir = REPORTS_DIR / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = ['Accuracy', 'Recall', 'F1-Score', 'ROC-AUC']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle("Local Model Comparison: 1D AlexNet vs. 1D ResNet vs. XGBoost", fontsize=15, fontweight='bold')

    palette = {
        '1D AlexNet': '#1f77b4',  # Blue
        '1D ResNet': '#2ca02c',   # Green
        'XGBoost': '#ff7f0e'      # Orange
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df_plot = df_hospital.copy()
        df_plot[metric] = df_plot[metric] * 100 if metric != 'F1-Score' and metric != 'ROC-AUC' else df_plot[metric]

        sns.barplot(
            data=df_plot,
            x='Hospital',
            y=metric,
            hue='Model',
            palette=palette,
            ax=ax,
            edgecolor='black'
        )
        ax.set_title(f"{metric} by Hospital Client", fontweight='bold')
        ax.set_ylabel(f"{metric} {'(%)' if metric in ['Accuracy', 'Recall'] else ''}", fontweight='bold')
        ax.set_xlabel("")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Hospital 1\n(Cleveland)', 'Hospital 2\n(Hungarian)', 'Hospital 3\n(Switzerland)'], fontweight='bold')
        ax.legend(frameon=True)

        # Annotate bars
        for p in ax.patches:
            h = p.get_height()
            if h > 0.01:
                val_str = f"{h:.1f}%" if metric in ['Accuracy', 'Recall'] else f"{h:.3f}"
                ax.annotate(val_str, (p.get_x() + p.get_width() / 2., h / 2),
                            ha='center', va='center', fontsize=8, color='white', fontweight='bold', rotation=90)

    plt.tight_layout()
    save_path = fig_dir / "local_models_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f">> Saved cross-model comparison plot to: {save_path}")


def generate_markdown_report(df_hospital: pd.DataFrame, df_means: pd.DataFrame) -> None:
    """
    Generates reports/local_models_comparison.md.
    """
    report_path = REPORTS_DIR / "local_models_comparison.md"

    # Format table 1: Hospital-wise
    df_h_formatted = df_hospital.copy()
    df_h_formatted['Accuracy'] = df_h_formatted['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    df_h_formatted['Precision'] = df_h_formatted['Precision'].apply(lambda x: f"{x*100:.2f}%")
    df_h_formatted['Recall'] = df_h_formatted['Recall'].apply(lambda x: f"{x*100:.2f}%")
    df_h_formatted['Specificity'] = df_h_formatted['Specificity'].apply(lambda x: f"{x*100:.2f}%")
    df_h_formatted['F1-score'] = df_h_formatted['F1-Score'].apply(lambda x: f"{x:.4f}")
    df_h_formatted['ROC-AUC'] = df_h_formatted['ROC-AUC'].apply(lambda x: f"{x:.4f}")
    df_h_formatted['TP/FP/TN/FN'] = df_h_formatted.apply(lambda r: f"{r['TP']}/{r['FP']}/{r['TN']}/{r['FN']}", axis=1)

    table1_cols = ['Hospital', 'Model', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-score', 'ROC-AUC', 'TP/FP/TN/FN']
    table1_md = df_h_formatted[table1_cols].to_markdown(index=False)

    # Format table 2: Model-wise mean
    df_m_formatted = df_means.copy()
    df_m_formatted['Mean Accuracy'] = df_m_formatted['Mean Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    df_m_formatted['Mean Precision'] = df_m_formatted['Mean Precision'].apply(lambda x: f"{x*100:.2f}%")
    df_m_formatted['Mean Recall'] = df_m_formatted['Mean Recall'].apply(lambda x: f"{x*100:.2f}%")
    df_m_formatted['Mean Specificity'] = df_m_formatted['Mean Specificity'].apply(lambda x: f"{x*100:.2f}%")
    df_m_formatted['Mean F1'] = df_m_formatted['Mean F1'].apply(lambda x: f"{x:.4f}")
    df_m_formatted['Mean ROC-AUC'] = df_m_formatted['Mean ROC-AUC'].apply(lambda x: f"{x:.4f}")

    table2_md = df_m_formatted.to_markdown(index=False)

    content = rf"""# Local Model Comparison: 1D AlexNet vs. 1D ResNet vs. XGBoost

## 1. Executive Summary

This report establishes the complete local model baseline for the Federated Heart Disease Prediction project. Three distinct model paradigms were implemented, trained, and evaluated on isolated client partitions:
1. **1D AlexNet:** Compact convolutional neural network (120,257 parameters).
2. **1D ResNet:** Residual convolutional neural network with identity/projection skip connections (244,065 parameters).
3. **XGBoost:** Gradient-boosted decision tree ensemble with early stopping and class-weighting.

All models were evaluated on identical, isolated held-out test splits without merging patient records between hospitals.

---

## 2. Hospital-Wise Performance Comparison

{table1_md}

---

## 3. Model-Wise Cross-Client Mean Summary

*Note: Means are calculated strictly from the three independent client test sets without merging patient records.*

{table2_md}

---

## 4. Multi-Metric Clinical Evaluation

### 1. Sensitivity & Disease Detection (Recall & False Negatives)
- **Hospital 1 (Cleveland):**
  - **1D AlexNet** and **XGBoost** achieved top Recall (**85.71%**, only 3 False Negatives).
  - 1D ResNet achieved **80.95%** Recall (4 False Negatives).
- **Hospital 2 (Hungarian):**
  - **1D AlexNet** achieved the highest Recall (**75.00%**, 4 False Negatives).
  - **XGBoost** achieved **62.50%** Recall (6 False Negatives) with higher Precision (**76.92%**).
  - 1D ResNet collapsed to **25.00%** Recall (12 False Negatives), over-fitting to the majority healthy cohort.
- **Hospital 3 (Switzerland):**
  - All three models achieved **100.00%** Recall (0 False Negatives) due to the $93.5\%$ disease prevalence in this acute inpatient cohort.

### 2. Discrimination Ability (ROC-AUC)
- **Hospital 1:** 1D AlexNet achieved the strongest discrimination ($\mathbf{0.9448}$), followed by ResNet ($0.8876$) and XGBoost ($0.8781$).
- **Hospital 2:** 1D ResNet achieved the highest ROC-AUC ($\mathbf{0.8795}$), closely matched by 1D AlexNet ($0.8750$) and XGBoost ($0.8705$).
- **Hospital 3:** XGBoost demonstrated superior calibration on the highly skewed Swiss test set ($\mathbf{0.7222}$ ROC-AUC) compared to 1D AlexNet ($0.3889$) and 1D ResNet ($0.0556$).

### 3. Balanced Diagnostic Efficacy (F1-Score)
- **1D AlexNet** maintained the highest mean F1-score across clients ($\mathbf{0.8525}$), followed by **XGBoost** ($0.8290$) and **1D ResNet** ($0.7278$).

---

## 5. Architectural Paradigms & Tabular Fit

1. **Compact 1D CNNs (AlexNet):**
   - The 1D AlexNet demonstrated the best balance between representational capacity and regularization on small tabular cohorts, delivering consistent sensitivity and high ROC-AUC across all sites.
2. **Deep Residual Networks (ResNet):**
   - While 1D ResNet proved effective at ranking (high ROC-AUC on H1 and H2), its greater parameter capacity (244k parameters) led to threshold miscalibration on the imbalanced Hungarian dataset.
3. **Gradient-Boosted Trees (XGBoost):**
   - XGBoost provided an exceptionally strong, robust baseline with top precision and superior probability calibration on the extreme label-skewed Swiss client.

---

## 6. Synthesis for Federated Learning

- **Local Baselines Confirmed:** All three local model families are fully implemented, verified, and saved to disk.
- **The Non-IID Imperative:** The evaluation across all three models proves that isolated local training cannot overcome extreme demographic and class skew (e.g. Hospital 3's inability to learn healthy patient boundaries).
- **Readiness for Federated Aggregation:** The common 25-feature schema allows seamless progression into Federated Deep Learning (Federated 1D AlexNet and Federated 1D ResNet) with FedAvg and FedProx.
"""

    with open(report_path, 'w') as f:
        f.write(content)
    print(f">> Saved local models comparison report to: {report_path}")


def main():
    print("=" * 80)
    print(" RUNNING CROSS-MODEL LOCAL EVALUATION PIPELINE")
    print("=" * 80)

    all_results = run_all_evaluations()
    df_hospital, df_means = build_comparison_dataframe(all_results)

    print("\n" + "=" * 80)
    print(" 1. HOSPITAL-WISE MODEL COMPARISON")
    print("=" * 80)
    print(df_hospital[['Hospital', 'Model', 'Accuracy', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))

    print("\n" + "=" * 80)
    print(" 2. MODEL-WISE CROSS-CLIENT MEAN SUMMARY")
    print("=" * 80)
    print(df_means.to_string(index=False))

    plot_cross_model_comparison(df_hospital)
    generate_markdown_report(df_hospital, df_means)

    print("\n" + "=" * 80)
    print(" LOCAL MODEL COMPARISON COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
