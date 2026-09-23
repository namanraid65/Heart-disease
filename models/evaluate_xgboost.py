"""
Evaluation and Visualization Module for Local XGBoost Models
Evaluates local XGBoost checkpoints on held-out test sets, computes clinical diagnostic metrics,
generates confusion matrix heatmaps, ROC curves, and feature importance bar plots.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

from models.config import (
    CLIENT_CONFIGS,
    CHECKPOINTS_DIR,
    XGBOOST_FIGURES_DIR
)
from models.xgboost_model import LocalXGBoostModel
from models.train_xgboost import load_client_tabular_data
from preprocessing.feature_schema import PROCESSED_FEATURE_NAMES


def evaluate_client_xgboost_checkpoint(
    client_id: str
) -> Dict[str, Any]:
    """
    Loads saved XGBoost model for a client and evaluates it on its held-out test partition.
    """
    client_meta = CLIENT_CONFIGS[client_id]
    client_name = client_meta['name']
    checkpoint_name = client_meta.get('xgboost_checkpoint', f"{client_id}_xgboost.json")
    checkpoint_file = CHECKPOINTS_DIR / checkpoint_name

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"XGBoost checkpoint not found at: {checkpoint_file}")

    # Load Model
    model = LocalXGBoostModel()
    model.load_model(checkpoint_file)

    # Load held-out test data
    splits = load_client_tabular_data(client_id)
    X_test, y_test = splits['test']

    y_true = y_test.to_numpy(dtype=int)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    total_test_samples = len(y_true)
    correct_predictions = int((y_true == y_pred).sum())
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Feature importances
    feat_imp = model.get_feature_importances(PROCESSED_FEATURE_NAMES)

    return {
        'client_id': client_id,
        'client_name': client_name,
        'model_type': 'XGBoost',
        'total_samples': total_test_samples,
        'correct_predictions': correct_predictions,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1': f1,
        'roc_auc': auc,
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'y_true': y_true,
        'y_prob': y_prob,
        'feature_importances': feat_imp
    }


def plot_xgboost_confusion_matrix(eval_res: dict):
    """
    Plots and saves confusion matrix heatmap for XGBoost.
    """
    XGBOOST_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = eval_res['confusion_matrix']
    client_id = eval_res['client_id']
    client_name = eval_res['client_name']

    plt.figure(figsize=(6, 5))
    plt.title(f"Test Confusion Matrix - {client_name} (XGBoost)", fontsize=12, fontweight='bold', pad=12)

    labels = ['Healthy (0)', 'Disease (1)']
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Oranges',
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.xlabel('Predicted Diagnosis', fontweight='bold')
    plt.ylabel('True Clinical Status', fontweight='bold')

    acc_text = f"Accuracy: {eval_res['accuracy']*100:.1f}%\nRecall: {eval_res['recall']*100:.1f}%\nF1-Score: {eval_res['f1']:.3f}"
    plt.figtext(0.5, -0.05, acc_text, ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = XGBOOST_FIGURES_DIR / f"{client_id}_xgboost_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_xgboost_roc_curve(eval_res: dict):
    """
    Plots and saves ROC curve for XGBoost on client test set.
    """
    XGBOOST_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    client_id = eval_res['client_id']
    client_name = eval_res['client_name']
    y_true = eval_res['y_true']
    y_prob = eval_res['y_prob']
    auc_val = eval_res['roc_auc']

    plt.figure(figsize=(6, 5.5))
    
    # Check if both classes exist in y_true
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, color='#d95f02', linewidth=2.5, label=f'XGBoost (AUC = {auc_val:.3f})')
    else:
        plt.text(0.5, 0.5, "Only 1 class present in test set", ha='center', va='center', fontsize=11)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1.5, label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontweight='bold')
    plt.title(f"ROC Curve - {client_name} (XGBoost)", fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = XGBOOST_FIGURES_DIR / f"{client_id}_xgboost_roc.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_xgboost_feature_importance(eval_res: dict):
    """
    Plots top-10 feature importances for local XGBoost model.
    """
    XGBOOST_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    client_id = eval_res['client_id']
    client_name = eval_res['client_name']
    feat_imp = eval_res['feature_importances'].head(10)

    plt.figure(figsize=(8, 4.5))
    bars = plt.barh(feat_imp.index[::-1], feat_imp.values[::-1], color='#e6550d', edgecolor='black', alpha=0.85)
    plt.xlabel('Relative Feature Importance (Gain / Weight)', fontweight='bold')
    plt.title(f"Top 10 Feature Importances - {client_name}", fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    for bar in bars:
        w = bar.get_width()
        if w > 0.001:
            plt.text(w + 0.005, bar.get_y() + bar.get_height()/2., f"{w:.3f}", ha='left', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    save_path = XGBOOST_FIGURES_DIR / f"{client_id}_xgboost_feature_importance.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path.name}")


def evaluate_all_local_xgboosts() -> Dict[str, Dict[str, Any]]:
    """
    Evaluates all local XGBoost models and generates all visual artifacts.
    """
    print("=" * 80)
    print(" EVALUATING LOCAL XGBOOST MODELS ON TEST PARTITIONS")
    print("=" * 80)

    results = {}
    for cid in CLIENT_CONFIGS:
        res = evaluate_client_xgboost_checkpoint(cid)
        results[cid] = res

        print(f"\n--- {res['client_name']} (XGBoost) ---")
        print(f"  Test Samples:          {res['total_samples']} (Correct: {res['correct_predictions']}/{res['total_samples']})")
        print(f"  Accuracy:              {res['accuracy']*100:.2f}%")
        print(f"  Precision:             {res['precision']*100:.2f}%")
        print(f"  Recall (Sensitivity):  {res['recall']*100:.2f}%")
        print(f"  Specificity:           {res['specificity']*100:.2f}%")
        print(f"  F1-Score:              {res['f1']:.4f}")
        print(f"  ROC-AUC:               {res['roc_auc']:.4f}")
        print(f"  Confusion Matrix:      [TN={res['tn']}, FP={res['fp']}, FN={res['fn']}, TP={res['tp']}]")

        plot_xgboost_confusion_matrix(res)
        plot_xgboost_roc_curve(res)
        plot_xgboost_feature_importance(res)

    return results


def print_xgboost_comparison_table(results: dict):
    """
    Prints summary comparison table of all local XGBoost models.
    """
    print("\n" + "=" * 80)
    print(" LOCAL XGBOOST MODEL PERFORMANCE COMPARISON")
    print("=" * 80)

    table_data = []
    for cid, r in results.items():
        table_data.append({
            'Hospital Client': r['client_name'],
            'Test N': r['total_samples'],
            'Accuracy': f"{r['accuracy']*100:.1f}%",
            'Precision': f"{r['precision']*100:.1f}%",
            'Recall': f"{r['recall']*100:.1f}%",
            'Specificity': f"{r['specificity']*100:.1f}%",
            'F1-Score': f"{r['f1']:.4f}",
            'ROC-AUC': f"{r['roc_auc']:.4f}",
            'TP/FP/TN/FN': f"{r['tp']}/{r['fp']}/{r['tn']}/{r['fn']}"
        })

    df_res = pd.DataFrame(table_data)
    print(df_res.to_string(index=False))


if __name__ == '__main__':
    results = evaluate_all_local_xgboosts()
    print_xgboost_comparison_table(results)
