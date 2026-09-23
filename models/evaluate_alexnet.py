"""
Evaluation and Visualization Module for Local 1D AlexNet Models
Evaluates local checkpoints on isolated test sets, computes clinical diagnostic metrics,
generates training loss/metric curves, and creates confusion matrix visualizations.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from models.config import (
    CLIENT_CONFIGS,
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    DEVICE
)
from models.alexnet_1d import AlexNet1D, build_alexnet_1d
from models.dataset import get_client_dataloaders


def evaluate_client_checkpoint(
    client_id: str,
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Loads saved checkpoint for a client and evaluates it on its held-out test partition.
    """
    client_meta = CLIENT_CONFIGS[client_id]
    client_name = client_meta['name']
    checkpoint_name = client_meta.get('alexnet_checkpoint', 'hospital_1_alexnet.pt')
    checkpoint_file = CHECKPOINTS_DIR / checkpoint_name

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    input_dim = checkpoint.get('input_dim', 25)
    history = checkpoint.get('history', {})
    best_epoch = checkpoint.get('best_epoch', 0)

    model = build_alexnet_1d(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load held-out test data
    loaders = get_client_dataloaders(client_id=client_id, batch_size=16, shuffle_train=False)
    test_loader = loaders['test']

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.numpy().flatten())

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_prob = np.array(all_probs, dtype=float)

    # Calculate metrics
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

    return {
        'client_id': client_id,
        'client_name': client_name,
        'best_epoch': best_epoch,
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
        'history': history
    }


def plot_training_curves(history: dict, client_id: str, client_name: str, best_epoch: int):
    """
    Plots and saves training loss, validation loss, accuracy, and ROC-AUC vs epochs.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"1D AlexNet Training Dynamics - {client_name}", fontsize=13, fontweight='bold')

    # Loss curve
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#d62728', linewidth=2, linestyle='--')
    ax1.axvline(best_epoch, color='green', linestyle=':', label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('BCE Loss', fontweight='bold')
    ax1.set_title('Loss vs Epoch', fontweight='bold')
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Accuracy / AUC curve
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], label='Train Acc (%)', color='#1f77b4', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], label='Val Acc (%)', color='#d62728', linewidth=2, linestyle='--')
    ax2.plot(epochs, [a * 100 for a in history['val_auc']], label='Val ROC-AUC (%)', color='#2ca02c', linewidth=1.5, linestyle='-.')
    ax2.axvline(best_epoch, color='green', linestyle=':', label=f'Best Epoch ({best_epoch})')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontweight='bold')
    ax2.set_title('Accuracy & ROC-AUC vs Epoch', fontweight='bold')
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = FIGURES_DIR / f"{client_id}_alexnet_training.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_confusion_matrix(eval_res: dict):
    """
    Plots and saves confusion matrix heatmap for a client.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = eval_res['confusion_matrix']
    client_id = eval_res['client_id']
    client_name = eval_res['client_name']

    plt.figure(figsize=(6, 5))
    plt.title(f"Test Confusion Matrix - {client_name}", fontsize=12, fontweight='bold', pad=12)

    labels = ['Healthy (0)', 'Disease (1)']
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.xlabel('Predicted Diagnosis', fontweight='bold')
    plt.ylabel('True Clinical Status', fontweight='bold')

    # Annotate summary stats on plot
    acc_text = f"Accuracy: {eval_res['accuracy']*100:.1f}%\nRecall: {eval_res['recall']*100:.1f}%\nF1-Score: {eval_res['f1']:.3f}"
    plt.figtext(0.5, -0.05, acc_text, ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = FIGURES_DIR / f"{client_id}_alexnet_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def evaluate_all_local_alexnets() -> Dict[str, Dict[str, Any]]:
    """
    Evaluates all local 1D AlexNet models and generates all visual artifacts.
    """
    print("=" * 80)
    print(" EVALUATING LOCAL 1D ALEXNET MODELS ON TEST PARTITIONS")
    print("=" * 80)

    results = {}
    for cid in CLIENT_CONFIGS:
        res = evaluate_client_checkpoint(cid)
        results[cid] = res

        print(f"\n--- {res['client_name']} ---")
        print(f"  Best Validation Epoch: {res['best_epoch']}")
        print(f"  Test Samples:          {res['total_samples']} (Correct: {res['correct_predictions']}/{res['total_samples']})")
        print(f"  Accuracy:              {res['accuracy']*100:.2f}%")
        print(f"  Precision:             {res['precision']*100:.2f}%")
        print(f"  Recall (Sensitivity):  {res['recall']*100:.2f}%")
        print(f"  Specificity:           {res['specificity']*100:.2f}%")
        print(f"  F1-Score:              {res['f1']:.4f}")
        print(f"  ROC-AUC:               {res['roc_auc']:.4f}")
        print(f"  Confusion Matrix:      [TN={res['tn']}, FP={res['fp']}, FN={res['fn']}, TP={res['tp']}]")

        # Plot training curves and confusion matrix
        plot_training_curves(res['history'], cid, res['client_name'], res['best_epoch'])
        plot_confusion_matrix(res)

    return results


def print_comparison_table(results: dict):
    """
    Prints a formatted summary comparison table of all local client models.
    """
    print("\n" + "=" * 80)
    print(" LOCAL 1D ALEXNET MODEL PERFORMANCE COMPARISON")
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
    results = evaluate_all_local_alexnets()
    print_comparison_table(results)
