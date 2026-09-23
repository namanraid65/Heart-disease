"""
Global Model Evaluation Module
Evaluates Federated Global 1D AlexNet on client-isolated validation or held-out test partitions.
Computes client-specific diagnostic metrics (Loss, Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC)
and cross-client macro and sample-weighted aggregates.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from federated.config import FEDERATED_CLIENTS, DEVICE, INPUT_FEATURES, DROPOUT_RATE
from federated.utils import get_model_bn_state, set_model_bn_state
from models.alexnet_1d import build_alexnet_1d
from models.dataset import get_client_dataloaders


def evaluate_model_on_client_split(
    model: nn.Module,
    client_id: str,
    split: str = "val",
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Evaluates a PyTorch model on a specified client split ('val' or 'test').
    Data remains strictly isolated within the respective DataLoader.
    """
    if split not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

    model.eval()
    loaders = get_client_dataloaders(client_id=client_id, batch_size=16, shuffle_train=False)
    loader = loaders[split]

    all_preds = []
    all_probs = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = criterion(logits, targets)

            total_loss += loss.item() * features.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_prob = np.array(all_probs, dtype=float)

    n_samples = len(y_true)
    num_pos = int((y_true == 1).sum())
    num_neg = int((y_true == 0).sum())

    avg_loss = float(total_loss / n_samples) if n_samples > 0 else 0.0
    acc = float(accuracy_score(y_true, y_pred)) if n_samples > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if n_samples > 0 else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if n_samples > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if n_samples > 0 else 0.0

    # ROC-AUC: Valid only if both binary classes {0, 1} are present
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float(np.nan)
    else:
        auc = float(np.nan)

    # PR-AUC (Average Precision Score): Valid only if both classes present and at least one positive
    if len(np.unique(y_true)) > 1 and num_pos > 0:
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            pr_auc = float(np.nan)
    else:
        pr_auc = float(np.nan)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float(np.nan)

    return {
        'client_id': client_id,
        'client_name': FEDERATED_CLIENTS[client_id]['name'],
        'split': split,
        'samples': n_samples,
        'num_positives': num_pos,
        'num_negatives': num_neg,
        'loss': avg_loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1': f1,
        'roc_auc': auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def evaluate_model_on_client_test(
    model: nn.Module,
    client_id: str,
    device: str = DEVICE
) -> Dict[str, Any]:
    """Backward-compatible alias for single-client held-out test evaluation."""
    res = evaluate_model_on_client_split(model, client_id, split="test", device=device)
    res['test_samples'] = res['samples']
    return res


def evaluate_global_model_on_all_clients(
    model_or_checkpoint: Union[nn.Module, Path, str],
    split: str = "val",
    device: str = DEVICE,
    client_bn_states: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
) -> Dict[str, Any]:
    """
    Evaluates global model across all participating hospital client partitions on the chosen split ('val' or 'test').
    Under FedBN, each hospital is evaluated using its own client-specific BatchNorm statistics alongside the global shared weights.
    Computes macro-average and sample-weighted aggregate scores across clients.
    Safely handles single-class client distributions (e.g. Hospital 3 validation) without inventing values.
    """
    if isinstance(model_or_checkpoint, (str, Path)):
        checkpoint_path = Path(model_or_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_alexnet_1d(
            input_dim=checkpoint.get('input_dim', INPUT_FEATURES),
            dropout_rate=checkpoint.get('dropout_rate', DROPOUT_RATE)
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if client_bn_states is None and 'client_bn_states' in checkpoint:
            client_bn_states = checkpoint['client_bn_states']
    else:
        model = model_or_checkpoint

    client_results = {}
    sample_weights = []

    original_bn_state = get_model_bn_state(model)
    try:
        for cid in FEDERATED_CLIENTS:
            if client_bn_states and cid in client_bn_states and client_bn_states[cid] is not None:
                set_model_bn_state(model, client_bn_states[cid])
            res = evaluate_model_on_client_split(model, cid, split=split, device=device)
            client_results[cid] = res
            sample_weights.append((res, res['samples']))
    finally:
        set_model_bn_state(model, original_bn_state)

    # Compute Cross-Client Macro Average (equal weight per hospital)
    macro_loss = float(np.mean([r['loss'] for r in client_results.values()]))
    macro_acc = float(np.mean([r['accuracy'] for r in client_results.values()]))
    macro_prec = float(np.mean([r['precision'] for r in client_results.values()]))
    macro_rec = float(np.mean([r['recall'] for r in client_results.values()]))
    macro_f1 = float(np.mean([r['f1'] for r in client_results.values()]))

    # Specificity: average over clients where specificity is defined (tn + fp > 0)
    spec_vals = [r['specificity'] for r in client_results.values() if not np.isnan(r['specificity'])]
    macro_spec = float(np.mean(spec_vals)) if len(spec_vals) > 0 else float(np.nan)

    # ROC-AUC: check validity across all clients
    all_roc_auc_valid = all(not np.isnan(r['roc_auc']) for r in client_results.values())
    valid_aucs = [r['roc_auc'] for r in client_results.values() if not np.isnan(r['roc_auc'])]
    macro_auc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else float(np.nan)

    # PR-AUC: check validity across all clients
    all_pr_auc_valid = all(not np.isnan(r['pr_auc']) for r in client_results.values())
    valid_pr_aucs = [r['pr_auc'] for r in client_results.values() if not np.isnan(r['pr_auc'])]
    macro_pr_auc = float(np.mean(valid_pr_aucs)) if len(valid_pr_aucs) > 0 else float(np.nan)

    # Predeclared Selection Metric:
    # Primary: macro_roc_auc when valid for all clients.
    # Fallback: macro_f1 if any client lacks binary classes for ROC-AUC.
    selection_metric_name = "macro_roc_auc" if all_roc_auc_valid else "macro_f1"
    selection_metric_value = macro_auc if all_roc_auc_valid else macro_f1

    # Compute Sample-Weighted Aggregate
    total_samples = sum(r['samples'] for r in client_results.values())
    weighted_acc = float(sum(r['accuracy'] * r['samples'] for r in client_results.values()) / total_samples) if total_samples > 0 else 0.0
    weighted_prec = float(sum(r['precision'] * r['samples'] for r in client_results.values()) / total_samples) if total_samples > 0 else 0.0
    weighted_rec = float(sum(r['recall'] * r['samples'] for r in client_results.values()) / total_samples) if total_samples > 0 else 0.0
    weighted_f1 = float(sum(r['f1'] * r['samples'] for r in client_results.values()) / total_samples) if total_samples > 0 else 0.0

    valid_spec_samples = sum(r['samples'] for r in client_results.values() if not np.isnan(r['specificity']))
    weighted_spec = float(sum(r['specificity'] * r['samples'] for r in client_results.values() if not np.isnan(r['specificity'])) / valid_spec_samples) if valid_spec_samples > 0 else float(np.nan)

    valid_auc_samples = sum(r['samples'] for r in client_results.values() if not np.isnan(r['roc_auc']))
    weighted_auc = float(sum(r['roc_auc'] * r['samples'] for r in client_results.values() if not np.isnan(r['roc_auc'])) / valid_auc_samples) if valid_auc_samples > 0 else float(np.nan)

    valid_pr_samples = sum(r['samples'] for r in client_results.values() if not np.isnan(r['pr_auc']))
    weighted_pr_auc = float(sum(r['pr_auc'] * r['samples'] for r in client_results.values() if not np.isnan(r['pr_auc'])) / valid_pr_samples) if valid_pr_samples > 0 else float(np.nan)

    return {
        'split': split,
        'client_results': client_results,
        'macro_metrics': {
            'loss': macro_loss,
            'accuracy': macro_acc,
            'precision': macro_prec,
            'recall': macro_rec,
            'specificity': macro_spec,
            'f1': macro_f1,
            'roc_auc': macro_auc,
            'pr_auc': macro_pr_auc
        },
        'weighted_metrics': {
            'accuracy': weighted_acc,
            'precision': weighted_prec,
            'recall': weighted_rec,
            'specificity': weighted_spec,
            'f1': weighted_f1,
            'roc_auc': weighted_auc,
            'pr_auc': weighted_pr_auc,
            'total_samples': total_samples
        },
        'selection_metric_name': selection_metric_name,
        'selection_metric_value': selection_metric_value,
        'all_roc_auc_valid': all_roc_auc_valid,
        'all_pr_auc_valid': all_pr_auc_valid
    }
