"""
Global 1D ResNet Evaluation Module
Evaluates Federated Global 1D ResNet independently on client-isolated validation or held-out test sets.
Computes comprehensive diagnostic metrics (Loss, Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC)
and aggregate multi-center summaries.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
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

from federated.config import (
    FEDERATED_CLIENTS,
    INPUT_FEATURES,
    RESNET_DROPOUT_RATE,
    DEVICE
)
from federated.utils import get_model_bn_state, set_model_bn_state
from models.resnet_1d import ResNet1D, build_resnet_1d
from models.dataset import get_client_dataloaders


def evaluate_global_resnet_on_client(
    model: nn.Module,
    client_id: str,
    split: str = "val",
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Evaluates global ResNet model on a single client's partition ('val' or 'test').
    Data remains strictly isolated within the respective client DataLoader.
    """
    if split not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

    client_meta = FEDERATED_CLIENTS[client_id]
    client_name = client_meta['name']

    loaders = get_client_dataloaders(client_id=client_id, batch_size=16, shuffle_train=False)
    loader = loaders[split]

    model.eval()
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
            all_targets.extend(targets.numpy().flatten())

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_prob = np.array(all_probs, dtype=float)

    total_samples = len(y_true)
    num_pos = int((y_true == 1).sum())
    num_neg = int((y_true == 0).sum())

    avg_loss = float(total_loss / total_samples) if total_samples > 0 else 0.0
    acc = float(accuracy_score(y_true, y_pred)) if total_samples > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if total_samples > 0 else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if total_samples > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if total_samples > 0 else 0.0

    # ROC-AUC: safely handled
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float(np.nan)
    else:
        auc = float(np.nan)

    # PR-AUC: safely handled
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
        'client_name': client_name,
        'split': split,
        'samples': total_samples,
        'total_samples': total_samples,
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


def evaluate_global_resnet_all_clients(
    model_or_checkpoint: Union[nn.Module, Path, str],
    split: str = "val",
    device: str = DEVICE,
    client_bn_states: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
) -> Dict[str, Any]:
    """
    Evaluates global ResNet across all 3 client partitions on split ('val' or 'test').
    Under FedBN, each hospital is evaluated using its own client-specific BatchNorm statistics alongside the global shared weights.
    Computes both Macro-Averaged and Sample-Weighted aggregate metrics.
    """
    if isinstance(model_or_checkpoint, (str, Path)):
        checkpoint_path = Path(model_or_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_resnet_1d(
            input_dim=checkpoint.get('input_dim', INPUT_FEATURES),
            dropout_rate=checkpoint.get('dropout_rate', RESNET_DROPOUT_RATE)
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if client_bn_states is None and 'client_bn_states' in checkpoint:
            client_bn_states = checkpoint['client_bn_states']
    else:
        model = model_or_checkpoint

    client_results = {}
    total_samples = 0

    original_bn_state = get_model_bn_state(model)
    try:
        for cid in FEDERATED_CLIENTS:
            if client_bn_states and cid in client_bn_states and client_bn_states[cid] is not None:
                set_model_bn_state(model, client_bn_states[cid])
            res = evaluate_global_resnet_on_client(model, cid, split=split, device=device)
            client_results[cid] = res
            total_samples += res['samples']
    finally:
        set_model_bn_state(model, original_bn_state)

    # Macro-Averaged Summary
    macro_loss = float(np.mean([r['loss'] for r in client_results.values()]))
    macro_acc = float(np.mean([r['accuracy'] for r in client_results.values()]))
    macro_prec = float(np.mean([r['precision'] for r in client_results.values()]))
    macro_rec = float(np.mean([r['recall'] for r in client_results.values()]))
    macro_f1 = float(np.mean([r['f1'] for r in client_results.values()]))

    spec_vals = [r['specificity'] for r in client_results.values() if not np.isnan(r['specificity'])]
    macro_spec = float(np.mean(spec_vals)) if len(spec_vals) > 0 else float(np.nan)

    all_roc_auc_valid = all(not np.isnan(r['roc_auc']) for r in client_results.values())
    valid_aucs = [r['roc_auc'] for r in client_results.values() if not np.isnan(r['roc_auc'])]
    macro_auc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else float(np.nan)

    all_pr_auc_valid = all(not np.isnan(r['pr_auc']) for r in client_results.values())
    valid_pr_aucs = [r['pr_auc'] for r in client_results.values() if not np.isnan(r['pr_auc'])]
    macro_pr_auc = float(np.mean(valid_pr_aucs)) if len(valid_pr_aucs) > 0 else float(np.nan)

    # Predeclared Selection Metric
    selection_metric_name = "macro_roc_auc" if all_roc_auc_valid else "macro_f1"
    selection_metric_value = macro_auc if all_roc_auc_valid else macro_f1

    # Sample-Weighted Summary
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
        'total_test_samples': total_samples,
        'selection_metric_name': selection_metric_name,
        'selection_metric_value': selection_metric_value,
        'all_roc_auc_valid': all_roc_auc_valid,
        'all_pr_auc_valid': all_pr_auc_valid
    }


if __name__ == '__main__':
    from federated.resnet_server import FederatedResNetServer
    server = FederatedResNetServer()
    eval_res = evaluate_global_resnet_all_clients(server.global_model, split="val")
    print("ResNet Initial Validation Evaluation (Round 0):")
    for cid, r in eval_res['client_results'].items():
        print(f"  {r['client_name']}: Acc={r['accuracy']*100:.1f}%, F1={r['f1']:.4f}")
    print(f"Macro Validation Accuracy: {eval_res['macro_metrics']['accuracy']*100:.2f}%")

