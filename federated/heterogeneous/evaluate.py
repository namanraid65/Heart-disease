"""
Evaluation Utilities for Heterogeneous Federated Learning
Evaluates the global shared predictor paired with each hospital's local private encoder
across hospital validation or test splits, and computes macro and weighted metrics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List, Optional
import numpy as np

from federated.heterogeneous.server import HeterogeneousFederatedServer
from federated.heterogeneous.client import HeterogeneousHospitalClient


def evaluate_heterogeneous_system(
    server: HeterogeneousFederatedServer,
    clients: Dict[str, HeterogeneousHospitalClient],
    split: str = "val"
) -> Dict[str, Any]:
    """
    Evaluates the federated heterogeneous system:
      1. Obtains the global shared predictor parameters from the server.
      2. Dispatches them to each client.
      3. Each client evaluates using its private local encoder + global shared predictor.
      4. Computes per-hospital, macro-averaged, and sample-weighted metrics.

    Args:
        server: Central server holding the global shared predictor.
        clients: Dictionary of hospital client objects.
        split: 'val' or 'test' partition.

    Returns:
        Dictionary containing client-specific and system-level performance metrics.
    """
    global_params = server.get_global_parameters()
    client_metrics: Dict[str, Dict[str, Any]] = {}
    total_samples = 0

    for cid, client in clients.items():
        loss, n_samples, metrics = client.evaluate(
            parameters=global_params,
            config={'split': split}
        )
        client_metrics[cid] = metrics
        total_samples += n_samples

    # Compute Macro Metrics (unweighted institutional average)
    num_clients = len(clients)
    macro_loss = float(np.mean([m['loss'] for m in client_metrics.values()]))
    macro_acc = float(np.mean([m['accuracy'] for m in client_metrics.values()]))
    macro_prec = float(np.mean([m['precision'] for m in client_metrics.values()]))
    macro_rec = float(np.mean([m['recall'] for m in client_metrics.values()]))
    macro_spec = float(np.mean([m['specificity'] for m in client_metrics.values()]))
    macro_f1 = float(np.mean([m['f1'] for m in client_metrics.values()]))
    macro_auc = float(np.mean([m['roc_auc'] for m in client_metrics.values()]))
    macro_pr_auc = float(np.mean([m['pr_auc'] for m in client_metrics.values()]))

    # Compute Sample-Weighted Metrics
    weighted_loss = float(sum(m['loss'] * m['num_samples'] for m in client_metrics.values()) / total_samples) if total_samples > 0 else 0.0
    weighted_acc = float(sum(m['accuracy'] * m['num_samples'] for m in client_metrics.values()) / total_samples) if total_samples > 0 else 0.0
    weighted_f1 = float(sum(m['f1'] * m['num_samples'] for m in client_metrics.values()) / total_samples) if total_samples > 0 else 0.0
    weighted_auc = float(sum(m['roc_auc'] * m['num_samples'] for m in client_metrics.values()) / total_samples) if total_samples > 0 else 0.0
    weighted_pr_auc = float(sum(m['pr_auc'] * m['num_samples'] for m in client_metrics.values()) / total_samples) if total_samples > 0 else 0.0

    return {
        'split': split,
        'client_metrics': client_metrics,
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
            'loss': weighted_loss,
            'accuracy': weighted_acc,
            'f1': weighted_f1,
            'roc_auc': weighted_auc,
            'pr_auc': weighted_pr_auc
        },
        'total_samples': total_samples
    }
