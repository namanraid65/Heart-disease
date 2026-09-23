"""
Central Results Store Builder (Phase 10)
Loads all genuinely completed checkpoints, performs untouched test evaluation with
confusion matrix auditing, calibration analysis (Brier score, ECE), and communication accounting,
then serializes records to results/experiment_results.jsonl.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, brier_score_loss
)

from models.alexnet_1d import build_alexnet_1d
from models.resnet_1d import build_resnet_1d
from models.xgboost_model import LocalXGBoostModel
from models.dataset import get_client_dataloaders
from models.heterogeneous.composite import HeterogeneousCompositeModel, build_heterogeneous_model
from models.heterogeneous.predictor import SharedPredictor
from preprocessing.heterogeneous_schema import get_client_schema
from federated.heterogeneous.config import HETEROGENEOUS_CLIENTS


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error across equal-width probability bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(targets)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = float(np.mean(targets[in_bin]))
            bin_conf = float(np.mean(probs[in_bin]))
            ece += (bin_size / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_metrics_from_preds(targets: np.ndarray, probs: np.ndarray) -> dict:
    """Computes full suite of classification and calibration metrics."""
    preds = (probs >= 0.5).astype(int)
    n = len(targets)
    pos = int(np.sum(targets == 1))
    neg = int(np.sum(targets == 0))

    tn, fp, fn, tp = 0, 0, 0, 0
    if n > 0:
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Assert matrix consistency
    assert (tp + tn + fp + fn) == n, f"Matrix sum {tp+tn+fp+fn} != N {n}"
    assert (tp + fn) == pos, f"TP+FN {tp+fn} != Pos {pos}"
    assert (tn + fp) == neg, f"TN+FP {tn+fp} != Neg {neg}"

    acc = float(accuracy_score(targets, preds)) if n > 0 else 0.0
    prec = float(precision_score(targets, preds, zero_division=0)) if n > 0 else 0.0
    rec = float(recall_score(targets, preds, zero_division=0)) if n > 0 else 0.0
    f1 = float(f1_score(targets, preds, zero_division=0)) if n > 0 else 0.0

    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else None

    # ROC-AUC with single-class safety
    if pos > 0 and neg > 0:
        roc_auc = float(roc_auc_score(targets, probs))
    else:
        roc_auc = None

    # PR-AUC
    if pos > 0:
        p_curve, r_curve, _ = precision_recall_curve(targets, probs)
        pr_auc = float(auc(r_curve, p_curve))
    else:
        pr_auc = None

    brier = float(brier_score_loss(targets, probs)) if n > 0 else 0.0
    ece = compute_ece(probs, targets)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'sensitivity': rec,
        'specificity': spec,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'ece': ece,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'sample_count': n,
        'positive_count': pos,
        'negative_count': neg
    }


def aggregate_macro(client_results: dict) -> dict:
    """Computes unweighted macro hospital average."""
    metrics = ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1', 'roc_auc', 'pr_auc', 'brier_score', 'ece']
    agg = {}
    for m in metrics:
        valid_vals = [c[m] for c in client_results.values() if c[m] is not None and not np.isnan(c[m])]
        agg[m] = float(np.mean(valid_vals)) if valid_vals else None

    agg['tp'] = sum(c['tp'] for c in client_results.values())
    agg['tn'] = sum(c['tn'] for c in client_results.values())
    agg['fp'] = sum(c['fp'] for c in client_results.values())
    agg['fn'] = sum(c['fn'] for c in client_results.values())
    agg['sample_count'] = sum(c['sample_count'] for c in client_results.values())
    agg['positive_count'] = sum(c['positive_count'] for c in client_results.values())
    agg['negative_count'] = sum(c['negative_count'] for c in client_results.values())
    return agg


def aggregate_sample_weighted(client_results: dict) -> dict:
    """Computes sample-weighted average across all patient test records."""
    total_n = sum(c['sample_count'] for c in client_results.values())
    if total_n == 0:
        return {}

    metrics = ['accuracy', 'precision', 'recall', 'sensitivity', 'f1', 'brier_score', 'ece']
    agg = {}
    for m in metrics:
        val = sum(c[m] * c['sample_count'] for c in client_results.values() if c[m] is not None) / total_n
        agg[m] = float(val)

    # For specificity, weight by negative count
    total_neg = sum(c['negative_count'] for c in client_results.values())
    if total_neg > 0:
        spec_val = sum(c['specificity'] * c['negative_count'] for c in client_results.values() if c['specificity'] is not None) / total_neg
        agg['specificity'] = float(spec_val)
    else:
        agg['specificity'] = None

    # Weighted ROC-AUC & PR-AUC
    valid_auc = [(c['roc_auc'], c['sample_count']) for c in client_results.values() if c['roc_auc'] is not None]
    if valid_auc:
        total_auc_w = sum(w for _, w in valid_auc)
        agg['roc_auc'] = float(sum(v * w for v, w in valid_auc) / total_auc_w)
    else:
        agg['roc_auc'] = None

    valid_prauc = [(c['pr_auc'], c['sample_count']) for c in client_results.values() if c['pr_auc'] is not None]
    if valid_prauc:
        total_pr_w = sum(w for _, w in valid_prauc)
        agg['pr_auc'] = float(sum(v * w for v, w in valid_prauc) / total_pr_w)
    else:
        agg['pr_auc'] = None

    agg['tp'] = sum(c['tp'] for c in client_results.values())
    agg['tn'] = sum(c['tn'] for c in client_results.values())
    agg['fp'] = sum(c['fp'] for c in client_results.values())
    agg['fn'] = sum(c['fn'] for c in client_results.values())
    agg['sample_count'] = total_n
    agg['positive_count'] = sum(c['positive_count'] for c in client_results.values())
    agg['negative_count'] = total_neg
    return agg


def build_and_export_results():
    output_path = Path('results/experiment_results.jsonl')
    records = []

    # 1. Load test data for all clients
    test_loaders = {}
    test_targets = {}
    for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
        loaders = get_client_dataloaders(cid, batch_size=16, shuffle_train=False)
        test_loaders[cid] = loaders['test']
        all_y = []
        for _, y in loaders['test']:
            all_y.extend(y.numpy().flatten())
        test_targets[cid] = np.array(all_y, dtype=int)

    # -------------------------------------------------------------------------
    # LOCAL ALEXNET
    # -------------------------------------------------------------------------
    client_res = {}
    for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
        ckpt_path = Path(f'models/checkpoints/{cid}_alexnet.pt')
        if ckpt_path.exists():
            payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model = build_alexnet_1d(input_dim=payload.get('input_dim', 25))
            model.load_state_dict(payload['model_state_dict'])
            model.eval()
            probs = []
            with torch.no_grad():
                for x, _ in test_loaders[cid]:
                    out = model(x)
                    p = torch.sigmoid(out).numpy().flatten()
                    probs.extend(p)
            client_res[cid] = compute_metrics_from_preds(test_targets[cid], np.array(probs))

    if len(client_res) == 3:
        macro_res = aggregate_macro(client_res)
        wgt_res = aggregate_sample_weighted(client_res)
        alexnet_params = sum(p.numel() for p in model.parameters())

        for cid, m in client_res.items():
            records.append({
                'experiment_id': 'local_alexnet',
                'seed': 42,
                'hospital': cid,
                'aggregation_level': 'hospital',
                'parameters': {'shared_parameters': alexnet_params, 'private_encoder_parameters': 0},
                'communication': {'bytes_per_round': 0, 'total_bytes_communicated': 0},
                'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
                'metrics': m
            })
        records.append({
            'experiment_id': 'local_alexnet',
            'seed': 42,
            'hospital': 'ALL',
            'aggregation_level': 'macro',
            'parameters': {'shared_parameters': alexnet_params, 'private_encoder_parameters': 0},
            'communication': {'bytes_per_round': 0, 'total_bytes_communicated': 0},
            'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
            'metrics': macro_res
        })
        records.append({
            'experiment_id': 'local_alexnet',
            'seed': 42,
            'hospital': 'ALL',
            'aggregation_level': 'sample_weighted',
            'parameters': {'shared_parameters': alexnet_params, 'private_encoder_parameters': 0},
            'communication': {'bytes_per_round': 0, 'total_bytes_communicated': 0},
            'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
            'metrics': wgt_res
        })

    # -------------------------------------------------------------------------
    # LOCAL RESNET
    # -------------------------------------------------------------------------
    client_res = {}
    for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
        ckpt_path = Path(f'models/checkpoints/{cid}_resnet.pt')
        if ckpt_path.exists():
            payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model = build_resnet_1d(input_dim=payload.get('input_dim', 25))
            model.load_state_dict(payload['model_state_dict'])
            model.eval()
            probs = []
            with torch.no_grad():
                for x, _ in test_loaders[cid]:
                    out = model(x)
                    p = torch.sigmoid(out).numpy().flatten()
                    probs.extend(p)
            client_res[cid] = compute_metrics_from_preds(test_targets[cid], np.array(probs))

    if len(client_res) == 3:
        macro_res = aggregate_macro(client_res)
        wgt_res = aggregate_sample_weighted(client_res)
        resnet_params = sum(p.numel() for p in model.parameters())

        for cid, m in client_res.items():
            records.append({
                'experiment_id': 'local_resnet',
                'seed': 42,
                'hospital': cid,
                'aggregation_level': 'hospital',
                'parameters': {'shared_parameters': resnet_params, 'private_encoder_parameters': 0},
                'communication': {'bytes_per_round': 0, 'total_bytes_communicated': 0},
                'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
                'metrics': m
            })
        records.append({
            'experiment_id': 'local_resnet',
            'seed': 42,
            'hospital': 'ALL',
            'aggregation_level': 'macro',
            'parameters': {'shared_parameters': resnet_params, 'private_encoder_parameters': 0},
            'communication': {'bytes_per_round': 0, 'total_bytes_communicated': 0},
            'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
            'metrics': macro_res
        })
        records.append({
            'experiment_id': 'local_resnet',
            'seed': 42,
            'hospital': 'ALL',
            'aggregation_level': 'sample_weighted',
            'parameters': {'shared_parameters': resnet_params, 'private_encoder_parameters': 0},
            'communication': {'bytes_per_round': 0, 'total_bytes_communicated': 0},
            'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
            'metrics': wgt_res
        })

    # -------------------------------------------------------------------------
    # LOCAL XGBOOST
    # -------------------------------------------------------------------------
    client_res = {}
    for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
        ckpt_path = Path(f'models/checkpoints/{cid}_xgboost.json')
        if ckpt_path.exists():
            xgb = LocalXGBoostModel()
            xgb.load_model(ckpt_path)
            x_test_df = pd.read_csv(Path(f'data/processed/{cid}/test/X_test.csv'))
            probs = xgb.predict_proba(x_test_df)[:, 1]
            client_res[cid] = compute_metrics_from_preds(test_targets[cid], probs)

    if len(client_res) == 3:
        macro_res = aggregate_macro(client_res)
        wgt_res = aggregate_sample_weighted(client_res)
        for cid, m in client_res.items():
            records.append({
                'experiment_id': 'local_xgboost',
                'seed': 42,
                'hospital': cid,
                'aggregation_level': 'hospital',
                'parameters': {'shared_parameters': 0, 'private_encoder_parameters': 0},
                'communication': {'bytes_per_round': 0, 'total_bytes_communicated': 0},
                'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
                'metrics': m
            })
        records.append({
            'experiment_id': 'local_xgboost',
            'seed': 42,
            'hospital': 'ALL',
            'aggregation_level': 'macro',
            'parameters': {'shared_parameters': 0, 'private_encoder_parameters': 0},
            'communication': {'bytes_per_round': 0, 'total_bytes_communicated': 0},
            'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
            'metrics': macro_res
        })
        records.append({
            'experiment_id': 'local_xgboost',
            'seed': 42,
            'hospital': 'ALL',
            'aggregation_level': 'sample_weighted',
            'parameters': {'shared_parameters': 0, 'private_encoder_parameters': 0},
            'communication': {'bytes_per_round': 0, 'total_bytes_communicated': 0},
            'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
            'metrics': wgt_res
        })

    # -------------------------------------------------------------------------
    # HOMOGENEOUS FEDAVG ALEXNET
    # -------------------------------------------------------------------------
    ckpt_path = Path('models/checkpoints/federated_alexnet/global_alexnet_final.pt')
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = build_alexnet_1d(input_dim=payload.get('input_dim', 25))
        model.load_state_dict(payload['model_state_dict'])
        client_bns = payload.get('client_bn_states', {})
        client_res = {}
        for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
            # Load FedBN state for this client if available
            if cid in client_bns:
                for k, v in client_bns[cid].items():
                    model.state_dict()[k].copy_(v)
            model.eval()
            probs = []
            with torch.no_grad():
                for x, _ in test_loaders[cid]:
                    out = model(x)
                    probs.extend(torch.sigmoid(out).numpy().flatten())
            client_res[cid] = compute_metrics_from_preds(test_targets[cid], np.array(probs))

        shared_params = sum(p.numel() for p in model.parameters())
        bytes_per_round = 2 * 3 * shared_params * 4
        total_bytes = 15 * bytes_per_round
        macro_res = aggregate_macro(client_res)
        wgt_res = aggregate_sample_weighted(client_res)

        for cid, m in client_res.items():
            records.append({
                'experiment_id': 'homogeneous_fedavg_alexnet',
                'seed': 42,
                'hospital': cid,
                'aggregation_level': 'hospital',
                'parameters': {'shared_parameters': shared_params, 'private_encoder_parameters': 0},
                'communication': {'bytes_per_round': bytes_per_round, 'total_bytes_communicated': total_bytes},
                'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
                'metrics': m
            })
        records.append({
            'experiment_id': 'homogeneous_fedavg_alexnet',
            'seed': 42,
            'hospital': 'ALL',
            'aggregation_level': 'macro',
            'parameters': {'shared_parameters': shared_params, 'private_encoder_parameters': 0},
            'communication': {'bytes_per_round': bytes_per_round, 'total_bytes_communicated': total_bytes},
            'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
            'metrics': macro_res
        })
        records.append({
            'experiment_id': 'homogeneous_fedavg_alexnet',
            'seed': 42,
            'hospital': 'ALL',
            'aggregation_level': 'sample_weighted',
            'parameters': {'shared_parameters': shared_params, 'private_encoder_parameters': 0},
            'communication': {'bytes_per_round': bytes_per_round, 'total_bytes_communicated': total_bytes},
            'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
            'metrics': wgt_res
        })

    # -------------------------------------------------------------------------
    # HOMOGENEOUS FEDAVG RESNET
    # -------------------------------------------------------------------------
    ckpt_path = Path('models/checkpoints/federated_resnet/global_resnet_final.pt')
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = build_resnet_1d(input_dim=payload.get('input_dim', 25))
        model.load_state_dict(payload['model_state_dict'])
        client_bns = payload.get('client_bn_states', {})
        client_res = {}
        for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
            if cid in client_bns:
                for k, v in client_bns[cid].items():
                    model.state_dict()[k].copy_(v)
            model.eval()
            probs = []
            with torch.no_grad():
                for x, _ in test_loaders[cid]:
                    out = model(x)
                    probs.extend(torch.sigmoid(out).numpy().flatten())
            client_res[cid] = compute_metrics_from_preds(test_targets[cid], np.array(probs))

        shared_params = sum(p.numel() for p in model.parameters())
        bytes_per_round = 2 * 3 * shared_params * 4
        total_bytes = 15 * bytes_per_round
        macro_res = aggregate_macro(client_res)
        wgt_res = aggregate_sample_weighted(client_res)

        for cid, m in client_res.items():
            records.append({
                'experiment_id': 'homogeneous_fedavg_resnet',
                'seed': 42,
                'hospital': cid,
                'aggregation_level': 'hospital',
                'parameters': {'shared_parameters': shared_params, 'private_encoder_parameters': 0},
                'communication': {'bytes_per_round': bytes_per_round, 'total_bytes_communicated': total_bytes},
                'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
                'metrics': m
            })
        records.append({
            'experiment_id': 'homogeneous_fedavg_resnet',
            'seed': 42,
            'hospital': 'ALL',
            'aggregation_level': 'macro',
            'parameters': {'shared_parameters': shared_params, 'private_encoder_parameters': 0},
            'communication': {'bytes_per_round': bytes_per_round, 'total_bytes_communicated': total_bytes},
            'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
            'metrics': macro_res
        })
        records.append({
            'experiment_id': 'homogeneous_fedavg_resnet',
            'seed': 42,
            'hospital': 'ALL',
            'aggregation_level': 'sample_weighted',
            'parameters': {'shared_parameters': shared_params, 'private_encoder_parameters': 0},
            'communication': {'bytes_per_round': bytes_per_round, 'total_bytes_communicated': total_bytes},
            'privacy': {'secure_aggregation': False, 'differential_privacy': False, 'epsilon': None},
            'metrics': wgt_res
        })

    # -------------------------------------------------------------------------
    # HETEROGENEOUS FEDERATED EXPERIMENTAL FAMILY (6 CONFIGURATIONS)
    # -------------------------------------------------------------------------
    het_ckpts = [
        {
            'exp_id': 'heterogeneous_fedavg_mlp',
            'ckpt': 'models/checkpoints/federated_heterogeneous/global_heterogeneous_fedavg_predictor_final.pt',
            'sec_agg': False, 'dp': False, 'eps': None, 'mu': 0.0
        },
        {
            'exp_id': 'heterogeneous_fedprox_mlp_mu_001',
            'ckpt': 'models/checkpoints/federated_heterogeneous/global_heterogeneousfedprox_predictor_final.pt',
            'sec_agg': False, 'dp': False, 'eps': None, 'mu': 0.01
        },
        {
            'exp_id': 'heterogeneous_fedadam_mlp',
            'ckpt': 'models/checkpoints/federated_heterogeneous/global_heterogeneousfedadam_predictor_final.pt',
            'sec_agg': False, 'dp': False, 'eps': None, 'mu': 0.0
        },
        {
            'exp_id': 'heterogeneous_fedavg_secure_aggregation',
            'ckpt': 'models/checkpoints/federated_heterogeneous/global_heterogeneous_fedavg_secure_predictor_final.pt',
            'sec_agg': True, 'dp': False, 'eps': None, 'mu': 0.0
        },
        {
            'exp_id': 'heterogeneous_fedavg_dp',
            'ckpt': 'models/checkpoints/federated_heterogeneous/global_heterogeneous_fedavg_dp_predictor_final.pt',
            'sec_agg': False, 'dp': True, 'eps': 148.0259, 'mu': 0.0
        },
        {
            'exp_id': 'heterogeneous_fedavg_secure_aggregation_dp',
            'ckpt': 'models/checkpoints/federated_heterogeneous/global_heterogeneous_fedavg_secure_dp_predictor_final.pt',
            'sec_agg': True, 'dp': True, 'eps': 148.0259, 'mu': 0.0
        },
        {
            'exp_id': 'heterogeneous_fedprox_secure_aggregation_dp',
            'ckpt': 'models/checkpoints/federated_heterogeneous/global_heterogeneous_fedprox_secure_dp_predictor_final.pt',
            'sec_agg': True, 'dp': True, 'eps': 148.0259, 'mu': 0.01
        },
        {
            'exp_id': 'heterogeneous_fedadam_secure_aggregation_dp',
            'ckpt': 'models/checkpoints/federated_heterogeneous/global_heterogeneous_fedadam_secure_dp_predictor_final.pt',
            'sec_agg': True, 'dp': True, 'eps': 148.0259, 'mu': 0.0
        }
    ]

    # Pre-instantiate client composite models to get private encoder params
    sample_client_composite = {}
    private_encoder_params_per_hosp = {}
    for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
        sch = get_client_schema(cid)
        comp = build_heterogeneous_model(input_dim=sch.input_dimension, latent_dim=32)
        sample_client_composite[cid] = comp
        private_encoder_params_per_hosp[cid] = sum(p.numel() for p in comp.encoder.parameters())

    sample_predictor = SharedPredictor(latent_dim=32, hidden_dims=[32])
    shared_predictor_params = sum(p.numel() for p in sample_predictor.parameters())
    # Communication strictly transmits shared predictor, ZERO encoder transmission
    bytes_per_round_het = 2 * 3 * shared_predictor_params * 4
    total_bytes_het = 15 * bytes_per_round_het

    for het in het_ckpts:
        p_file = Path(het['ckpt'])
        if not p_file.exists():
            continue

        payload = torch.load(p_file, map_location='cpu', weights_only=False)
        shared_pred = SharedPredictor(latent_dim=payload.get('latent_dim', 32), hidden_dims=payload.get('hidden_dims', [32]))
        shared_pred.load_state_dict(payload['shared_predictor_state_dict'])
        shared_pred.eval()

        client_res = {}
        for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
            sch = get_client_schema(cid)
            comp = build_heterogeneous_model(input_dim=sch.input_dimension, latent_dim=payload.get('latent_dim', 32))
            # Load trained client encoder if available in final client ckpt, else evaluate composite
            c_ckpt_file = Path(f'models/checkpoints/federated_heterogeneous/{cid}_heterogeneous_final.pt')
            if c_ckpt_file.exists():
                c_payload = torch.load(c_ckpt_file, map_location='cpu', weights_only=False)
                if 'encoder_state_dict' in c_payload:
                    comp.encoder.load_state_dict(c_payload['encoder_state_dict'])
            comp.predictor.load_state_dict(shared_pred.state_dict())
            comp.eval()

            probs = []
            with torch.no_grad():
                for x, _ in test_loaders[cid]:
                    out = comp(x)
                    probs.extend(torch.sigmoid(out).numpy().flatten())
            client_res[cid] = compute_metrics_from_preds(test_targets[cid], np.array(probs))

        macro_res = aggregate_macro(client_res)
        wgt_res = aggregate_sample_weighted(client_res)

        for cid, m in client_res.items():
            records.append({
                'experiment_id': het['exp_id'],
                'seed': 42,
                'hospital': cid,
                'aggregation_level': 'hospital',
                'parameters': {
                    'shared_parameters': shared_predictor_params,
                    'private_encoder_parameters': private_encoder_params_per_hosp[cid]
                },
                'communication': {
                    'bytes_per_round': bytes_per_round_het,
                    'total_bytes_communicated': total_bytes_het,
                    'private_encoder_bytes_communicated': 0
                },
                'privacy': {
                    'secure_aggregation': het['sec_agg'],
                    'differential_privacy': het['dp'],
                    'epsilon': het['eps']
                },
                'metrics': m
            })
        records.append({
            'experiment_id': het['exp_id'],
            'seed': 42,
            'hospital': 'ALL',
            'aggregation_level': 'macro',
            'parameters': {
                'shared_parameters': shared_predictor_params,
                'private_encoder_parameters': int(np.mean(list(private_encoder_params_per_hosp.values())))
            },
            'communication': {
                'bytes_per_round': bytes_per_round_het,
                'total_bytes_communicated': total_bytes_het,
                'private_encoder_bytes_communicated': 0
            },
            'privacy': {
                'secure_aggregation': het['sec_agg'],
                'differential_privacy': het['dp'],
                'epsilon': het['eps']
            },
            'metrics': macro_res
        })
        records.append({
            'experiment_id': het['exp_id'],
            'seed': 42,
            'hospital': 'ALL',
            'aggregation_level': 'sample_weighted',
            'parameters': {
                'shared_parameters': shared_predictor_params,
                'private_encoder_parameters': int(np.mean(list(private_encoder_params_per_hosp.values())))
            },
            'communication': {
                'bytes_per_round': bytes_per_round_het,
                'total_bytes_communicated': total_bytes_het,
                'private_encoder_bytes_communicated': 0
            },
            'privacy': {
                'secure_aggregation': het['sec_agg'],
                'differential_privacy': het['dp'],
                'epsilon': het['eps']
            },
            'metrics': wgt_res
        })

    # Write out to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    print(f'Successfully built authoritative central results: {output_path} with {len(records)} records.')
    return records


if __name__ == '__main__':
    build_and_export_results()
