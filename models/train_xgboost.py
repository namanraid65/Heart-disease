"""
Training Pipeline for Local XGBoost Models
Trains independent local XGBoost classifiers for Hospital 1 (Cleveland),
Hospital 2 (Hungarian), and Hospital 3 (Switzerland) without federated aggregation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from models.config import (
    CLIENT_CONFIGS,
    CHECKPOINTS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    XGBOOST_PARAMS,
    INPUT_FEATURES
)
from models.xgboost_model import LocalXGBoostModel, build_xgboost_model
from preprocessing.feature_schema import PROCESSED_FEATURE_NAMES


def load_client_tabular_data(client_id: str) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Loads raw CSVs for train, validation, and test sets for a single client.
    """
    client_dir = PROCESSED_DATA_DIR / client_id
    if not client_dir.exists():
        raise FileNotFoundError(f"Client directory not found: {client_dir}")

    X_train = pd.read_csv(client_dir / 'train' / 'X_train.csv')
    y_train = pd.read_csv(client_dir / 'train' / 'y_train.csv').squeeze('columns')

    X_val = pd.read_csv(client_dir / 'validation' / 'X_val.csv')
    y_val = pd.read_csv(client_dir / 'validation' / 'y_val.csv').squeeze('columns')

    X_test = pd.read_csv(client_dir / 'test' / 'X_test.csv')
    y_test = pd.read_csv(client_dir / 'test' / 'y_test.csv').squeeze('columns')

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def train_client_xgboost(
    client_id: str,
    verbose: bool = True,
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Trains a local XGBoost model on a single client's training partition.
    Utilizes validation partition strictly for early stopping.
    Saves JSON model checkpoint to models/checkpoints/<xgboost_checkpoint>.
    """
    client_meta = CLIENT_CONFIGS[client_id]
    client_name = client_meta['name']
    checkpoint_name = client_meta.get('xgboost_checkpoint', f"{client_id}_xgboost.json")
    checkpoint_file = CHECKPOINTS_DIR / checkpoint_name

    if verbose:
        print("\n" + "=" * 80)
        print(f" TRAINING LOCAL XGBOOST: {client_name}")
        print("=" * 80)

    # 1. Load Data
    splits = load_client_tabular_data(client_id)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']

    # Compute scale_pos_weight from local train distribution
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

    if verbose:
        print(f"  Training Samples:   {len(X_train)} (Class 0: {neg_count}, Class 1: {pos_count})")
        print(f"  Validation Samples: {len(X_val)}")
        print(f"  Scale Pos Weight:   {scale_pos_weight:.4f}")

    # 2. Build Model
    model = build_xgboost_model(scale_pos_weight=scale_pos_weight, random_state=seed)

    # 3. Fit Model with Early Stopping on Validation Split
    model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        verbose=False
    )

    # 4. Evaluate Validation Performance
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, zero_division=0)
    try:
        val_auc = roc_auc_score(y_val, val_probs)
    except ValueError:
        val_auc = 0.5

    if verbose:
        print(f"  Best Iteration:     {model.best_iteration}")
        print(f"  Validation Acc:     {val_acc * 100:.2f}%")
        print(f"  Validation F1:      {val_f1:.4f}")
        print(f"  Validation ROC-AUC: {val_auc:.4f}")

    # 5. Save Model Checkpoint
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(checkpoint_file)
    if verbose:
        print(f">> Saved XGBoost checkpoint to: {checkpoint_file}")

    return {
        'client_id': client_id,
        'client_name': client_name,
        'model': model,
        'best_iteration': model.best_iteration,
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'checkpoint_path': checkpoint_file
    }


def train_all_local_xgboosts(seed: int = RANDOM_SEED) -> Dict[str, Dict[str, Any]]:
    """
    Trains independent local XGBoost models for all 3 hospital clients.
    """
    print("=" * 80)
    print(" STARTING LOCAL XGBOOST TRAINING FOR ALL CLIENTS")
    print("=" * 80)

    results = {}
    for cid in CLIENT_CONFIGS:
        res = train_client_xgboost(cid, verbose=True, seed=seed)
        results[cid] = res

    print("\n" + "=" * 80)
    print(" ALL LOCAL XGBOOST MODELS TRAINED SUCCESSFULLY!")
    print("=" * 80)
    return results


if __name__ == '__main__':
    train_all_local_xgboosts()
