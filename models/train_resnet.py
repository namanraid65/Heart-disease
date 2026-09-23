"""
Training Pipeline for Local 1D ResNet Models
Trains independent local ResNet models for Hospital 1 (Cleveland), Hospital 2 (Hungarian),
and Hospital 3 (Switzerland) without federated aggregation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from models.config import (
    CLIENT_CONFIGS,
    CHECKPOINTS_DIR,
    RANDOM_SEED,
    DEVICE,
    INPUT_FEATURES,
    DROPOUT_RATE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MIN_DELTA
)
from models.resnet_1d import ResNet1D, build_resnet_1d
from models.dataset import get_client_dataloaders


def set_seed(seed: int = RANDOM_SEED):
    """Sets random seeds for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_single_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float, float, float]:
    """
    Executes one training epoch.
    Returns: (epoch_loss, epoch_accuracy, epoch_f1, epoch_auc)
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []

    for features, targets in train_loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs.flatten())
        all_preds.extend(preds.flatten())
        all_targets.extend(targets.cpu().numpy().flatten())

    n_samples = len(all_targets)
    epoch_loss = total_loss / n_samples
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_acc, epoch_f1, epoch_auc


def evaluate_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, float, float]:
    """
    Evaluates model on validation or test split.
    Returns: (loss, accuracy, f1, auc)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []

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

    n_samples = len(all_targets)
    eval_loss = total_loss / n_samples
    eval_acc = accuracy_score(all_targets, all_preds)
    eval_f1 = f1_score(all_targets, all_preds, zero_division=0)

    try:
        eval_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        eval_auc = 0.5

    return eval_loss, eval_acc, eval_f1, eval_auc


def train_client_resnet(
    client_id: str,
    max_epochs: int = MAX_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = BATCH_SIZE,
    patience: int = EARLY_STOPPING_PATIENCE,
    device: str = DEVICE,
    verbose: bool = True,
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Trains a local 1D ResNet model on a single hospital client's dataset.
    Saves the best checkpoint to models/checkpoints/<resnet_checkpoint>.
    """
    set_seed(seed)
    client_meta = CLIENT_CONFIGS[client_id]
    client_name = client_meta['name']
    checkpoint_name = client_meta.get('resnet_checkpoint', f"{client_id}_resnet.pt")
    checkpoint_file = CHECKPOINTS_DIR / checkpoint_name

    if verbose:
        print("\n" + "=" * 80)
        print(f" TRAINING LOCAL 1D RESNET: {client_name}")
        print("=" * 80)

    # 1. Load DataLoaders
    loaders = get_client_dataloaders(client_id=client_id, batch_size=batch_size, shuffle_train=True)
    train_loader = loaders['train']
    val_loader = loaders['val']

    sample_feat, _ = next(iter(train_loader))
    input_dim = sample_feat.shape[1]

    # 2. Build 1D ResNet Model
    model = build_resnet_1d(input_dim=input_dim, dropout_rate=DROPOUT_RATE).to(device)

    # 3. Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 4. Training Loop with Early Stopping
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_auc': [], 'val_auc': []
    }

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_weights = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc, tr_f1, tr_auc = train_single_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, val_auc = evaluate_split(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(tr_f1)
        history['val_f1'].append(val_f1)
        history['train_auc'].append(tr_auc)
        history['val_auc'].append(val_auc)

        if val_loss < (best_val_loss - MIN_DELTA):
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == max_epochs or epochs_no_improve == patience):
            print(f"Epoch [{epoch:>3}/{max_epochs}] | "
                  f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.1f}%, AUC: {tr_auc:.3f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%, AUC: {val_auc:.3f} "
                  f"{'(Best)' if epoch == best_epoch else ''}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping triggered at Epoch {epoch}. Best epoch was {best_epoch} with Val Loss: {best_val_loss:.4f}")
            break

    # 5. Restore Best Model Weights & Save Checkpoint
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        'model_state_dict': model.state_dict(),
        'model_type': 'ResNet1D',
        'client_id': client_id,
        'client_name': client_name,
        'input_dim': input_dim,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'hyperparameters': {
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'dropout_rate': DROPOUT_RATE,
            'random_seed': RANDOM_SEED
        },
        'history': history
    }
    torch.save(checkpoint_payload, checkpoint_file)
    if verbose:
        print(f">> Saved ResNet checkpoint to: {checkpoint_file}")

    return {
        'client_id': client_id,
        'client_name': client_name,
        'model': model,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'checkpoint_path': checkpoint_file,
        'history': history
    }


def train_all_local_resnets(seed: int = RANDOM_SEED) -> Dict[str, Dict[str, Any]]:
    """
    Trains independent local 1D ResNet models for all 3 hospital clients.
    """
    print("=" * 80)
    print(" STARTING LOCAL 1D RESNET TRAINING FOR ALL CLIENTS")
    print("=" * 80)

    training_results = {}
    for cid in CLIENT_CONFIGS:
        res = train_client_resnet(client_id=cid, verbose=True, seed=seed)
        training_results[cid] = res

    print("\n" + "=" * 80)
    print(" ALL LOCAL RESNET MODELS TRAINED SUCCESSFULLY!")
    print("=" * 80)
    return training_results


if __name__ == '__main__':
    train_all_local_resnets()
