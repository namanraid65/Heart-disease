"""
Hospital Client Implementation for Federated Learning
Simulates an independent hospital silo participating in Federated 1D AlexNet training.
Exchanges exclusively model weight parameters; never transmits raw patient data.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import flwr as fl

from federated.config import (
    LOCAL_EPOCHS,
    LOCAL_BATCH_SIZE,
    LOCAL_LEARNING_RATE,
    LOCAL_WEIGHT_DECAY,
    DEVICE,
    INPUT_FEATURES,
    DROPOUT_RATE,
    FEDERATED_CLIENTS
)
from federated.utils import (
    get_model_shared_parameters,
    set_model_shared_parameters,
    get_model_bn_state,
    set_model_bn_state,
    verify_privacy_and_data_locality
)
from models.alexnet_1d import build_alexnet_1d
from models.dataset import get_client_dataloaders


class HospitalClient(fl.client.NumPyClient):
    """
    Federated Client representing a single hospital medical institution.
    Operates strictly within its local data boundary.
    Under FedBN, shared parameters are synchronized while BatchNorm running
    statistics remain strictly client-local.
    """

    def __init__(
        self,
        client_id: str,
        device: str = DEVICE
    ):
        super().__init__()
        self.client_id = client_id
        self.client_meta = FEDERATED_CLIENTS[client_id]
        self.client_name = self.client_meta['name']
        self.device = device

        # Initialize local 1D AlexNet architecture
        self.model = build_alexnet_1d(input_dim=INPUT_FEATURES, dropout_rate=DROPOUT_RATE).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

        # Load local datasets strictly from this client's processed partition
        loaders = get_client_dataloaders(client_id=client_id, batch_size=LOCAL_BATCH_SIZE, shuffle_train=True)
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']

        self.num_train_samples = len(self.train_loader.dataset)
        self.num_val_samples = len(self.val_loader.dataset)
        self.num_test_samples = len(self.test_loader.dataset)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Returns local shared model weights as NumPy ndarrays (excludes local BN state)."""
        return get_model_shared_parameters(self.model)

    def get_bn_state(self) -> Dict[str, torch.Tensor]:
        """Returns client-local BatchNorm buffer tensors."""
        return get_model_bn_state(self.model)

    def set_bn_state(self, bn_state: Dict[str, torch.Tensor]) -> None:
        """Sets client-local BatchNorm buffer tensors."""
        set_model_bn_state(self.model, bn_state)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Executes local SGD training on local client training partition.
        Receives global shared parameters, preserves local BN statistics,
        performs local epochs, and returns updated shared weights.
        """
        # 1. Update local model with global shared weights (local BN state preserved)
        set_model_shared_parameters(self.model, parameters)

        # 2. Configure local optimizer
        lr = config.get('lr', LOCAL_LEARNING_RATE)
        epochs = config.get('local_epochs', LOCAL_EPOCHS)
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=LOCAL_WEIGHT_DECAY)

        self.model.train()
        epoch_losses = []
        epoch_accs = []

        for epoch in range(epochs):
            total_loss = 0.0
            all_preds = []
            all_targets = []

            for features, targets in self.train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits = self.model(features)
                loss = self.criterion(logits, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * features.size(0)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                all_preds.extend(preds.flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

            epoch_loss = total_loss / self.num_train_samples
            epoch_acc = accuracy_score(all_targets, all_preds)
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)

        # 3. Extract updated local shared parameters (local BN buffers remain local)
        updated_params = get_model_shared_parameters(self.model)

        # 4. Privacy Audit: Ensure no patient-level records are included
        verify_privacy_and_data_locality(updated_params)

        metrics = {
            'client_id': self.client_id,
            'client_name': self.client_name,
            'train_loss': float(epoch_losses[-1]),
            'train_loss_mean': float(np.mean(epoch_losses)),
            'train_accuracy': float(epoch_accs[-1]),
            'train_accuracy_mean': float(np.mean(epoch_accs)),
            'num_samples': self.num_train_samples
        }

        return updated_params, self.num_train_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluates received shared model parameters with client's local BN state on local validation split.
        """
        set_model_shared_parameters(self.model, parameters)
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(features)
                loss = self.criterion(logits, targets)

                total_loss += loss.item() * features.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        y_true = np.array(all_targets, dtype=int)
        y_pred = np.array(all_preds, dtype=int)
        y_prob = np.array(all_probs, dtype=float)

        val_loss = float(total_loss / self.num_val_samples) if self.num_val_samples > 0 else 0.0
        val_acc = float(accuracy_score(y_true, y_pred)) if self.num_val_samples > 0 else 0.0
        val_f1 = float(f1_score(y_true, y_pred, zero_division=0)) if self.num_val_samples > 0 else 0.0

        if len(np.unique(y_true)) > 1:
            try:
                val_auc = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                val_auc = float(np.nan)
        else:
            val_auc = float(np.nan)

        from sklearn.metrics import average_precision_score
        num_pos = int((y_true == 1).sum())
        if len(np.unique(y_true)) > 1 and num_pos > 0:
            try:
                val_pr_auc = float(average_precision_score(y_true, y_prob))
            except ValueError:
                val_pr_auc = float(np.nan)
        else:
            val_pr_auc = float(np.nan)

        metrics = {
            'client_id': self.client_id,
            'client_name': self.client_name,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'val_roc_auc': val_auc,
            'val_pr_auc': val_pr_auc,
            'num_samples': self.num_val_samples
        }

        return float(val_loss), self.num_val_samples, metrics


if __name__ == '__main__':
    # Test client instantiation
    c1 = HospitalClient('hospital_1')
    params = c1.get_parameters({})
    print(f"HospitalClient 1 instantiated successfully! Param layers: {len(params)}, Train samples: {c1.num_train_samples}")
