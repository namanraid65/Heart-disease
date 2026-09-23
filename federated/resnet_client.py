"""
Hospital Client Implementation for Federated 1D ResNet
Simulates an independent hospital node participating in collaborative ResNet training.
Only model parameter tensors are communicated; patient records strictly remain local.
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
    RESNET_LOCAL_EPOCHS,
    RESNET_LOCAL_BATCH_SIZE,
    RESNET_LOCAL_LEARNING_RATE,
    RESNET_LOCAL_WEIGHT_DECAY,
    RESNET_DROPOUT_RATE,
    DEVICE,
    INPUT_FEATURES,
    FEDERATED_CLIENTS
)
from federated.utils import (
    get_model_parameters,
    set_model_parameters,
    get_model_shared_parameters,
    set_model_shared_parameters,
    get_model_bn_state,
    set_model_bn_state,
    verify_privacy_and_data_locality
)
from models.resnet_1d import ResNet1D, build_resnet_1d
from models.dataset import get_client_dataloaders


class HospitalResNetClient(fl.client.NumPyClient):
    """
    Federated Client representing an individual hospital for 1D ResNet training.
    Enforces strict data isolation.
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

        # Initialize local 1D ResNet architecture
        self.model = build_resnet_1d(
            input_dim=INPUT_FEATURES,
            dropout_rate=RESNET_DROPOUT_RATE
        ).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

        # Load local dataloaders (ONLY client's own data partition)
        self.loaders = get_client_dataloaders(
            client_id=client_id,
            batch_size=RESNET_LOCAL_BATCH_SIZE,
            shuffle_train=True
        )
        self.train_loader = self.loaders['train']
        self.val_loader = self.loaders['val']
        self.test_loader = self.loaders['test']

        self.num_train_samples = len(self.train_loader.dataset)
        self.num_val_samples = len(self.val_loader.dataset)
        self.num_test_samples = len(self.test_loader.dataset)

    def get_parameters(self, config: Dict[str, Any] = None) -> List[np.ndarray]:
        """Returns local shared model weight parameters as NumPy ndarrays (excludes local BN state)."""
        params = get_model_shared_parameters(self.model)
        verify_privacy_and_data_locality(params)
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Sets local shared model weights while preserving client-local BN state."""
        verify_privacy_and_data_locality(parameters)
        set_model_shared_parameters(self.model, parameters)

    def get_bn_state(self) -> Dict[str, torch.Tensor]:
        """Returns client-local BatchNorm buffer tensors."""
        return get_model_bn_state(self.model)

    def set_bn_state(self, bn_state: Dict[str, torch.Tensor]) -> None:
        """Sets client-local BatchNorm buffer tensors."""
        set_model_bn_state(self.model, bn_state)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any] = None
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Executes local training for defined local epochs using current global weights.
        """
        self.set_parameters(parameters)

        local_epochs = config.get('local_epochs', RESNET_LOCAL_EPOCHS) if config else RESNET_LOCAL_EPOCHS
        lr = config.get('lr', RESNET_LOCAL_LEARNING_RATE) if config else RESNET_LOCAL_LEARNING_RATE
        weight_decay = config.get('weight_decay', RESNET_LOCAL_WEIGHT_DECAY) if config else RESNET_LOCAL_WEIGHT_DECAY

        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.model.train()
        total_epoch_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for features, targets in self.train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device).view(-1, 1)

                optimizer.zero_grad()
                logits = self.model(features)
                loss = self.criterion(logits, targets)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * features.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).int()
                correct += (preds == targets.int()).sum().item()
                total += targets.size(0)

            total_epoch_loss += epoch_loss

        avg_train_loss = total_epoch_loss / (total * local_epochs) if total > 0 else 0.0
        train_accuracy = correct / (total * local_epochs) if total > 0 else 0.0

        updated_params = self.get_parameters()
        verify_privacy_and_data_locality(updated_params)

        metrics = {
            'client_id': self.client_id,
            'client_name': self.client_name,
            'train_loss': float(avg_train_loss),
            'train_accuracy': float(train_accuracy),
            'num_samples': self.num_train_samples
        }

        return updated_params, self.num_train_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any] = None
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluates current parameters on the local validation partition.
        """
        self.set_parameters(parameters)
        self.model.eval()

        eval_loss = 0.0
        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device).view(-1, 1)

                logits = self.model(features)
                loss = self.criterion(logits, targets)
                eval_loss += loss.item() * features.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        val_loss = float(eval_loss / self.num_val_samples) if self.num_val_samples > 0 else 0.0
        y_true = np.array(all_targets, dtype=int)
        y_pred = np.array(all_preds, dtype=int)
        y_prob = np.array(all_probs, dtype=float)

        acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0

        if len(np.unique(y_true)) > 1:
            try:
                auc = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                auc = float(np.nan)
        else:
            auc = float(np.nan)

        from sklearn.metrics import average_precision_score
        num_pos = int((y_true == 1).sum())
        if len(np.unique(y_true)) > 1 and num_pos > 0:
            try:
                pr_auc = float(average_precision_score(y_true, y_prob))
            except ValueError:
                pr_auc = float(np.nan)
        else:
            pr_auc = float(np.nan)

        metrics = {
            'client_id': self.client_id,
            'client_name': self.client_name,
            'val_loss': float(val_loss),
            'val_accuracy': float(acc),
            'val_f1': float(f1),
            'val_roc_auc': float(auc),
            'val_pr_auc': float(pr_auc),
            'num_samples': self.num_val_samples
        }

        return float(val_loss), self.num_val_samples, metrics


if __name__ == '__main__':
    client = HospitalResNetClient('hospital_1')
    params = client.get_parameters()
    print(f"HospitalResNetClient (H1) initialized successfully. Layers: {len(params)}")
    updated_p, n_samples, metrics = client.fit(params)
    print(f"Local fit complete on {n_samples} samples. Metrics: {metrics}")
