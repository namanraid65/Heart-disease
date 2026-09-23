"""
Heterogeneous Hospital Client Implementation
Encapsulates a private institution-local feature encoder and a shared federated predictor.
During FL communication rounds, only the shared predictor parameters are transmitted.
The private encoder weights remain strictly local to the hospital node.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import flwr as fl

from preprocessing.heterogeneous_schema import get_client_schema, ClientFeatureSchema
from models.heterogeneous.encoder import HospitalEncoder
from models.heterogeneous.predictor import SharedPredictor
from models.heterogeneous.composite import HeterogeneousCompositeModel, build_heterogeneous_model
from models.dataset import get_client_dataloaders
from federated.heterogeneous.config import (
    LATENT_DIM,
    ENCODER_HIDDEN_DIMS,
    PREDICTOR_HIDDEN_DIMS,
    DROPOUT_RATE,
    LOCAL_EPOCHS,
    LOCAL_BATCH_SIZE,
    LOCAL_LEARNING_RATE,
    LOCAL_WEIGHT_DECAY,
    DEVICE,
    HETEROGENEOUS_CLIENTS
)
from federated.utils import verify_privacy_and_data_locality
from federated.heterogeneous.privacy import clip_and_noise_update


class HeterogeneousHospitalClient(fl.client.NumPyClient):
    """
    Federated client representing an independent hospital with its own native feature schema.
    Maintains a private encoder (local) and synchronizes a shared predictor (federated).
    """

    def __init__(
        self,
        client_id: str,
        schema: Optional[ClientFeatureSchema] = None,
        latent_dim: int = LATENT_DIM,
        device: str = DEVICE,
        custom_loaders: Optional[Dict[str, DataLoader]] = None,
        encoder_hidden_dims: Optional[List[int]] = None,
        predictor_hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = DROPOUT_RATE,
        lr: float = LOCAL_LEARNING_RATE,
        weight_decay: float = LOCAL_WEIGHT_DECAY
    ):
        super().__init__()
        self.client_id = client_id
        self.device = device
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay

        # 1. Load Client Schema & derive input dimension dynamically
        if schema is not None:
            self.schema = schema
        else:
            self.schema = get_client_schema(client_id)

        self.input_dim = self.schema.input_dimension
        self.client_name = self.schema.hospital_name

        # 2. Build Composite Heterogeneous Model
        enc_hidden = encoder_hidden_dims or self.schema.encoder_config.get('hidden_dims', ENCODER_HIDDEN_DIMS)
        pred_hidden = predictor_hidden_dims or PREDICTOR_HIDDEN_DIMS

        encoder = HospitalEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=enc_hidden,
            dropout_rate=dropout_rate
        )
        predictor = SharedPredictor(
            latent_dim=self.latent_dim,
            hidden_dims=pred_hidden,
            dropout_rate=dropout_rate
        )
        self.model = HeterogeneousCompositeModel(encoder=encoder, predictor=predictor).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

        # 3. Load Datasets
        if custom_loaders is not None:
            self.train_loader = custom_loaders['train']
            self.val_loader = custom_loaders.get('val')
            self.test_loader = custom_loaders.get('test')
        else:
            loaders = get_client_dataloaders(
                client_id=client_id,
                batch_size=LOCAL_BATCH_SIZE,
                shuffle_train=True
            )
            self.train_loader = loaders['train']
            self.val_loader = loaders['val']
            self.test_loader = loaders['test']

        self.num_train_samples = len(self.train_loader.dataset)
        self.num_val_samples = len(self.val_loader.dataset) if self.val_loader else 0
        self.num_test_samples = len(self.test_loader.dataset) if self.test_loader else 0

    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """
        Extracts exclusively the shared predictor parameters as NumPy arrays.
        Private encoder weights are strictly preserved locally and NEVER returned.
        """
        params = self.model.get_shared_parameters()
        verify_privacy_and_data_locality(params)
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Updates strictly the shared predictor parameters.
        The private encoder parameters remain unchanged.
        """
        self.model.set_shared_parameters(parameters)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Executes local SGD/Adam training:
          1. Sets received global shared predictor parameters.
          2. Retains local private encoder weights.
          3. Trains BOTH private encoder and shared predictor on local client data.
          4. Returns updated shared predictor parameters, sample count, and training metrics.
        """
        # Step 1: Update shared predictor weights from server
        self.set_parameters(parameters)

        proximal_mu = float(config.get('proximal_mu', 0.0)) if config else 0.0
        # If FedProx is active, cache detached clones of received global shared predictor parameters
        global_predictor_params = None
        if proximal_mu > 0.0:
            global_predictor_params = [
                p.clone().detach().to(self.device)
                for p in self.model.predictor.parameters()
            ]

        # Step 2: Configure local optimizer over BOTH private encoder and shared predictor
        epochs = config.get('local_epochs', LOCAL_EPOCHS) if config else LOCAL_EPOCHS
        learning_rate = config.get('lr', self.lr) if config else self.lr

        optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.weight_decay
        )

        self.model.train()
        epoch_losses: List[float] = []
        epoch_task_losses: List[float] = []
        epoch_prox_losses: List[float] = []
        all_train_preds: List[int] = []
        all_train_targets: List[int] = []

        for _ in range(epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Validate input dimensions match this client's schema
                self.schema.validate_tensor_shape(batch_x)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                task_loss = self.criterion(logits, batch_y)

                if proximal_mu > 0.0 and global_predictor_params is not None:
                    # FedProx: Proximal term applies strictly to shared predictor parameters
                    prox_loss = 0.5 * proximal_mu * sum(
                        torch.sum((p - p_init) ** 2)
                        for p, p_init in zip(self.model.predictor.parameters(), global_predictor_params)
                    )
                    loss = task_loss + prox_loss
                    epoch_prox_losses.append(prox_loss.item())
                else:
                    loss = task_loss

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_task_losses.append(task_loss.item())
                preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy().flatten()
                targets = batch_y.long().cpu().numpy().flatten()
                all_train_preds.extend(preds)
                all_train_targets.extend(targets)

        avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        avg_task_loss = float(np.mean(epoch_task_losses)) if epoch_task_losses else 0.0
        avg_prox_loss = float(np.mean(epoch_prox_losses)) if epoch_prox_losses else 0.0
        train_acc = float(accuracy_score(all_train_targets, all_train_preds)) if all_train_targets else 0.0

        metrics = {
            'train_loss': avg_train_loss,
            'task_loss': avg_task_loss,
            'prox_loss': avg_prox_loss,
            'train_accuracy': train_acc
        }

        # Step 3: Return ONLY the shared predictor parameters
        updated_shared_params = self.get_parameters()

        # Step 4: Client-Side Differential Privacy (clipping + Gaussian noise)
        privacy_cfg = config.get('privacy_config', {}) if config else {}
        if isinstance(privacy_cfg, dict) and privacy_cfg.get('differential_privacy', False):
            # Compute update delta strictly on shared predictor: Delta_i = w_i - w_global
            raw_update = [u.astype(np.float32) - g.astype(np.float32) for u, g in zip(updated_shared_params, parameters)]
            max_norm = float(privacy_cfg.get('max_update_norm', 1.0))
            noise_mult = float(privacy_cfg.get('noise_multiplier', 0.0))
            seed = privacy_cfg.get('seed')
            client_seed = None
            if seed is not None:
                round_num = config.get('server_round', 1) if config else 1
                client_seed = (int(seed) + hash(self.client_id) + round_num * 10007) & 0xFFFFFFFF

            priv_update, dp_telemetry = clip_and_noise_update(
                update=raw_update,
                max_norm=max_norm,
                noise_multiplier=noise_mult,
                seed=client_seed
            )
            # Reconstruct privatized shared predictor parameters: w_priv = w_global + Delta_priv
            priv_shared_params = [g.astype(np.float32) + d for g, d in zip(parameters, priv_update)]
            self.set_parameters(priv_shared_params)
            updated_shared_params = priv_shared_params
            metrics['dp_telemetry'] = dp_telemetry

        return updated_shared_params, self.num_train_samples, metrics

    def evaluate(
        self,
        parameters: Optional[List[np.ndarray]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluates composite model (private encoder + shared predictor) on local validation or test split.
        """
        if parameters is not None:
            self.set_parameters(parameters)

        split = config.get('split', 'val') if config else 'val'
        loader = self.val_loader if split == 'val' else self.test_loader

        if loader is None:
            raise ValueError(f"No dataloader available for split '{split}' on client '{self.client_id}'.")

        self.model.eval()
        total_loss = 0.0
        all_logits: List[float] = []
        all_probs: List[float] = []
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.schema.validate_tensor_shape(batch_x)

                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                total_loss += loss.item() * len(batch_y)

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(int)
                targets = batch_y.cpu().numpy().flatten().astype(int)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(targets)

        num_samples = len(loader.dataset)
        avg_loss = float(total_loss / num_samples) if num_samples > 0 else 0.0

        targets_arr = np.array(all_targets)
        preds_arr = np.array(all_preds)
        probs_arr = np.array(all_probs)

        acc = float(accuracy_score(targets_arr, preds_arr))
        prec = float(precision_score(targets_arr, preds_arr, zero_division=0))
        rec = float(recall_score(targets_arr, preds_arr, zero_division=0))
        f1 = float(f1_score(targets_arr, preds_arr, zero_division=0))

        # Specificity
        if len(np.unique(targets_arr)) > 1:
            tn, fp, fn, tp = confusion_matrix(targets_arr, preds_arr).ravel()
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            try:
                auc = float(roc_auc_score(targets_arr, probs_arr))
            except ValueError:
                auc = 0.5
            try:
                pr_auc = float(average_precision_score(targets_arr, probs_arr))
            except ValueError:
                pr_auc = 0.5
        else:
            spec = 0.0
            auc = 0.5
            pr_auc = 0.5

        metrics = {
            'loss': avg_loss,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'specificity': spec,
            'f1': f1,
            'roc_auc': auc,
            'pr_auc': pr_auc,
            'num_samples': num_samples,
            'y_true': targets_arr.tolist(),
            'y_pred': preds_arr.tolist(),
            'y_prob': probs_arr.tolist()
        }

        return avg_loss, num_samples, metrics

    def get_client_checkpoint(self) -> Dict[str, Any]:
        """
        Exports client-specific state distinguishing private encoder and shared predictor.
        """
        return {
            'hospital_id': self.client_id,
            'hospital_name': self.client_name,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'encoder_state_dict': self.model.get_encoder_state_dict(),
            'shared_predictor_state_dict': self.model.get_shared_state_dict(),
            'feature_names': self.schema.feature_names,
            'feature_types': self.schema.feature_types
        }

    def load_client_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restores client-specific private encoder and shared predictor states."""
        if checkpoint['hospital_id'] != self.client_id:
            raise ValueError(f"Checkpoint hospital_id '{checkpoint['hospital_id']}' != client_id '{self.client_id}'.")
        self.model.set_encoder_state_dict(checkpoint['encoder_state_dict'])
        self.model.set_shared_state_dict(checkpoint['shared_predictor_state_dict'])
