"""
Heterogeneous Federated Server Module
Coordinates global shared predictor synchronization across heterogeneous hospital nodes.
Contains ONLY shared predictor parameters; never stores or aggregates client private encoders.
Enforces strict parameter key and shape validation on all incoming client updates.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Any, Tuple, Optional
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from models.heterogeneous.predictor import SharedPredictor
from federated.heterogeneous.config import (
    LATENT_DIM,
    PREDICTOR_HIDDEN_DIMS,
    DROPOUT_RATE,
    DEVICE,
    RANDOM_SEED,
    HETEROGENEOUS_CHECKPOINTS_DIR
)
from federated.heterogeneous.strategy import HeterogeneousFedAvgStrategy
from federated.utils import verify_privacy_and_data_locality


class HeterogeneousFederatedServer:
    """
    Central coordinator for Heterogeneous-Feature Federated Learning.
    Maintains strictly the global shared predictor over the common latent space Z.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = DROPOUT_RATE,
        strategy: Optional[HeterogeneousFedAvgStrategy] = None,
        privacy_config: Optional[Dict[str, Any]] = None,
        device: str = DEVICE
    ):
        self.device = device
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or PREDICTOR_HIDDEN_DIMS
        self.dropout_rate = dropout_rate
        self.privacy_config = privacy_config
        self.epsilon: Optional[float] = None
        self.privacy_accountant_status: str = "Differential privacy disabled."

        # Initialize Global Shared Predictor
        torch.manual_seed(RANDOM_SEED)
        self.shared_predictor = SharedPredictor(
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Expected layer shapes for parameter validation
        expected_shapes = [val.detach().cpu().numpy().shape for val in self.shared_predictor.state_dict().values()]
        self.strategy = strategy if strategy is not None else HeterogeneousFedAvgStrategy(expected_parameter_shapes=expected_shapes)
        self.strategy.set_expected_parameter_shapes(expected_shapes)

        self.current_round = 0
        self.round_checkpoints: Dict[int, Path] = {}

    def get_global_parameters(self) -> List[np.ndarray]:
        """
        Extracts current global shared predictor weights as NumPy arrays.
        """
        state = self.shared_predictor.state_dict()
        params = [val.detach().cpu().numpy().copy() for val in state.values()]
        verify_privacy_and_data_locality(params)
        return params

    def set_global_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Updates global shared predictor weights with aggregated NumPy arrays.
        """
        expected_keys = list(self.shared_predictor.state_dict().keys())
        if len(parameters) != len(expected_keys):
            raise ValueError(
                f"Parameter length mismatch: shared predictor has {len(expected_keys)} layers, "
                f"received {len(parameters)}."
            )

        new_state = OrderedDict()
        current_state = self.shared_predictor.state_dict()

        for key, arr in zip(expected_keys, parameters):
            expected_shape = current_state[key].shape
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Parameter shape mismatch for key '{key}': "
                    f"expected {list(expected_shape)}, got {list(arr.shape)}."
                )
            new_state[key] = torch.as_tensor(
                arr,
                dtype=current_state[key].dtype,
                device=current_state[key].device
            )

        self.shared_predictor.load_state_dict(new_state, strict=True)

    def aggregate_round(
        self,
        server_round: int,
        client_results: List[Tuple[str, List[np.ndarray], int, Dict[str, Any]]],
        privacy_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Aggregates client updates into an updated global shared predictor.
        """
        self.current_round = server_round
        active_privacy = privacy_config or getattr(self, 'privacy_config', None)

        # 1. Perform strategy aggregation on shared predictor parameters
        current_params = self.get_global_parameters()
        aggregated_weights, round_summary = self.strategy.aggregate_fit(
            server_round=server_round,
            results=client_results,
            current_global_parameters=current_params,
            privacy_config=active_privacy
        )

        # 2. Update global shared predictor weights
        self.set_global_parameters(aggregated_weights)

        # 3. Privacy Accounting (RDP)
        if active_privacy and active_privacy.get('differential_privacy', False):
            from federated.heterogeneous.privacy import RDPAccountant
            noise_mult = float(active_privacy.get('noise_multiplier', 0.0))
            delta = float(active_privacy.get('delta', 1e-5))
            eps, status = RDPAccountant.compute_accumulated_epsilon(
                noise_multiplier=noise_mult,
                num_rounds=server_round,
                delta=delta
            )
            self.epsilon = eps
            self.privacy_accountant_status = status
            round_summary['epsilon'] = eps
            round_summary['privacy_accountant_status'] = status
        else:
            self.epsilon = None
            self.privacy_accountant_status = "Differential privacy disabled."
            round_summary['epsilon'] = None
            round_summary['privacy_accountant_status'] = self.privacy_accountant_status

        # 4. Save round checkpoint of the shared predictor
        checkpoint_path = self.save_checkpoint(round_num=server_round, is_final=False)
        round_summary['checkpoint_path'] = str(checkpoint_path)

        return round_summary

    def save_checkpoint(
        self,
        round_num: int,
        is_final: bool = False,
        metrics: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None
    ) -> Path:
        """
        Saves global shared predictor checkpoint to disk.
        Does NOT store any client's private encoder as 'global model'.
        """
        HETEROGENEOUS_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        base_label = experiment_name or getattr(self.strategy, 'strategy_name', 'HeterogeneousFedAvg')
        strategy_label = base_label.lower()
        if is_final:
            filename = f"global_{strategy_label}_predictor_final.pt"
        else:
            filename = f"global_{strategy_label}_predictor_round_{round_num:02d}.pt"

        checkpoint_file = HETEROGENEOUS_CHECKPOINTS_DIR / filename
        payload = {
            'model_type': 'Federated_Heterogeneous_SharedPredictor',
            'strategy': getattr(self.strategy, 'strategy_name', 'HeterogeneousFedAvg'),
            'shared_predictor_state_dict': self.shared_predictor.state_dict(),
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'round': round_num,
            'is_final': is_final,
            'seed': RANDOM_SEED,
            'privacy_config': getattr(self, 'privacy_config', None),
            'privacy_accountant_status': getattr(self, 'privacy_accountant_status', None),
            'epsilon': getattr(self, 'epsilon', None),
            'metrics': metrics or {}
        }
        torch.save(payload, checkpoint_file)
        self.round_checkpoints[round_num] = checkpoint_file
        return checkpoint_file

    def save_client_checkpoint(
        self,
        client_dict: Dict[str, Any],
        round_num: int,
        is_final: bool = False
    ) -> Path:
        """
        Saves a hospital client's paired state (private encoder + shared predictor).
        """
        HETEROGENEOUS_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        cid = client_dict['hospital_id']
        suffix = "final" if is_final else f"round_{round_num:02d}"
        filename = f"{cid}_heterogeneous_{suffix}.pt"
        checkpoint_file = HETEROGENEOUS_CHECKPOINTS_DIR / filename

        torch.save(client_dict, checkpoint_file)
        return checkpoint_file
