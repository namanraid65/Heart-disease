"""
Federated Server Module
Manages global 1D AlexNet model state, orchestrates client communication rounds,
executes weighted FedAvg aggregation, and serializes round checkpoints.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from federated.config import (
    FEDERATED_CHECKPOINTS_DIR,
    INPUT_FEATURES,
    DROPOUT_RATE,
    DEVICE,
    RANDOM_SEED
)
from federated.utils import (
    get_model_parameters,
    set_model_parameters,
    get_model_shared_parameters,
    set_model_shared_parameters,
    verify_privacy_and_data_locality
)
from federated.strategy import FedAvgStrategy
from models.alexnet_1d import AlexNet1D, build_alexnet_1d


class FederatedServer:
    """
    Central Federated Coordinator under FedBN.
    Coordinates global shared parameter synchronization across hospital nodes.
    Preserves and never averages client-local BatchNorm statistics.
    Never accesses or receives raw patient records.
    """

    def __init__(
        self,
        strategy: Optional[FedAvgStrategy] = None,
        device: str = DEVICE
    ):
        self.device = device
        self.strategy = strategy if strategy is not None else FedAvgStrategy()

        # Initialize Global 1D AlexNet
        torch.manual_seed(RANDOM_SEED)
        self.global_model = build_alexnet_1d(input_dim=INPUT_FEATURES, dropout_rate=DROPOUT_RATE).to(self.device)
        self.current_round = 0
        self.round_checkpoints = {}

    def get_global_parameters(self) -> List[np.ndarray]:
        """Extracts current global shared model weights as NumPy ndarrays (excludes BN buffers)."""
        return get_model_shared_parameters(self.global_model)

    def set_global_parameters(self, parameters: List[np.ndarray]) -> None:
        """Sets global shared model weights from NumPy ndarrays."""
        set_model_shared_parameters(self.global_model, parameters)

    def aggregate_round(
        self,
        server_round: int,
        client_results: List[Tuple[str, List[np.ndarray], int, Dict[str, Any]]],
        client_bn_states: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, Any]:
        """
        Aggregates client updates into a new global model for the communication round.
        """
        self.current_round = server_round

        # 1. Perform weighted FedAvg aggregation on shared parameters
        aggregated_weights, round_summary = self.strategy.aggregate_fit(server_round, client_results)

        # 2. Update global model shared parameters
        self.set_global_parameters(aggregated_weights)

        # 3. Save round checkpoint
        checkpoint_path = self.save_checkpoint(
            round_num=server_round,
            is_final=False,
            client_bn_states=client_bn_states
        )
        round_summary['checkpoint_path'] = str(checkpoint_path)

        return round_summary

    def save_checkpoint(
        self,
        round_num: int,
        is_final: bool = False,
        client_bn_states: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ) -> Path:
        """
        Saves global AlexNet checkpoint to disk.
        Under FedBN, stores global shared parameters and explicit hospital-specific BN states.
        """
        FEDERATED_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        if is_final:
            filename = "global_alexnet_final.pt"
        else:
            filename = f"global_alexnet_round_{round_num:02d}.pt"

        checkpoint_file = FEDERATED_CHECKPOINTS_DIR / filename
        payload = {
            'model_state_dict': self.global_model.state_dict(),
            'model_type': 'Federated_1D_AlexNet',
            'round': round_num,
            'is_final': is_final,
            'input_dim': INPUT_FEATURES,
            'dropout_rate': DROPOUT_RATE,
            'seed': RANDOM_SEED,
            'client_bn_states': client_bn_states
        }
        torch.save(payload, checkpoint_file)
        self.round_checkpoints[round_num] = checkpoint_file
        return checkpoint_file


if __name__ == '__main__':
    server = FederatedServer()
    params = server.get_global_parameters()
    print(f"FederatedServer initialized successfully! Parameter layers: {len(params)}")
