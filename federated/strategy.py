"""
Federated Aggregation Strategy (FedAvg)
Implements sample-weighted parameter aggregation across independent hospital clients.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Tuple, Dict, Any
import numpy as np

from federated.utils import aggregate_fedavg, verify_privacy_and_data_locality


class FedAvgStrategy:
    """
    Implements Federated Averaging (FedAvg) with sample-proportional weighting.
    """

    def __init__(self, strategy_name: str = "FedAvg"):
        self.strategy_name = strategy_name
        self.round_history = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[str, List[np.ndarray], int, Dict[str, Any]]]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Aggregates local model updates from participating clients.
        results: List of (client_id, parameters, num_samples, metrics)
        """
        if not results:
            raise ValueError(f"Round {server_round}: No client results received for aggregation.")

        total_samples = sum(num_samples for _, _, num_samples, _ in results)

        # Audit incoming updates for data locality & privacy
        fit_pairs = []
        client_weights_log = {}

        for client_id, params, num_samples, metrics in results:
            verify_privacy_and_data_locality(params)
            fit_pairs.append((params, num_samples))
            fraction = num_samples / total_samples
            client_weights_log[client_id] = {
                'samples': num_samples,
                'weight_fraction': float(fraction),
                'train_loss': metrics.get('train_loss', 0.0),
                'train_acc': metrics.get('train_accuracy', 0.0)
            }

        # Perform weighted FedAvg
        aggregated_parameters = aggregate_fedavg(fit_pairs)

        round_summary = {
            'round': server_round,
            'total_participating_samples': total_samples,
            'num_clients': len(results),
            'client_contributions': client_weights_log
        }
        self.round_history.append(round_summary)

        return aggregated_parameters, round_summary
