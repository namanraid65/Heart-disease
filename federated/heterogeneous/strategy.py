"""
Federated Aggregation Strategy for Heterogeneous-Feature Clients
Aggregates ONLY the shared predictor parameters using sample-weighted FedAvg.
Hospital-specific private encoders are strictly excluded from aggregation.
Enforces strict parameter key and shape validation before averaging.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from federated.utils import verify_privacy_and_data_locality


class HeterogeneousFedAvgStrategy:
    """
    Sample-weighted Federated Averaging Strategy for Heterogeneous-Feature Federated Learning.
    Validates parameter counts and shapes against the shared predictor architecture.
    """

    def __init__(
        self,
        expected_parameter_shapes: Optional[List[Tuple[int, ...]]] = None,
        strategy_name: str = "HeterogeneousFedAvg"
    ):
        self.strategy_name = strategy_name
        self.expected_parameter_shapes = expected_parameter_shapes
        self.round_history: List[Dict[str, Any]] = []

    def set_expected_parameter_shapes(self, shapes: List[Tuple[int, ...]]) -> None:
        """Sets or updates the expected tensor shapes of the shared predictor."""
        self.expected_parameter_shapes = shapes

    def validate_client_parameters(
        self,
        client_id: str,
        parameters: List[np.ndarray]
    ) -> None:
        """
        Validates client parameter payload:
          1. Number of parameter arrays must match the shared predictor layer count.
          2. Shape of each array must exactly match the shared predictor layer shape.
          3. Rejects illegal dimensions or non-numeric arrays.
        """
        if not isinstance(parameters, list) or len(parameters) == 0:
            raise ValueError(
                f"Client '{client_id}' sent an empty or invalid parameter payload: {type(parameters)}."
            )

        if self.expected_parameter_shapes is not None:
            if len(parameters) != len(self.expected_parameter_shapes):
                raise ValueError(
                    f"Parameter count mismatch from client '{client_id}': "
                    f"shared predictor has {len(self.expected_parameter_shapes)} layers, "
                    f"received {len(parameters)} layers. Ensure private encoder parameters are NOT transmitted."
                )

            for i, (param, expected_shape) in enumerate(zip(parameters, self.expected_parameter_shapes)):
                if param.shape != expected_shape:
                    raise ValueError(
                        f"Parameter shape mismatch from client '{client_id}' at layer {i}: "
                        f"expected shape {expected_shape}, got {param.shape}. "
                        f"Do NOT silently reshape or transmit encoder parameters."
                    )

        # Audit privacy (no patient features or non-numeric items)
        verify_privacy_and_data_locality(parameters)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[str, List[np.ndarray], int, Dict[str, Any]]],
        current_global_parameters: Optional[List[np.ndarray]] = None,
        privacy_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Aggregates local shared predictor updates from participating clients.

        Args:
            server_round: Current communication round index.
            results: List of (client_id, shared_parameters, num_samples, metrics)
            current_global_parameters: Current global parameters (optional for FedAvg, used by FedOpt).

        Returns:
            Tuple of (aggregated_parameters, round_summary)
        """
        if not results:
            raise ValueError(f"Round {server_round}: No client results received for aggregation.")

        total_samples = sum(num_samples for _, _, num_samples, _ in results)
        if total_samples <= 0:
            raise ValueError(f"Round {server_round}: Total participating sample count must be positive, got {total_samples}.")

        client_weights_log: Dict[str, Dict[str, Any]] = {}
        validated_updates: List[Tuple[List[np.ndarray], int]] = []

        # Validate each client update before aggregation
        for client_id, params, num_samples, metrics in results:
            self.validate_client_parameters(client_id, params)
            validated_updates.append((params, num_samples))

            fraction = num_samples / total_samples
            client_weights_log[client_id] = {
                'samples': num_samples,
                'weight_fraction': float(fraction),
                'train_loss': float(metrics.get('train_loss', 0.0)),
                'train_acc': float(metrics.get('train_accuracy', 0.0))
            }

        # Check if Secure Aggregation simulation is enabled
        sec_agg = bool(privacy_config and privacy_config.get('secure_aggregation', False))
        if sec_agg:
            from federated.heterogeneous.privacy import PairwiseMaskingProtocol
            participating_ids = [cid for cid, _, _, _ in results]
            seed_val = privacy_config.get('seed', 42)
            round_seed = int(((seed_val if seed_val is not None else 42) + server_round * 10007) & 0xFFFFFFFF)

            # In simulated secure aggregation, client updates are pairwise masked
            masked_updates = []
            for (cid, params, num_samples, _) in results:
                weight_fraction = num_samples / total_samples
                masked_update = PairwiseMaskingProtocol.mask_update(
                    update=params,
                    client_id=cid,
                    participating_client_ids=participating_ids,
                    round_seed=round_seed,
                    weight_fraction=weight_fraction
                )
                masked_updates.append(masked_update)

            # Server receives and sums strictly masked updates (all masks cancel out)
            aggregated_parameters = PairwiseMaskingProtocol.aggregate_masked_updates(masked_updates)
        else:
            # Initialize accumulator with zeros matching the first client's parameter shapes
            first_params = validated_updates[0][0]
            aggregated_parameters = [np.zeros_like(layer, dtype=np.float32) for layer in first_params]

            # Compute sample-weighted FedAvg
            for client_params, num_samples in validated_updates:
                weight_fraction = num_samples / total_samples
                for i, layer in enumerate(client_params):
                    aggregated_parameters[i] += layer.astype(np.float32) * weight_fraction

        round_summary = {
            'strategy': self.strategy_name,
            'round': server_round,
            'secure_aggregation': sec_agg,
            'differential_privacy': bool(privacy_config and privacy_config.get('differential_privacy', False)),
            'total_participating_samples': total_samples,
            'num_clients': len(results),
            'client_contributions': client_weights_log
        }
        self.round_history.append(round_summary)

        return aggregated_parameters, round_summary


class HeterogeneousFedProxStrategy(HeterogeneousFedAvgStrategy):
    """
    FedProx Strategy for Heterogeneous-Feature Federated Learning.
    Applies proximal regularization mu * ||w - w_global||^2 on the shared predictor during local training.
    Server performs sample-weighted aggregation on the resulting shared predictor parameters.
    """

    def __init__(
        self,
        proximal_mu: float = 0.01,
        expected_parameter_shapes: Optional[List[Tuple[int, ...]]] = None,
        strategy_name: str = "HeterogeneousFedProx"
    ):
        super().__init__(
            expected_parameter_shapes=expected_parameter_shapes,
            strategy_name=strategy_name
        )
        self.proximal_mu = proximal_mu

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[str, List[np.ndarray], int, Dict[str, Any]]],
        current_global_parameters: Optional[List[np.ndarray]] = None,
        privacy_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        aggregated_params, summary = super().aggregate_fit(
            server_round=server_round,
            results=results,
            current_global_parameters=current_global_parameters,
            privacy_config=privacy_config
        )
        summary['proximal_mu'] = self.proximal_mu
        return aggregated_params, summary


class HeterogeneousFedOptStrategy(HeterogeneousFedAvgStrategy):
    """
    Base class for Server-Side Adaptive Federated Optimization (FedOpt)
    operating exclusively on the shared predictor parameters.
    Computes pseudo-gradients: Delta_t = sum_i (n_i / N) * (w_{i, t} - w_t)
    """

    def __init__(
        self,
        server_lr: float = 0.1,
        tau: float = 1e-3,
        expected_parameter_shapes: Optional[List[Tuple[int, ...]]] = None,
        strategy_name: str = "HeterogeneousFedOpt"
    ):
        super().__init__(
            expected_parameter_shapes=expected_parameter_shapes,
            strategy_name=strategy_name
        )
        self.server_lr = server_lr
        self.tau = tau
        self.m_t: Optional[List[np.ndarray]] = None  # First moment vector
        self.v_t: Optional[List[np.ndarray]] = None  # Second moment vector

    def init_optimizer_state(self, shapes: List[Tuple[int, ...]]) -> None:
        """Initializes server optimizer moments matching predictor layer shapes."""
        self.m_t = [np.zeros(shape, dtype=np.float32) for shape in shapes]
        self.v_t = [np.zeros(shape, dtype=np.float32) for shape in shapes]

    def compute_pseudo_gradient(
        self,
        validated_updates: List[Tuple[List[np.ndarray], int]],
        current_global_parameters: List[np.ndarray],
        total_samples: int,
        client_ids: Optional[List[str]] = None,
        round_seed: Optional[int] = None,
        secure_aggregation: bool = False
    ) -> List[np.ndarray]:
        """
        Computes sample-weighted pseudo-gradient Delta_t:
          Delta_t = sum_i (n_i / N) * (w_{i, t} - w_t)
        If secure_aggregation is True, client updates are pairwise-masked and summed.
        """
        if secure_aggregation and client_ids is not None and round_seed is not None:
            from federated.heterogeneous.privacy import PairwiseMaskingProtocol
            masked_deltas = []
            for (client_params, num_samples), cid in zip(validated_updates, client_ids):
                weight_fraction = num_samples / total_samples
                delta = [
                    (c_l.astype(np.float32) - g_l.astype(np.float32))
                    for c_l, g_l in zip(client_params, current_global_parameters)
                ]
                m_delta = PairwiseMaskingProtocol.mask_update(
                    update=delta,
                    client_id=cid,
                    participating_client_ids=client_ids,
                    round_seed=round_seed,
                    weight_fraction=weight_fraction
                )
                masked_deltas.append(m_delta)
            return PairwiseMaskingProtocol.aggregate_masked_updates(masked_deltas)

        pseudo_grad = [np.zeros_like(layer, dtype=np.float32) for layer in current_global_parameters]
        for client_params, num_samples in validated_updates:
            weight_fraction = num_samples / total_samples
            for i, (client_layer, global_layer) in enumerate(zip(client_params, current_global_parameters)):
                delta = client_layer.astype(np.float32) - global_layer.astype(np.float32)
                pseudo_grad[i] += delta * weight_fraction
        return pseudo_grad


class HeterogeneousFedAdamStrategy(HeterogeneousFedOptStrategy):
    """
    FedAdam Strategy for Heterogeneous-Feature Federated Learning.
    Server maintains Adam moments (m_t, v_t) strictly for the shared predictor parameters.
    """

    def __init__(
        self,
        server_lr: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        expected_parameter_shapes: Optional[List[Tuple[int, ...]]] = None,
        strategy_name: str = "HeterogeneousFedAdam"
    ):
        super().__init__(
            server_lr=server_lr,
            tau=tau,
            expected_parameter_shapes=expected_parameter_shapes,
            strategy_name=strategy_name
        )
        self.beta1 = beta1
        self.beta2 = beta2

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[str, List[np.ndarray], int, Dict[str, Any]]],
        current_global_parameters: Optional[List[np.ndarray]] = None,
        privacy_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not results:
            raise ValueError(f"Round {server_round}: No client results received.")
        if current_global_parameters is None:
            raise ValueError("FedAdam requires current_global_parameters to compute pseudo-gradient Delta_t.")

        total_samples = sum(num_samples for _, _, num_samples, _ in results)
        client_weights_log: Dict[str, Dict[str, Any]] = {}
        validated_updates: List[Tuple[List[np.ndarray], int]] = []

        for client_id, params, num_samples, metrics in results:
            self.validate_client_parameters(client_id, params)
            validated_updates.append((params, num_samples))
            fraction = num_samples / total_samples
            client_weights_log[client_id] = {
                'samples': num_samples,
                'weight_fraction': float(fraction),
                'train_loss': float(metrics.get('train_loss', 0.0)),
                'train_acc': float(metrics.get('train_accuracy', 0.0))
            }

        # Initialize moments if not yet initialized
        if self.m_t is None or self.v_t is None:
            shapes = [p.shape for p in current_global_parameters]
            self.init_optimizer_state(shapes)

        sec_agg = bool(privacy_config and privacy_config.get('secure_aggregation', False))
        participating_ids = [cid for cid, _, _, _ in results]
        seed_val = privacy_config.get('seed', 42) if privacy_config else 42
        round_seed = int(((seed_val if seed_val is not None else 42) + server_round * 10007) & 0xFFFFFFFF)

        # 1. Compute pseudo-gradient Delta_t (supports secure aggregation unmasking)
        delta_t = self.compute_pseudo_gradient(
            validated_updates=validated_updates,
            current_global_parameters=current_global_parameters,
            total_samples=total_samples,
            client_ids=participating_ids,
            round_seed=round_seed,
            secure_aggregation=sec_agg
        )

        # 2. Update server Adam moments & global shared predictor parameters
        updated_parameters: List[np.ndarray] = []
        for i, (g_layer, d_layer) in enumerate(zip(current_global_parameters, delta_t)):
            # First moment: m_t = beta1 * m_{t-1} + (1 - beta1) * Delta_t
            self.m_t[i] = self.beta1 * self.m_t[i] + (1.0 - self.beta1) * d_layer
            # Second moment: v_t = beta2 * v_{t-1} + (1 - beta2) * (Delta_t^2)
            self.v_t[i] = self.beta2 * self.v_t[i] + (1.0 - self.beta2) * np.square(d_layer)
            # Parameter update: w_{t+1} = w_t + server_lr * m_t / (sqrt(v_t) + tau)
            step = self.server_lr * self.m_t[i] / (np.sqrt(self.v_t[i]) + self.tau)
            new_layer = g_layer.astype(np.float32) + step
            updated_parameters.append(new_layer)

        round_summary = {
            'strategy': self.strategy_name,
            'round': server_round,
            'secure_aggregation': sec_agg,
            'differential_privacy': bool(privacy_config and privacy_config.get('differential_privacy', False)),
            'server_lr': self.server_lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'tau': self.tau,
            'total_participating_samples': total_samples,
            'num_clients': len(results),
            'client_contributions': client_weights_log
        }
        self.round_history.append(round_summary)
        return updated_parameters, round_summary


class HeterogeneousFedYogiStrategy(HeterogeneousFedOptStrategy):
    """
    FedYogi Strategy for Heterogeneous-Feature Federated Learning.
    Applies Yogi additive second-moment tracking to avoid aggressive step size shrinking.
    """

    def __init__(
        self,
        server_lr: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        expected_parameter_shapes: Optional[List[Tuple[int, ...]]] = None,
        strategy_name: str = "HeterogeneousFedYogi"
    ):
        super().__init__(
            server_lr=server_lr,
            tau=tau,
            expected_parameter_shapes=expected_parameter_shapes,
            strategy_name=strategy_name
        )
        self.beta1 = beta1
        self.beta2 = beta2

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[str, List[np.ndarray], int, Dict[str, Any]]],
        current_global_parameters: Optional[List[np.ndarray]] = None,
        privacy_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not results:
            raise ValueError(f"Round {server_round}: No client results received.")
        if current_global_parameters is None:
            raise ValueError("FedYogi requires current_global_parameters to compute pseudo-gradient Delta_t.")

        total_samples = sum(num_samples for _, _, num_samples, _ in results)
        client_weights_log: Dict[str, Dict[str, Any]] = {}
        validated_updates: List[Tuple[List[np.ndarray], int]] = []

        for client_id, params, num_samples, metrics in results:
            self.validate_client_parameters(client_id, params)
            validated_updates.append((params, num_samples))
            fraction = num_samples / total_samples
            client_weights_log[client_id] = {
                'samples': num_samples,
                'weight_fraction': float(fraction),
                'train_loss': float(metrics.get('train_loss', 0.0)),
                'train_acc': float(metrics.get('train_accuracy', 0.0))
            }

        if self.m_t is None or self.v_t is None:
            shapes = [p.shape for p in current_global_parameters]
            self.init_optimizer_state(shapes)

        sec_agg = bool(privacy_config and privacy_config.get('secure_aggregation', False))
        participating_ids = [cid for cid, _, _, _ in results]
        seed_val = privacy_config.get('seed', 42) if privacy_config else 42
        round_seed = int(((seed_val if seed_val is not None else 42) + server_round * 10007) & 0xFFFFFFFF)

        delta_t = self.compute_pseudo_gradient(
            validated_updates=validated_updates,
            current_global_parameters=current_global_parameters,
            total_samples=total_samples,
            client_ids=participating_ids,
            round_seed=round_seed,
            secure_aggregation=sec_agg
        )

        updated_parameters: List[np.ndarray] = []
        for i, (g_layer, d_layer) in enumerate(zip(current_global_parameters, delta_t)):
            self.m_t[i] = self.beta1 * self.m_t[i] + (1.0 - self.beta1) * d_layer
            d_sq = np.square(d_layer)
            # Yogi update: v_t = v_{t-1} - (1 - beta2) * Delta_t^2 * sign(v_{t-1} - Delta_t^2)
            self.v_t[i] = self.v_t[i] - (1.0 - self.beta2) * d_sq * np.sign(self.v_t[i] - d_sq)
            step = self.server_lr * self.m_t[i] / (np.sqrt(np.maximum(self.v_t[i], 0.0)) + self.tau)
            new_layer = g_layer.astype(np.float32) + step
            updated_parameters.append(new_layer)

        round_summary = {
            'strategy': self.strategy_name,
            'round': server_round,
            'secure_aggregation': sec_agg,
            'differential_privacy': bool(privacy_config and privacy_config.get('differential_privacy', False)),
            'server_lr': self.server_lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'tau': self.tau,
            'total_participating_samples': total_samples,
            'num_clients': len(results),
            'client_contributions': client_weights_log
        }
        self.round_history.append(round_summary)
        return updated_parameters, round_summary


class HeterogeneousFedAdagradStrategy(HeterogeneousFedOptStrategy):
    """
    FedAdagrad Strategy for Heterogeneous-Feature Federated Learning.
    Server accumulates squared pseudo-gradients strictly for the shared predictor parameters.
    """

    def __init__(
        self,
        server_lr: float = 0.1,
        tau: float = 1e-3,
        expected_parameter_shapes: Optional[List[Tuple[int, ...]]] = None,
        strategy_name: str = "HeterogeneousFedAdagrad"
    ):
        super().__init__(
            server_lr=server_lr,
            tau=tau,
            expected_parameter_shapes=expected_parameter_shapes,
            strategy_name=strategy_name
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[str, List[np.ndarray], int, Dict[str, Any]]],
        current_global_parameters: Optional[List[np.ndarray]] = None,
        privacy_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if not results:
            raise ValueError(f"Round {server_round}: No client results received.")
        if current_global_parameters is None:
            raise ValueError("FedAdagrad requires current_global_parameters to compute pseudo-gradient Delta_t.")

        total_samples = sum(num_samples for _, _, num_samples, _ in results)
        client_weights_log: Dict[str, Dict[str, Any]] = {}
        validated_updates: List[Tuple[List[np.ndarray], int]] = []

        for client_id, params, num_samples, metrics in results:
            self.validate_client_parameters(client_id, params)
            validated_updates.append((params, num_samples))
            fraction = num_samples / total_samples
            client_weights_log[client_id] = {
                'samples': num_samples,
                'weight_fraction': float(fraction),
                'train_loss': float(metrics.get('train_loss', 0.0)),
                'train_acc': float(metrics.get('train_accuracy', 0.0))
            }

        if self.v_t is None:
            shapes = [p.shape for p in current_global_parameters]
            self.v_t = [np.zeros(shape, dtype=np.float32) for shape in shapes]

        sec_agg = bool(privacy_config and privacy_config.get('secure_aggregation', False))
        participating_ids = [cid for cid, _, _, _ in results]
        seed_val = privacy_config.get('seed', 42) if privacy_config else 42
        round_seed = int(((seed_val if seed_val is not None else 42) + server_round * 10007) & 0xFFFFFFFF)

        delta_t = self.compute_pseudo_gradient(
            validated_updates=validated_updates,
            current_global_parameters=current_global_parameters,
            total_samples=total_samples,
            client_ids=participating_ids,
            round_seed=round_seed,
            secure_aggregation=sec_agg
        )

        updated_parameters: List[np.ndarray] = []
        for i, (g_layer, d_layer) in enumerate(zip(current_global_parameters, delta_t)):
            # Adagrad: v_t = v_{t-1} + Delta_t^2
            self.v_t[i] = self.v_t[i] + np.square(d_layer)
            step = self.server_lr * d_layer / (np.sqrt(self.v_t[i]) + self.tau)
            new_layer = g_layer.astype(np.float32) + step
            updated_parameters.append(new_layer)

        round_summary = {
            'strategy': self.strategy_name,
            'round': server_round,
            'secure_aggregation': sec_agg,
            'differential_privacy': bool(privacy_config and privacy_config.get('differential_privacy', False)),
            'server_lr': self.server_lr,
            'tau': self.tau,
            'total_participating_samples': total_samples,
            'num_clients': len(results),
            'client_contributions': client_weights_log
        }
        self.round_history.append(round_summary)
        return updated_parameters, round_summary
