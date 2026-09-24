"""
Privacy and Security Module for Heterogeneous Federated Learning (Research Prototype)
Implements:
  1. Simulated Secure Aggregation via Pairwise Additive Masking
  2. Client-Side Differential Privacy (L2-norm update clipping + calibrated Gaussian noise)
  3. Formal Rényi Differential Privacy (RDP) Accounting

CRITICAL BOUNDARY GUARANTEE:
  These mechanisms operate EXCLUSIVELY on the communicated shared predictor parameters (Z -> 1).
  Client-local private encoders (D_i -> Z) remain strictly local and NEVER enter this layer.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import hashlib
import numpy as np


@dataclass
class PrivacyConfig:
    """
    Configuration for privacy and security mechanisms.
    """
    secure_aggregation: bool = False
    differential_privacy: bool = False
    max_update_norm: float = 1.0
    noise_multiplier: float = 0.0
    delta: float = 1e-5
    seed: Optional[int] = 42
    dropout_assumption: str = "Secure aggregation prototype assumes all participating clients complete the round."

    def to_dict(self) -> Dict[str, Any]:
        return {
            'secure_aggregation': self.secure_aggregation,
            'differential_privacy': self.differential_privacy,
            'max_update_norm': self.max_update_norm,
            'noise_multiplier': self.noise_multiplier,
            'delta': self.delta,
            'seed': self.seed,
            'dropout_assumption': self.dropout_assumption
        }


def compute_l2_norm(update: List[np.ndarray]) -> float:
    """Computes the overall L2 norm across all parameter layers in an update."""
    sum_sq = sum(float(np.sum(np.square(layer))) for layer in update)
    return float(np.sqrt(sum_sq))


def clip_update(
    update: List[np.ndarray],
    max_norm: float
) -> Tuple[List[np.ndarray], float, float]:
    """
    Clips client update to a maximum L2 norm:
      scale = min(1.0, max_norm / (||update||_2 + 1e-12))
      clipped_update = scale * update

    Returns:
        Tuple of (clipped_update, original_norm, scale_factor)
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")

    orig_norm = compute_l2_norm(update)
    scale = min(1.0, float(max_norm / (orig_norm + 1e-12)))
    clipped = [layer.astype(np.float32) * scale for layer in update]
    return clipped, orig_norm, scale


def add_gaussian_noise(
    update: List[np.ndarray],
    noise_multiplier: float,
    max_norm: float,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Adds calibrated Gaussian noise N(0, (noise_multiplier * max_norm)^2 I)
    to each parameter tensor.
    """
    if noise_multiplier <= 0.0:
        return [layer.copy() for layer in update]

    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    sigma = float(noise_multiplier * max_norm)

    noisy_update = []
    for layer in update:
        noise = rng.normal(loc=0.0, scale=sigma, size=layer.shape).astype(np.float32)
        noisy_update.append(layer.astype(np.float32) + noise)

    return noisy_update


def clip_and_noise_update(
    update: List[np.ndarray],
    max_norm: float,
    noise_multiplier: float,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Applies L2-norm clipping followed by Gaussian perturbation on the communicated update.

    Returns:
        Tuple of (privatized_update, telemetry_dict)
    """
    clipped, orig_norm, scale = clip_update(update, max_norm)
    noised = add_gaussian_noise(clipped, noise_multiplier, max_norm, seed=seed)
    final_norm = compute_l2_norm(noised)

    telemetry = {
        'original_norm': orig_norm,
        'clipping_norm': max_norm,
        'clipping_scale': scale,
        'was_clipped': bool(scale < 1.0),
        'noise_multiplier': noise_multiplier,
        'final_norm': final_norm
    }
    return noised, telemetry


class PairwiseMaskingProtocol:
    """
    Simulated Secure Aggregation via Pairwise Additive Masking.

    For every pair of clients (c_i, c_j) with i < j:
      Generates pseudorandom mask M_{ij}.
      Client i adds +M_{ij}.
      Client j subtracts -M_{ij}.
    When the server sums all masked updates across participating nodes,
    all masks cancel out exactly: sum_i M_i = 0.

    CRITICAL RESEARCH PROTOTYPE NOTE:
      Assumes all participating clients complete the communication round (no dropout resilience).
    """

    @staticmethod
    def derive_pair_seed(cid_a: str, cid_b: str, round_seed: int) -> int:
        """
        Derives a deterministic, process-independent 32-bit pseudorandom seed
        for pair (cid_a, cid_b) using cryptographic SHA-256.
        Guarantees:
          derive_pair_seed(A, B, s) == derive_pair_seed(B, A, s)
          derive_pair_seed(A, B, s) != derive_pair_seed(A, C, s)
        """
        pair_key = sorted([str(cid_a), str(cid_b)])
        key_str = f"secagg_pair::{pair_key[0]}::{pair_key[1]}::{int(round_seed)}"
        digest = hashlib.sha256(key_str.encode('utf-8')).digest()
        return int.from_bytes(digest[:4], byteorder='big')

    @classmethod
    def _generate_pair_mask(
        cls,
        cid_a: str,
        cid_b: str,
        layer_shapes: List[Tuple[int, ...]],
        round_seed: int
    ) -> List[np.ndarray]:
        """
        Generates a deterministic pseudorandom mask for the pair (cid_a, cid_b).
        Ordered deterministically so pair (A, B) and (B, A) generate identical masks.
        """
        combined_seed = cls.derive_pair_seed(cid_a, cid_b, round_seed)
        rng = np.random.RandomState(combined_seed)

        mask_layers = []
        for shape in layer_shapes:
            mask = rng.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)
            mask_layers.append(mask)
        return mask_layers

    @classmethod
    def generate_client_mask(
        cls,
        client_id: str,
        participating_client_ids: List[str],
        layer_shapes: List[Tuple[int, ...]],
        round_seed: int
    ) -> List[np.ndarray]:
        """
        Computes the composite pairwise mask for client_id:
          M_i = sum_{j > i} M_{ij} - sum_{j < i} M_{ji}
        """
        sorted_ids = sorted(participating_client_ids)
        if client_id not in sorted_ids:
            raise ValueError(f"client_id '{client_id}' is not in participating_client_ids: {sorted_ids}")

        i_idx = sorted_ids.index(client_id)
        composite_mask = [np.zeros(shape, dtype=np.float32) for shape in layer_shapes]

        for j_idx, other_id in enumerate(sorted_ids):
            if i_idx == j_idx:
                continue
            pair_mask = cls._generate_pair_mask(client_id, other_id, layer_shapes, round_seed)
            if i_idx < j_idx:
                # Add mask
                for l_idx, m_layer in enumerate(pair_mask):
                    composite_mask[l_idx] += m_layer
            else:
                # Subtract mask
                for l_idx, m_layer in enumerate(pair_mask):
                    composite_mask[l_idx] -= m_layer

        return composite_mask

    @classmethod
    def mask_update(
        cls,
        update: List[np.ndarray],
        client_id: str,
        participating_client_ids: List[str],
        round_seed: int,
        weight_fraction: float = 1.0
    ) -> List[np.ndarray]:
        """
        Applies composite pairwise mask to a client update:
          masked_update = weight_fraction * update + M_i
        """
        layer_shapes = [layer.shape for layer in update]
        mask = cls.generate_client_mask(client_id, participating_client_ids, layer_shapes, round_seed)
        masked_update = []
        for u_layer, m_layer in zip(update, mask):
            masked_update.append(u_layer.astype(np.float32) * float(weight_fraction) + m_layer)
        return masked_update

    @classmethod
    def aggregate_masked_updates(
        cls,
        masked_updates: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Sums masked updates across all participating clients:
          sum_i (weight_i * update_i + M_i) = sum_i weight_i * update_i
        """
        if not masked_updates:
            raise ValueError("No masked updates received for aggregation.")

        aggregated = [np.zeros_like(layer, dtype=np.float32) for layer in masked_updates[0]]
        for client_masked in masked_updates:
            for l_idx, layer in enumerate(client_masked):
                aggregated[l_idx] += layer.astype(np.float32)
        return aggregated

    @classmethod
    def verify_cancellation(
        cls,
        participating_client_ids: List[str],
        layer_shapes: List[Tuple[int, ...]],
        round_seed: int
    ) -> bool:
        """
        Verifies mathematically that the sum of all client composite masks equals zero.
        """
        summed_masks = [np.zeros(shape, dtype=np.float32) for shape in layer_shapes]
        for cid in participating_client_ids:
            c_mask = cls.generate_client_mask(cid, participating_client_ids, layer_shapes, round_seed)
            for l_idx, m_layer in enumerate(c_mask):
                summed_masks[l_idx] += m_layer

        max_err = max(float(np.max(np.abs(m))) for m in summed_masks)
        return bool(max_err < 1e-5)


class RDPAccountant:
    """
    Rényi Differential Privacy (RDP) Accountant for Gaussian mechanism.
    Tracks composition across FL communication rounds and converts to (epsilon, delta).
    """

    @staticmethod
    def compute_gaussian_rdp(alpha: float, sigma: float) -> float:
        """Computes Rényi DP of order alpha for Gaussian noise with multiplier sigma."""
        if sigma <= 0.0 or alpha <= 1.0:
            return float('inf')
        return float(alpha / (2.0 * (sigma ** 2)))

    @classmethod
    def compute_accumulated_epsilon(
        cls,
        noise_multiplier: float,
        num_rounds: int,
        delta: float = 1e-5,
        orders: Optional[List[float]] = None
    ) -> Tuple[Optional[float], str]:
        """
        Computes formal (epsilon, delta) privacy guarantee using RDP composition.
        If noise_multiplier <= 0, returns (None, 'Differential privacy disabled or zero noise added.').
        """
        if noise_multiplier <= 0.0:
            return None, "Differential privacy disabled or zero noise added."
        if num_rounds <= 0:
            return None, "No communication rounds completed."
        if delta <= 0.0 or delta >= 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        if orders is None:
            orders = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0]

        best_eps = float('inf')
        for alpha in orders:
            rdp_per_round = cls.compute_gaussian_rdp(alpha, noise_multiplier)
            total_rdp = num_rounds * rdp_per_round
            # Conversion from RDP to (epsilon, delta): eps = total_rdp + log(1/delta) / (alpha - 1)
            eps = total_rdp + np.log(1.0 / delta) / (alpha - 1.0)
            if eps < best_eps:
                best_eps = eps

        status = f"Formal RDP guarantee computed at delta={delta:.1e} over {num_rounds} rounds."
        return float(best_eps), status
