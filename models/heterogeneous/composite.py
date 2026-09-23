"""
Heterogeneous Composite Model
Combines client-local private encoder and federated shared predictor.
Maintains strict parameter segregation so only predictor parameters can be serialized for FL.
"""

from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from models.heterogeneous.encoder import HospitalEncoder
from models.heterogeneous.predictor import SharedPredictor


class HeterogeneousCompositeModel(nn.Module):
    """
    Composite model comprising:
      1. Hospital-specific Private Encoder (D_i -> Z)
      2. Globally-shared Federated Predictor (Z -> 1)
    """

    def __init__(
        self,
        encoder: HospitalEncoder,
        predictor: SharedPredictor
    ):
        super().__init__()
        if encoder.latent_dim != predictor.latent_dim:
            raise ValueError(
                f"Dimension mismatch between encoder latent_dim ({encoder.latent_dim}) "
                f"and predictor latent_dim ({predictor.latent_dim})."
            )

        self.encoder = encoder
        self.predictor = predictor
        self.input_dim = encoder.input_dim
        self.latent_dim = encoder.latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Projects client native features into common latent representation Z."""
        return self.encoder(x)

    def predict_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Computes logits directly from latent representation Z."""
        return self.predictor(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass:
          x_native [N, D_i] -> private_encoder -> z [N, Z] -> shared_predictor -> logit [N, 1]
        """
        z = self.encode(x)
        logits = self.predict_from_latent(z)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sigmoid probabilities for class 1."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    # -------------------------------------------------------------------------
    # Federated Parameter Isolation Methods (Exclusively for Shared Predictor)
    # -------------------------------------------------------------------------

    def get_shared_parameter_names(self) -> List[str]:
        """Returns the ordered state_dict keys of the shared predictor ONLY."""
        return list(self.predictor.state_dict().keys())

    def get_shared_parameters(self) -> List[np.ndarray]:
        """
        Extracts exclusively the shared predictor parameters as a list of NumPy arrays.
        Private encoder parameters are NEVER included.
        """
        state = self.predictor.state_dict()
        return [val.detach().cpu().numpy().copy() for val in state.values()]

    def set_shared_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Updates strictly the shared predictor parameters with the aggregated global weights.
        The private encoder parameters remain completely untouched.
        """
        expected_keys = list(self.predictor.state_dict().keys())
        if len(parameters) != len(expected_keys):
            raise ValueError(
                f"Parameter length mismatch: shared predictor has {len(expected_keys)} layers, "
                f"received {len(parameters)}."
            )

        new_state = OrderedDict()
        current_state = self.predictor.state_dict()

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

        self.predictor.load_state_dict(new_state, strict=True)

    def get_shared_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns a cloned copy of the shared predictor state dict."""
        return {k: v.clone().detach().cpu() for k, v in self.predictor.state_dict().items()}

    def set_shared_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Loads shared predictor state dict."""
        self.predictor.load_state_dict(state_dict, strict=True)

    def get_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns a cloned copy of the client-local private encoder state dict."""
        return {k: v.clone().detach().cpu() for k, v in self.encoder.state_dict().items()}

    def set_encoder_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Loads client-local private encoder state dict."""
        self.encoder.load_state_dict(state_dict, strict=True)


def build_heterogeneous_model(
    input_dim: int,
    latent_dim: int = 32,
    encoder_hidden_dims: Optional[List[int]] = None,
    predictor_hidden_dims: Optional[List[int]] = None,
    dropout_rate: float = 0.2
) -> HeterogeneousCompositeModel:
    """Factory helper to build a complete HeterogeneousCompositeModel."""
    encoder = HospitalEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=encoder_hidden_dims if encoder_hidden_dims is not None else [64],
        dropout_rate=dropout_rate
    )
    predictor = SharedPredictor(
        latent_dim=latent_dim,
        hidden_dims=predictor_hidden_dims if predictor_hidden_dims is not None else [32],
        dropout_rate=dropout_rate
    )
    return HeterogeneousCompositeModel(encoder=encoder, predictor=predictor)
