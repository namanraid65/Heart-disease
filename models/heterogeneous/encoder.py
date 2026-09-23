"""
Hospital-Specific Private Feature Encoder
Maps client-local native feature space [N, D_i] into a unified latent space [N, Z].
The encoder remains strictly private to the hospital institution and is never federated.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class HospitalEncoder(nn.Module):
    """
    Configurable client-local encoder mapping heterogeneous native features to a common latent space.

    Architecture:
      Input (D_i) -> [Linear -> Activation -> Dropout] x L -> Linear -> Latent (Z)

    Enforces strict input and output dimensionality validation.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        activation: str = "relu"
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64]
        self.dropout_rate = dropout_rate

        # Resolve activation function
        if activation.lower() == "relu":
            act_cls = nn.ReLU
        elif activation.lower() == "leaky_relu":
            act_cls = nn.LeakyReLU
        elif activation.lower() == "gelu":
            act_cls = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: '{activation}'. Choose 'relu', 'leaky_relu', or 'gelu'.")

        layers: List[nn.Module] = []
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))
            prev_dim = hidden_dim

        # Final projection to common latent space Z
        layers.append(nn.Linear(prev_dim, self.latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with strict input and output shape validation.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Latent representation tensor of shape [batch_size, latent_dim]
        """
        if x.ndim != 2:
            raise ValueError(
                f"HospitalEncoder expected 2D input [batch_size, input_dim], got shape {list(x.shape)}"
            )

        batch_size, feature_dim = x.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"HospitalEncoder input dimension mismatch: expected {self.input_dim} features, "
                f"got tensor with {feature_dim} features."
            )

        z = self.network(x)

        # Strict latent dimension validation
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"HospitalEncoder output latent dimension mismatch: expected latent_dim={self.latent_dim}, "
                f"got shape {list(z.shape)}"
            )

        return z
