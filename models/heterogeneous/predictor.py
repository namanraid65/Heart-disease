"""
Shared Federated Predictor Module
Operates exclusively on the common latent space [N, Z] and outputs class logits [N, 1].
This is the ONLY component federated and aggregated across hospitals via FedAvg.
Every participating hospital shares identical parameter shapes for this predictor.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class SharedPredictor(nn.Module):
    """
    Shared classification head receiving common latent representation Z and outputting logit.

    Architecture:
      Latent (Z) -> [Linear -> Activation -> Dropout] x L -> Linear -> Logit (1)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        output_dim: int = 1,
        activation: str = "relu"
    ):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [32]
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim

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
        prev_dim = self.latent_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))
            prev_dim = hidden_dim

        # Final classification projection layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from latent representation Z to binary logit.

        Args:
            z: Latent tensor of shape [batch_size, latent_dim]

        Returns:
            Logit tensor of shape [batch_size, output_dim]
        """
        if z.ndim != 2:
            raise ValueError(
                f"SharedPredictor expected 2D input [batch_size, latent_dim], got shape {list(z.shape)}"
            )

        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"SharedPredictor input dimension mismatch: expected latent_dim={self.latent_dim}, "
                f"got tensor with {z.shape[1]} features."
            )

        logits = self.network(z)
        return logits
