"""
1D AlexNet Architecture for Tabular Clinical Data
Adapted for 1D feature representations in Heart Disease Prediction.
Features 5 convolutional layers with batch normalization and max-pooling,
followed by a 3-layer fully connected classification head with dropout.
"""

import torch
import torch.nn as nn
from typing import Optional


class AlexNet1D(nn.Module):
    """
    1D AlexNet-style Convolutional Neural Network for Tabular Clinical Data.

    Architecture:
      - Feature Extractor (5 Convolutional Layers):
          Conv1: 1 -> 32 filters (kernel=3, pad=1), BatchNorm, ReLU, MaxPool1d(2)
          Conv2: 32 -> 64 filters (kernel=3, pad=1), BatchNorm, ReLU, MaxPool1d(2)
          Conv3: 64 -> 128 filters (kernel=3, pad=1), BatchNorm, ReLU
          Conv4: 128 -> 128 filters (kernel=3, pad=1), BatchNorm, ReLU
          Conv5: 128 -> 64 filters (kernel=3, pad=1), BatchNorm, ReLU, AdaptiveAvgPool1d(3)
      - Classifier Head (3 Fully Connected Layers):
          Linear(192 -> 64), ReLU, Dropout(p)
          Linear(64 -> 32), ReLU, Dropout(p)
          Linear(32 -> 1) (Single output logit for binary classification)
    """

    def __init__(
        self,
        input_dim: int = 25,
        in_channels: int = 1,
        dropout_rate: float = 0.3,
        num_classes: int = 1
    ):
        super(AlexNet1D, self).__init__()
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # Feature Extractor: 5 Conv1D Blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Block 5
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(3)  # Ensures fixed output size regardless of slight input dimension variance
        )

        # Classifier Head: 3 Fully Connected Layers with Dropout
        flattened_dim = 64 * 3  # 192
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects input x of shape (Batch_size, 1, Input_dim) or (Batch_size, Input_dim).
        Returns raw logit tensor of shape (Batch_size, 1).
        """
        if x.dim() == 2:
            # Reshape (B, D) -> (B, 1, D)
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) != 1 and x.size(2) == 1:
            # Reshape (B, D, 1) -> (B, 1, D)
            x = x.permute(0, 2, 1)

        feat = self.features(x)
        out = self.classifier(feat)
        return out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes predicted probabilities via sigmoid activation.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_alexnet_1d(
    input_dim: int = 25,
    dropout_rate: float = 0.3
) -> AlexNet1D:
    """Factory function to instantiate AlexNet1D."""
    return AlexNet1D(input_dim=input_dim, dropout_rate=dropout_rate)


if __name__ == '__main__':
    # Quick sanity check
    model = build_alexnet_1d(input_dim=25, dropout_rate=0.3)
    dummy_input = torch.randn(8, 25)
    output = model(dummy_input)
    print(f"AlexNet1D Architecture Instantiated Successfully!")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Total Trainable Parameters: {model.count_parameters():,}")
