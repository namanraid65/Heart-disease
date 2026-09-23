"""
1D ResNet Architecture for Tabular Clinical Data
Adapted for 1D feature representations in Heart Disease Prediction.
Features residual skip connections, batch normalization, 1D convolutions,
and global average pooling for robust gradient propagation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """
    Reusable 1D Residual Block with Identity and Projection Shortcuts.

    Forward computation:
      x ─────────────────────────────────────┐
      │                                      │ (Shortcut)
      ▼                                      ▼
    Conv1d(k=3, stride=s)                 Conv1d(k=1, stride=s) [if dim changes]
    BatchNorm1d                           BatchNorm1d
    ReLU
    Conv1d(k=3, stride=1)
    BatchNorm1d
      │                                      │
      ▼                                      ▼
      └────────────── ( + ) ─────────────────┘
                        │
                      ReLU
                        ▼
                      Output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ):
        super(ResidualBlock1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Main residual path (2 convolutional layers)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut path (Projection if dimension changes, Identity otherwise)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    1D Residual Network for Tabular Clinical Data.

    Architecture:
      - Stem: Conv1d(1 -> 32, k=3, s=1, p=1), BatchNorm, ReLU
      - Stage 1 (32 channels):  2x ResidualBlock1D(32 -> 32, stride=1)
      - Stage 2 (64 channels):  1x ResidualBlock1D(32 -> 64, stride=2) + 1x ResidualBlock1D(64 -> 64, stride=1)
      - Stage 3 (128 channels): 1x ResidualBlock1D(64 -> 128, stride=2) + 1x ResidualBlock1D(128 -> 128, stride=1)
      - Global Pooling: AdaptiveAvgPool1d(1) -> 128-dim vector
      - Classification Head: Linear(128 -> 32) -> ReLU -> Dropout -> Linear(32 -> 1)
    """

    def __init__(
        self,
        input_dim: int = 25,
        in_channels: int = 1,
        dropout_rate: float = 0.3,
        num_classes: int = 1
    ):
        super(ResNet1D, self).__init__()
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # Stem: Initial Feature Projection
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 32 Channels (Spatial: 25)
        self.layer1 = nn.Sequential(
            ResidualBlock1D(32, 32, stride=1),
            ResidualBlock1D(32, 32, stride=1)
        )

        # Stage 2: 64 Channels (Spatial: 25 -> 13)
        self.layer2 = nn.Sequential(
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64, stride=1)
        )

        # Stage 3: 128 Channels (Spatial: 13 -> 7)
        self.layer3 = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128, stride=1)
        )

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects input of shape (Batch_size, 1, Input_dim) or (Batch_size, Input_dim).
        Returns raw logit tensor of shape (Batch_size, 1).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) != 1 and x.size(2) == 1:
            x = x.permute(0, 2, 1)

        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_pool(out)
        logits = self.classifier(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sigmoid output probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_resnet_1d(
    input_dim: int = 25,
    dropout_rate: float = 0.3
) -> ResNet1D:
    """Factory function to instantiate ResNet1D."""
    return ResNet1D(input_dim=input_dim, dropout_rate=dropout_rate)


if __name__ == '__main__':
    model = build_resnet_1d(input_dim=25, dropout_rate=0.3)
    dummy_input = torch.randn(8, 25)
    output = model(dummy_input)
    print(f"ResNet1D Architecture Instantiated Successfully!")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Total Trainable Parameters: {model.count_parameters():,}")
