"""
XAI Compatibility Module for Heterogeneous-Feature Federated Learning
Enables client-local explainability (e.g. SHAP, LIME, Integrated Gradients)
to compute feature attributions directly on each hospital's native feature space.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Union, List, Optional
import numpy as np
import torch
import torch.nn as nn

from models.heterogeneous.composite import HeterogeneousCompositeModel


class HeterogeneousLocalXAIWrapper:
    """
    Client-local explainability wrapper around a hospital's composite model
    (HospitalEncoder + SharedPredictor).

    Exposes:
      1. predict_proba(X_numpy): Standard scikit-learn format [N, 2] for SHAP / LIME.
      2. forward_tensor(x_tensor): Differentiable PyTorch forward for gradient-based XAI (Integrated Gradients).
    """

    def __init__(
        self,
        model: HeterogeneousCompositeModel,
        device: str = "cpu"
    ):
        self.model = model
        self.device = device
        self.input_dim = model.input_dim
        self.model.eval()

    def predict_proba(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Black-box probability prediction for LIME and SHAP KernelExplainer.

        Args:
            x: Input array of shape [N, D_i] with native features.

        Returns:
            NumPy array of shape [N, 2] containing [P(y=0), P(y=1)].
        """
        self.model.eval()
        if isinstance(x, np.ndarray):
            tensor_x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            tensor_x = x.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(tensor_x)
            prob_1 = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
            prob_0 = 1.0 - prob_1
            return np.hstack([prob_0, prob_1])

    def compute_input_gradients(
        self,
        x: Union[np.ndarray, torch.Tensor],
        target_class: int = 1
    ) -> np.ndarray:
        """
        Computes input gradients d(logit) / d(x_native) for saliency / Integrated Gradients.

        Args:
            x: Input samples [N, D_i]
            target_class: Target class (1: disease, 0: no disease)

        Returns:
            Gradient array of shape [N, D_i] in native feature space.
        """
        self.model.eval()
        if isinstance(x, np.ndarray):
            tensor_x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            tensor_x = x.to(self.device, dtype=torch.float32).clone()

        tensor_x.requires_grad_(True)
        logits = self.model(tensor_x)

        if target_class == 1:
            target_output = logits.sum()
        else:
            target_output = (-logits).sum()

        target_output.backward()
        grads = tensor_x.grad.detach().cpu().numpy()
        return grads
