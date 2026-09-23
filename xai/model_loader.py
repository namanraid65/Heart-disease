"""
Unified Model Loader Interface for Explainable AI (XAI)
Loads and wraps Federated 1D AlexNet, Federated 1D ResNet, and Local XGBoost Ensemble
with standardized probability prediction interfaces for LIME and SHAP explainers.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBClassifier

from xai.config import (
    ALEXNET_CKPT_PATH,
    RESNET_CKPT_PATH,
    CHECKPOINTS_DIR,
    DEVICE,
    PROCESSED_FEATURE_NAMES,
    NUM_PROCESSED_FEATURES
)
from models.config import PROCESSED_DATA_DIR
from preprocessing.feature_schema import INPUT_FEATURES
from preprocessing.preprocess_client import load_client_preprocessor
from models.alexnet_1d import build_alexnet_1d, AlexNet1D
from models.resnet_1d import build_resnet_1d, ResNet1D
from models.xgboost_model import build_xgboost_model, LocalXGBoostModel


class PyTorchModelWrapper:
    """
    Standardized wrapper for PyTorch deep learning models (AlexNet1D / ResNet1D).
    Converts 2D NumPy arrays into expected (N, 1, 25) tensor inputs and returns (N, 2) probability arrays.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: str = DEVICE
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.model.eval()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Computes 2-class probability matrix of shape (N, 2):
          [:, 0] = Probability of Healthy (Class 0)
          [:, 1] = Probability of Disease (Class 1)
        """
        self.model.eval()
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        tensor_x = torch.tensor(X_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(tensor_x)
            prob_1 = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
            prob_0 = 1.0 - prob_1

        return np.hstack([prob_0, prob_1])

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Computes discrete binary predictions {0, 1}."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def __call__(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.predict_proba(X)


class SampleWeightedXGBoostEnsemble:
    """
    Sample-Weighted Local XGBoost Ensemble Model.

    Combines independent hospital-trained gradient boosted decision tree models
    using sample-proportional weighting:
      P_ensemble(x) = sum_k (n_k / N_total) * P_k(x)
      P_ensemble(x) = 0.4215 * P_H1(x) + 0.4076 * P_H2(x) + 0.1710 * P_H3(x)

    TERMINOLOGY CLARIFICATION:
    This model is a probability ensemble of client-isolated XGBoost baselines, NOT
    federated gradient boosting. True federated tree boosting requires distributed/secure
    histogram aggregation during tree construction.

    PREPROCESSING CONSISTENCY:
    Each constituent hospital model was trained on features scaled and imputed by that hospital's
    specific preprocessor. When raw clinical patient features (13 attributes) are provided,
    each local model transforms the raw input using its own client preprocessor before generating
    probabilities:
      P_ensemble(x_raw) = sum_k w_k * P_k(prep_k(x_raw))
    Preprocessed 25-feature matrices are also supported for backward-compatibility.
    """

    def __init__(
        self,
        checkpoints_dir: Path = CHECKPOINTS_DIR,
        data_dir: Path = PROCESSED_DATA_DIR
    ):
        self.checkpoints_dir = checkpoints_dir
        self.data_dir = data_dir
        self.client_weights = {
            'hospital_1': 212 / 503,  # 0.42147
            'hospital_2': 205 / 503,  # 0.40755
            'hospital_3': 86 / 503    # 0.17097
        }
        self.client_models: Dict[str, LocalXGBoostModel] = {}
        self.client_preprocessors: Dict[str, Any] = {}
        self._load_constituent_models()
        self._load_constituent_preprocessors()

    def _load_constituent_models(self):
        """Loads client XGBoost model checkpoints."""
        for cid in self.client_weights:
            ckpt_path = self.checkpoints_dir / f"{cid}_xgboost.json"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Client XGBoost checkpoint not found at: {ckpt_path}")
            model = build_xgboost_model()
            model.load_model(ckpt_path)
            self.client_models[cid] = model

    def _load_constituent_preprocessors(self):
        """Loads client preprocessor objects for preprocessing consistency."""
        for cid in self.client_weights:
            prep_path = self.data_dir / cid / 'preprocessor.joblib'
            if prep_path.exists():
                try:
                    self.client_preprocessors[cid] = load_client_preprocessor(prep_path)
                except Exception:
                    self.client_preprocessors[cid] = None
            else:
                self.client_preprocessors[cid] = None

    def _is_raw_input(self, X: Any) -> bool:
        """Determines if the input X represents raw clinical features (13 attributes)."""
        if isinstance(X, pd.DataFrame):
            if set(INPUT_FEATURES).issubset(set(X.columns)):
                return True
            if X.shape[1] == len(INPUT_FEATURES) and not any(c in PROCESSED_FEATURE_NAMES for c in X.columns if c not in INPUT_FEATURES):
                return True
        elif isinstance(X, np.ndarray):
            if X.ndim == 1 and len(X) == len(INPUT_FEATURES):
                return True
            if X.ndim == 2 and X.shape[1] == len(INPUT_FEATURES):
                return True
        return False

    def predict_proba_raw(self, raw_df: pd.DataFrame) -> np.ndarray:
        """
        Computes ensemble probabilities from raw clinical patient attributes.
        Each hospital's local model transforms the raw inputs using its own
        native preprocessor (imputation & scaling), ensuring 100% preprocessing consistency:
          P_ensemble(x_raw) = sum_k w_k * P_k(prep_k(x_raw))
        """
        df_clean = raw_df.copy()
        if 'chol' in df_clean.columns:
            df_clean['chol'] = df_clean['chol'].replace(0, np.nan)
        if 'oldpeak' in df_clean.columns:
            df_clean['oldpeak'] = df_clean['oldpeak'].clip(lower=0.0)

        n_samples = len(df_clean)
        aggregated_probs = np.zeros((n_samples, 2), dtype=np.float32)

        for cid, weight in self.client_weights.items():
            model = self.client_models[cid]
            prep = self.client_preprocessors.get(cid)
            if prep is not None:
                X_proc = prep.transform(df_clean)
            else:
                raise RuntimeError(f"Preprocessor for {cid} is required for raw prediction but not found.")
            client_prob = model.predict_proba(X_proc)
            aggregated_probs += client_prob * weight

        return aggregated_probs

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Computes sample-weighted ensemble probability matrix of shape (N, 2).
        If raw features (13 columns) are provided, each constituent model evaluates
        features preprocessed by its own client preprocessor.
        If preprocessed features (25 columns) are provided, evaluates directly across models.
        """
        if self._is_raw_input(X):
            if isinstance(X, np.ndarray):
                arr = X.reshape(1, -1) if X.ndim == 1 else X
                raw_df = pd.DataFrame(arr, columns=INPUT_FEATURES)
            else:
                raw_df = X
            return self.predict_proba_raw(raw_df)

        # Preprocessed (25-dim) input handling
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        aggregated_probs = np.zeros((X_arr.shape[0], 2), dtype=np.float32)
        for cid, weight in self.client_weights.items():
            model = self.client_models[cid]
            client_prob = model.predict_proba(X_arr)
            aggregated_probs += client_prob * weight

        return aggregated_probs

    def predict_raw(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Computes discrete binary predictions {0, 1} for raw clinical patient inputs."""
        probs = self.predict_proba_raw(raw_df)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Computes discrete binary predictions {0, 1}."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_client_model(self, client_id: str) -> LocalXGBoostModel:
        """Returns the underlying XGBoost model for a specific hospital node."""
        return self.client_models[client_id]

    def get_client_preprocessor(self, client_id: str) -> Optional[Any]:
        """Returns the preprocessor object for a specific hospital node."""
        return self.client_preprocessors.get(client_id)


# Backward-compatible aliases
LocalXGBoostEnsemble = SampleWeightedXGBoostEnsemble
FederatedXGBoostModel = SampleWeightedXGBoostEnsemble


def load_federated_model(
    model_name: str,
    device: str = DEVICE
) -> Union[PyTorchModelWrapper, SampleWeightedXGBoostEnsemble]:
    """
    Factory function to load any of the collaborative models.
    Supports:
      - 'Federated_1D_AlexNet'
      - 'Federated_1D_ResNet'
      - 'Local_XGBoost_Ensemble' / 'Sample_Weighted_XGBoost_Ensemble' (and backward-compatible 'Federated_XGBoost')
    """
    if model_name == 'Federated_1D_AlexNet':
        if not ALEXNET_CKPT_PATH.exists():
            raise FileNotFoundError(f"AlexNet checkpoint not found at: {ALEXNET_CKPT_PATH}")
        ckpt = torch.load(ALEXNET_CKPT_PATH, map_location=device, weights_only=False)
        raw_model = build_alexnet_1d(input_dim=NUM_PROCESSED_FEATURES).to(device)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        return PyTorchModelWrapper(raw_model, model_name='Federated 1D AlexNet', device=device)

    elif model_name == 'Federated_1D_ResNet':
        if not RESNET_CKPT_PATH.exists():
            raise FileNotFoundError(f"ResNet checkpoint not found at: {RESNET_CKPT_PATH}")
        ckpt = torch.load(RESNET_CKPT_PATH, map_location=device, weights_only=False)
        raw_model = build_resnet_1d(input_dim=NUM_PROCESSED_FEATURES).to(device)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        return PyTorchModelWrapper(raw_model, model_name='Federated 1D ResNet', device=device)

    elif model_name in ('Federated_XGBoost', 'Sample_Weighted_XGBoost_Ensemble', 'Local_XGBoost_Ensemble'):
        return SampleWeightedXGBoostEnsemble(checkpoints_dir=CHECKPOINTS_DIR)

    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Choose from: 'Federated_1D_AlexNet', "
            f"'Federated_1D_ResNet', 'Local_XGBoost_Ensemble' (or 'Federated_XGBoost')"
        )


if __name__ == '__main__':
    dummy_input = np.random.randn(5, 25).astype(np.float32)
    for mname in ['Federated_1D_AlexNet', 'Federated_1D_ResNet', 'Federated_XGBoost']:
        wrapper = load_federated_model(mname)
        probs = wrapper.predict_proba(dummy_input)
        preds = wrapper.predict(dummy_input)
        print(f"Loaded {mname} successfully! Probs shape: {probs.shape}, Probs[0]: {probs[0]}, Preds: {preds}")
