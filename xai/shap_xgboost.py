"""
SHAP Explainer Module for Local XGBoost Baselines and Probability Ensemble
Provides:
  1. LocalXGBoostTreeExplainer: Exact TreeSHAP for individual hospital-specific XGBoost models.
  2. LocalXGBoostEnsembleKernelExplainer: Game-theoretic attributions for the full sample-weighted
     probability ensemble without mathematically invalid log-odds averaging.
  3. SampleWeightedXGBoostShapExplainer: Unified interface for explaining both the probability ensemble
     and constituent hospital models.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from xai.config import (
    PROCESSED_FEATURE_NAMES,
    FEATURE_DISPLAY_NAMES,
    SHAP_FIGURES_DIR,
    RANDOM_SEED
)
from models.xgboost_model import LocalXGBoostModel
from xai.model_loader import SampleWeightedXGBoostEnsemble


class LocalXGBoostTreeExplainer:
    """
    Computes exact TreeSHAP values for an individual hospital's local XGBoost model.
    Operates strictly within that hospital's own preprocessed feature space:
      - Uses that hospital's trained model
      - Aligns with that hospital's feature scaling & imputation
    """

    def __init__(
        self,
        model: LocalXGBoostModel,
        client_id: str,
        client_name: str,
        feature_names: List[str] = PROCESSED_FEATURE_NAMES
    ):
        self.model = model
        self.client_id = client_id
        self.client_name = client_name
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model.clf)

    def explain_instance(self, sample: np.ndarray) -> Dict[str, Any]:
        """
        Computes TreeSHAP attributions for a single patient record.
        sample: (25,) array or (1, 25) array.
        """
        sample_flat = np.asarray(sample, dtype=np.float32).flatten()
        if len(sample_flat) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(sample_flat)}")
        sample_arr = sample_flat.reshape(1, -1)

        c_shap = self.explainer.shap_values(sample_arr)
        if isinstance(c_shap, list):
            c_shap = c_shap[1]  # Disease class if returned as list
        vals = np.asarray(c_shap, dtype=np.float32).flatten()

        probs = self.model.predict_proba(sample_arr)[0]
        prob_disease = float(probs[1])
        prob_healthy = float(probs[0])
        pred_label = int(prob_disease >= 0.5)

        feature_shap_dict = {
            self.feature_names[i]: float(vals[i]) for i in range(len(self.feature_names))
        }
        sorted_features = sorted(feature_shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive = [f for f, v in sorted_features if v > 0]
        top_negative = [f for f, v in sorted_features if v < 0]

        return {
            'hospital': self.client_id,
            'hospital_name': self.client_name,
            'model_type': 'Local_XGBoost_TreeSHAP',
            'attribution_space': 'tree_margin_log_odds',
            'predicted_label': pred_label,
            'prob_healthy': prob_healthy,
            'prob_disease': prob_disease,
            'shap_values': vals,
            'feature_shap_dict': feature_shap_dict,
            'sorted_features': sorted_features,
            'top_positive': top_positive,
            'top_negative': top_negative
        }

    def explain_cohort(self, X_test: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Computes TreeSHAP values across a full hospital test partition.
        """
        X_arr = np.asarray(X_test, dtype=np.float32)
        c_shap = self.explainer.shap_values(X_arr)
        if isinstance(c_shap, list):
            c_shap = c_shap[1]
        shap_matrix = np.asarray(c_shap, dtype=np.float32)

        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        df_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_|SHAP|': mean_abs_shap
        }).sort_values('Mean_|SHAP|', ascending=False).reset_index(drop=True)

        return shap_matrix, df_importance


class LocalXGBoostEnsembleKernelExplainer:
    """
    Computes exact game-theoretic SHAP attributions for the full Sample-Weighted
    XGBoost Ensemble prediction probability function:
      P_ensemble(x) = sum_k w_k P_k(x)
    Operates directly in probability space, preserving the efficiency axiom:
      sum_i phi_i = P_ensemble(x) - E[P_ensemble]
    Avoids mathematically invalid linear averaging of TreeSHAP log-odds.
    """

    def __init__(
        self,
        ensemble_model: SampleWeightedXGBoostEnsemble,
        background_data: np.ndarray,
        feature_names: List[str] = PROCESSED_FEATURE_NAMES,
        n_background: int = 30,
        random_state: int = RANDOM_SEED
    ):
        self.ensemble = ensemble_model
        self.feature_names = feature_names
        self.random_state = random_state

        bg_arr = np.asarray(background_data, dtype=np.float32)
        if len(bg_arr) > n_background:
            np.random.seed(self.random_state)
            indices = np.random.choice(len(bg_arr), size=n_background, replace=False)
            self.background = bg_arr[indices]
        else:
            self.background = bg_arr

        # Prediction function returning disease probability from ensemble
        def ensemble_prob_disease(x):
            arr = np.asarray(x, dtype=np.float32)
            probs = self.ensemble.predict_proba(arr)
            return probs[:, 1]

        self.explainer = shap.KernelExplainer(
            model=ensemble_prob_disease,
            data=self.background
        )

    def explain_instance(self, sample: np.ndarray, nsamples: int = 150) -> Dict[str, Any]:
        """
        Computes SHAP values on the ensemble probability prediction for a single patient record.
        """
        sample_flat = np.asarray(sample, dtype=np.float32).flatten()
        if len(sample_flat) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(sample_flat)}")
        sample_arr = sample_flat.reshape(1, -1)

        shap_vals = self.explainer.shap_values(sample_arr, nsamples=nsamples)
        if isinstance(shap_vals, list):
            vals = np.array(shap_vals[0]).flatten()
        else:
            vals = np.asarray(shap_vals).flatten()

        probs = self.ensemble.predict_proba(sample_arr)[0]
        prob_disease = float(probs[1])
        prob_healthy = float(probs[0])
        pred_label = int(prob_disease >= 0.5)

        feature_shap_dict = {
            self.feature_names[i]: float(vals[i]) for i in range(len(self.feature_names))
        }
        sorted_features = sorted(feature_shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive = [f for f, v in sorted_features if v > 0]
        top_negative = [f for f, v in sorted_features if v < 0]

        expected_val = float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.5

        return {
            'model_type': 'Sample_Weighted_XGBoost_Ensemble_KernelSHAP',
            'attribution_space': 'ensemble_probability',
            'predicted_label': pred_label,
            'prob_healthy': prob_healthy,
            'prob_disease': prob_disease,
            'base_value': expected_val,
            'shap_values': vals,
            'feature_shap_dict': feature_shap_dict,
            'sorted_features': sorted_features,
            'top_positive': top_positive,
            'top_negative': top_negative
        }

    def explain_cohort(self, X_test: np.ndarray, nsamples: int = 150) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Computes SHAP values across an entire cohort using the ensemble probability explainer.
        """
        X_arr = np.asarray(X_test, dtype=np.float32)
        shap_matrix = self.explainer.shap_values(X_arr, nsamples=nsamples)
        if isinstance(shap_matrix, list):
            shap_matrix = shap_matrix[0]
        shap_matrix = np.asarray(shap_matrix, dtype=np.float32)

        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        df_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_|SHAP|': mean_abs_shap
        }).sort_values('Mean_|SHAP|', ascending=False).reset_index(drop=True)

        return shap_matrix, df_importance


class SampleWeightedXGBoostShapExplainer:
    """
    Unified SHAP Explainer for XGBoost.
    Default: Computes probability-space SHAP on the ensemble function using KernelExplainer.
    Also encapsulates LocalXGBoostTreeExplainer instances for each hospital to allow
    inspection of individual client tree models.
    """

    def __init__(
        self,
        ensemble_model: Optional[SampleWeightedXGBoostEnsemble] = None,
        background_data: Optional[np.ndarray] = None,
        feature_names: List[str] = PROCESSED_FEATURE_NAMES,
        client_models: Optional[Dict[str, LocalXGBoostModel]] = None,
        **kwargs
    ):
        self.ensemble = ensemble_model
        self.feature_names = feature_names

        # Per-hospital local tree explainers
        self.local_explainers: Dict[str, LocalXGBoostTreeExplainer] = {}
        c_models = client_models or (self.ensemble.client_models if self.ensemble is not None and hasattr(self.ensemble, 'client_models') else {})
        for cid, model_wrapper in c_models.items():
            self.local_explainers[cid] = LocalXGBoostTreeExplainer(
                model=model_wrapper,
                client_id=cid,
                client_name=cid.replace('_', ' ').title(),
                feature_names=self.feature_names
            )

        # Ensemble probability explainer
        if background_data is not None:
            self.ensemble_explainer = LocalXGBoostEnsembleKernelExplainer(
                ensemble_model=self.ensemble,
                background_data=background_data,
                feature_names=self.feature_names
            )
        else:
            self.ensemble_explainer = None

    def explain_instance(self, sample: np.ndarray, client_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Explains an instance.
        If client_id is specified: uses that hospital's exact LocalXGBoostTreeExplainer.
        Else if ensemble_explainer is initialized: uses ensemble probability KernelExplainer.
        Else: defaults to the specified hospital's local TreeExplainer or 'hospital_1'.
        """
        if client_id is not None and client_id in self.local_explainers:
            return self.local_explainers[client_id].explain_instance(sample)
        if self.ensemble_explainer is not None:
            return self.ensemble_explainer.explain_instance(sample)
        # Fallback to hospital_1 local TreeExplainer if no background provided
        return self.local_explainers['hospital_1'].explain_instance(sample)

    def explain_cohort(
        self,
        X_test: np.ndarray,
        client_id: Optional[str] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Computes cohort feature importances.
        If client_id is specified: computes TreeSHAP on that client's local model.
        Else if ensemble_explainer is available: computes on ensemble.
        """
        if client_id is not None and client_id in self.local_explainers:
            return self.local_explainers[client_id].explain_cohort(X_test)
        if self.ensemble_explainer is not None:
            return self.ensemble_explainer.explain_cohort(X_test)
        return self.local_explainers['hospital_1'].explain_cohort(X_test)

    def plot_instance_explanation(
        self,
        explanation_result: Dict[str, Any],
        client_name: str,
        arg3: str,
        arg4: Optional[Any] = None,
        actual_label: Optional[int] = None,
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Plots horizontal bar chart of local SHAP attributions for the XGBoost model.
        Supports both (result, client_name, sample_id) and (result, client_name, model_name, sample_id).
        """
        if arg4 is not None and isinstance(arg4, str):
            model_name = arg3
            sample_id = arg4
        else:
            sample_id = arg3
            model_name = "Local XGBoost"
            if actual_label is None and isinstance(arg4, (int, np.integer)):
                actual_label = int(arg4)

        SHAP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        if save_path is None:
            save_path = SHAP_FIGURES_DIR / f"{client_name.lower().replace(' ', '_')}_local_xgboost_{sample_id}_shap.png"

        sorted_feats = explanation_result['sorted_features'][:10]
        feats = [x[0] for x in sorted_feats][::-1]
        vals = [x[1] for x in sorted_feats][::-1]
        display_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in feats]
        colors = ['#d62728' if v > 0 else '#1f77b4' for v in vals]

        plt.figure(figsize=(9, 5.5))
        plt.barh(display_labels, vals, color=colors, alpha=0.85, edgecolor='black')
        plt.axvline(0, color='black', linewidth=1, linestyle='--')

        pred_lbl = "Heart Disease (1)" if explanation_result['predicted_label'] == 1 else "Healthy (0)"
        act_lbl_str = f" | True: {'Disease (1)' if actual_label == 1 else 'Healthy (0)'}" if actual_label is not None else ""

        space_label = explanation_result.get('attribution_space', 'log-odds')
        plt.title(
            f"SHAP Attribution: XGBoost on {client_name} ({sample_id})\n"
            f"Predicted: {pred_lbl} (P={explanation_result['prob_disease']:.1%}){act_lbl_str}",
            fontsize=11,
            fontweight='bold'
        )
        plt.xlabel(f"SHAP Value ({space_label} contribution to model prediction)", fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5, axis='x')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', edgecolor='black', label='Increases Model Risk Score (+)'),
            Patch(facecolor='#1f77b4', edgecolor='black', label='Decreases Model Risk Score (-)')
        ]
        plt.legend(handles=legend_elements, loc='lower right', frameon=True)

        plt.figtext(
            0.5, -0.02,
            "Note: Attributions quantify statistical model contribution; they do not establish clinical causality.",
            ha='center', fontsize=8, style='italic', color='#555555'
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_global_summary(
        self,
        shap_values: np.ndarray,
        X_test: np.ndarray,
        client_name: str,
        model_name: Optional[str] = None,
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Renders SHAP global feature importance bar plot.
        """
        SHAP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        if save_path is None:
            save_path = SHAP_FIGURES_DIR / f"{client_name.lower().replace(' ', '_')}_xgboost_global_shap.png"

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        indices = np.argsort(mean_abs_shap)[::-1][:15]

        top_feats = [self.feature_names[i] for i in indices][::-1]
        top_vals = [mean_abs_shap[i] for i in indices][::-1]
        display_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in top_feats]

        plt.figure(figsize=(9, 6.5))
        plt.barh(display_labels, top_vals, color='#ff7f0e', alpha=0.85, edgecolor='black')
        plt.title(f"Global SHAP Feature Importance: XGBoost\nCohort: {client_name}", fontsize=11, fontweight='bold')
        plt.xlabel("Mean |SHAP Value| (Average Attribution Magnitude)", fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5, axis='x')

        plt.figtext(
            0.5, -0.02,
            "Note: Global feature importance reflects average contribution magnitude to model predictions across cohort.",
            ha='center', fontsize=8, style='italic', color='#555555'
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path


# Backward-compatible aliases
FederatedXGBoostShapExplainer = SampleWeightedXGBoostShapExplainer
LocalXGBoostEnsembleShapExplainer = SampleWeightedXGBoostShapExplainer


if __name__ == '__main__':
    from xai.model_loader import load_federated_model
    from models.dataset import get_client_dataloaders

    ensemble = load_federated_model('Local_XGBoost_Ensemble')
    loaders = get_client_dataloaders('hospital_1', batch_size=16)
    train_x = loaders['train'].dataset.features.numpy()
    test_x = loaders['test'].dataset.features.numpy()

    # Test exact local TreeExplainer
    local_tree_explainer = LocalXGBoostTreeExplainer(
        model=ensemble.get_client_model('hospital_1'),
        client_id='hospital_1',
        client_name='Hospital 1 (Cleveland)'
    )
    res_local = local_tree_explainer.explain_instance(test_x[0])
    print("Exact Local TreeSHAP Test:")
    print(f"  Attribution space: {res_local['attribution_space']}")
    print(f"  Top Local Features: {res_local['sorted_features'][:5]}")

    # Test ensemble probability KernelExplainer
    ensemble_explainer = LocalXGBoostEnsembleKernelExplainer(
        ensemble_model=ensemble,
        background_data=train_x[:25]
    )
    res_ens = ensemble_explainer.explain_instance(test_x[0], nsamples=50)
    print("\nEnsemble Probability SHAP Test:")
    print(f"  Attribution space: {res_ens['attribution_space']}")
    print(f"  Top Ensemble Features: {res_ens['sorted_features'][:5]}")
