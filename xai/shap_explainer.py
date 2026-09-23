"""
SHAP Explainer Module for Federated Deep Learning Models (1D AlexNet & 1D ResNet)
Utilizes SHAP KernelExplainer with a representative background summary distribution
to compute game-theoretic Shapley feature attributions.
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
    NUM_PROCESSED_FEATURES,
    FEATURE_DISPLAY_NAMES,
    SHAP_FIGURES_DIR,
    SHAP_BACKGROUND_SAMPLES,
    RANDOM_SEED
)


class DeepLearningShapExplainer:
    """
    Computes SHAP feature attributions for PyTorch tabular deep learning models.
    Uses KernelExplainer with a k-means or medoid background summary for efficiency.
    """

    def __init__(
        self,
        predict_fn,
        background_data: np.ndarray,
        feature_names: List[str] = PROCESSED_FEATURE_NAMES,
        n_background: int = SHAP_BACKGROUND_SAMPLES,
        random_state: int = RANDOM_SEED
    ):
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.random_state = random_state

        # Summarize background data using shap.sample or kmeans
        bg_arr = np.asarray(background_data, dtype=np.float32)
        if len(bg_arr) > n_background:
            np.random.seed(self.random_state)
            indices = np.random.choice(len(bg_arr), size=n_background, replace=False)
            self.background = bg_arr[indices]
        else:
            self.background = bg_arr

        # Initialize KernelExplainer (evaluating on disease probability column [:, 1])
        def model_probability_output(x):
            arr = np.asarray(x, dtype=np.float32)
            probs = self.predict_fn(arr)
            # Ensure probabilities are extracted correctly and bounded in [0, 1]
            if isinstance(probs, np.ndarray):
                if probs.ndim == 2 and probs.shape[1] == 2:
                    return probs[:, 1]
                elif probs.ndim == 1:
                    return probs
            return probs[:, 1]

        self.explainer = shap.KernelExplainer(
            model=model_probability_output,
            data=self.background
        )

    def explain_instance(
        self,
        sample: np.ndarray,
        nsamples: int = 200
    ) -> Dict[str, Any]:
        """
        Computes SHAP values for a single patient record.
        sample: (25,) array or (1, 25) array.
        """
        sample_flat = np.asarray(sample, dtype=np.float32).flatten()
        if len(sample_flat) != NUM_PROCESSED_FEATURES:
            raise ValueError(f"Expected {NUM_PROCESSED_FEATURES} features, got {len(sample_flat)}")
        sample_arr = sample_flat.reshape(1, -1)

        shap_vals = self.explainer.shap_values(sample_arr, nsamples=nsamples)

        # Handle different return formats from shap
        if isinstance(shap_vals, list):
            vals = np.array(shap_vals[0]).flatten()
        elif isinstance(shap_vals, np.ndarray):
            vals = shap_vals.flatten()
        else:
            vals = np.array(shap_vals).flatten()

        expected_val = float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.5
        probs = self.predict_fn(sample_arr)[0]
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
            'predicted_label': pred_label,
            'prob_healthy': prob_healthy,
            'prob_disease': prob_disease,
            'expected_value': expected_val,
            'shap_values': vals,
            'feature_shap_dict': feature_shap_dict,
            'sorted_features': sorted_features,
            'top_positive': top_positive,
            'top_negative': top_negative
        }

    def explain_cohort(
        self,
        X_test: np.ndarray,
        nsamples: int = 150
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Computes SHAP values across an entire hospital test split.
        """
        X_arr = np.asarray(X_test, dtype=np.float32)
        shap_matrix = self.explainer.shap_values(X_arr, nsamples=nsamples)
        if isinstance(shap_matrix, list):
            shap_matrix = shap_matrix[0]

        # Mean absolute SHAP importance
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        df_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_|SHAP|': mean_abs_shap
        }).sort_values('Mean_|SHAP|', ascending=False).reset_index(drop=True)

        return shap_matrix, df_importance

    def plot_instance_explanation(
        self,
        explanation_result: Dict[str, Any],
        client_name: str,
        model_name: str,
        sample_id: str,
        actual_label: Optional[int] = None,
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Generates a local SHAP waterfall/bar chart for a specific patient instance.
        """
        SHAP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        if save_path is None:
            save_path = SHAP_FIGURES_DIR / f"{client_name.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}_{sample_id}_shap.png"

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

        plt.title(
            f"SHAP Local Attribution: {model_name} on {client_name} ({sample_id})\n"
            f"Predicted: {pred_lbl} (P={explanation_result['prob_disease']:.1%}){act_lbl_str}",
            fontsize=11,
            fontweight='bold'
        )
        plt.xlabel("SHAP Attribution (Probability Contribution Relative to Expected Value)", fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5, axis='x')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', edgecolor='black', label='Increases Model Probability (+)'),
            Patch(facecolor='#1f77b4', edgecolor='black', label='Decreases Model Probability (-)')
        ]
        plt.legend(handles=legend_elements, loc='lower right', frameon=True)

        plt.figtext(
            0.5, -0.02,
            "Note: SHAP values represent statistical attribution toward model output; they do not establish clinical causality.",
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
        model_name: str,
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Renders SHAP global feature importance bar plot for the cohort.
        """
        SHAP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        if save_path is None:
            save_path = SHAP_FIGURES_DIR / f"{client_name.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}_global_shap.png"

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        indices = np.argsort(mean_abs_shap)[::-1][:15]

        top_feats = [self.feature_names[i] for i in indices][::-1]
        top_vals = [mean_abs_shap[i] for i in indices][::-1]
        display_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in top_feats]

        plt.figure(figsize=(9, 6.5))
        plt.barh(display_labels, top_vals, color='#2ca02c', alpha=0.85, edgecolor='black')
        plt.title(f"Global SHAP Feature Importance: {model_name}\nCohort: {client_name}", fontsize=11, fontweight='bold')
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


if __name__ == '__main__':
    from xai.model_loader import load_federated_model
    from models.dataset import get_client_dataloaders

    loaders = get_client_dataloaders('hospital_1', batch_size=16)
    train_x = loaders['train'].dataset.features.numpy()
    test_x = loaders['test'].dataset.features.numpy()

    wrapper = load_federated_model('Federated_1D_AlexNet')
    explainer = DeepLearningShapExplainer(wrapper.predict_proba, train_x, n_background=20)
    res = explainer.explain_instance(test_x[0], nsamples=50)
    print("SHAP Single Instance Test Complete:")
    print(f"  Top SHAP Features: {res['sorted_features'][:5]}")
