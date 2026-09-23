"""
LIME (Local Interpretable Model-agnostic Explanations) Explainer Module
Provides instance-level local surrogate explanations for Federated 1D AlexNet,
Federated 1D ResNet, and Local XGBoost Ensemble.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

from xai.config import (
    PROCESSED_FEATURE_NAMES,
    FEATURE_DISPLAY_NAMES,
    LIME_FIGURES_DIR,
    LIME_NUM_SAMPLES,
    RANDOM_SEED
)


class LimeTabularExplainerWrapper:
    """
    Wrapper around lime.lime_tabular.LimeTabularExplainer for clinical tabular inputs.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str] = PROCESSED_FEATURE_NAMES,
        class_names: List[str] = ['Healthy (0)', 'Heart Disease (1)'],
        random_state: int = RANDOM_SEED
    ):
        self.feature_names = feature_names
        self.class_names = class_names
        self.random_state = random_state

        # Initialize LIME Tabular Explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.asarray(training_data, dtype=np.float32),
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification',
            discretize_continuous=False,  # Continuous features are z-score standardized
            random_state=self.random_state
        )

    def explain_instance(
        self,
        sample: np.ndarray,
        predict_fn,
        num_features: int = 10,
        num_samples: int = LIME_NUM_SAMPLES,
        target_class: int = 1
    ) -> Dict[str, Any]:
        """
        Generates local explanation for a single patient record.
        sample: (25,) 1D array or (1, 25) array.
        predict_fn: Callable returning (N, 2) probability array.
        """
        sample_1d = np.asarray(sample, dtype=np.float32).flatten()
        if len(sample_1d) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(sample_1d)}")

        # Obtain actual model prediction
        probs = predict_fn(sample_1d.reshape(1, -1))[0]
        prob_disease = float(probs[1])
        prob_healthy = float(probs[0])
        pred_label = int(prob_disease >= 0.5)

        # Generate LIME explanation
        exp = self.explainer.explain_instance(
            data_row=sample_1d,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(target_class,)
        )

        # Parse feature importance using exact integer feature indices from exp.as_map()
        # This completely eliminates fragile substring matching (e.g. cp_1 vs cp_10)
        feature_weights = {}
        explanation_map = exp.as_map().get(target_class, [])
        for feat_idx, weight in explanation_map:
            if 0 <= feat_idx < len(self.feature_names):
                fname = self.feature_names[feat_idx]
                feature_weights[fname] = float(weight)

        sorted_features = sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive = [f for f, w in sorted_features if w > 0]
        top_negative = [f for f, w in sorted_features if w < 0]

        return {
            'predicted_label': pred_label,
            'prob_healthy': prob_healthy,
            'prob_disease': prob_disease,
            'feature_weights': feature_weights,
            'sorted_features': sorted_features,
            'top_positive': top_positive,
            'top_negative': top_negative,
            'lime_raw_explanation': exp
        }

    def plot_explanation(
        self,
        explanation_result: Dict[str, Any],
        client_name: str,
        model_name: str,
        sample_id: str,
        actual_label: Optional[int] = None,
        save_path: Optional[Path] = None
    ) -> Path:
        """
        Renders a clean horizontal bar chart visualization of local LIME weights
        with schema-faithful, human-understandable feature labels.
        """
        LIME_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        if save_path is None:
            save_path = LIME_FIGURES_DIR / f"{client_name.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}_{sample_id}_lime.png"

        sorted_feats = explanation_result['sorted_features'][:10]
        feats = [x[0] for x in sorted_feats][::-1]
        weights = [x[1] for x in sorted_feats][::-1]
        display_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in feats]
        colors = ['#d62728' if w > 0 else '#1f77b4' for w in weights]  # Red = Risk increasing, Blue = Protective

        plt.figure(figsize=(9, 5.5))
        bars = plt.barh(display_labels, weights, color=colors, alpha=0.85, edgecolor='black')
        plt.axvline(0, color='black', linewidth=1, linestyle='--')

        pred_lbl = "Heart Disease (1)" if explanation_result['predicted_label'] == 1 else "Healthy (0)"
        act_lbl_str = f" | True: {'Disease (1)' if actual_label == 1 else 'Healthy (0)'}" if actual_label is not None else ""

        plt.title(
            f"LIME Feature Attribution: {model_name} on {client_name} ({sample_id})\n"
            f"Predicted: {pred_lbl} (P={explanation_result['prob_disease']:.1%}){act_lbl_str}",
            fontsize=11,
            fontweight='bold'
        )
        plt.xlabel("LIME Feature Weight (Contribution to Model Prediction)", fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5, axis='x')

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', edgecolor='black', label='Increases Model Risk Score (+)'),
            Patch(facecolor='#1f77b4', edgecolor='black', label='Decreases Model Risk Score (-)')
        ]
        plt.legend(handles=legend_elements, loc='lower right', frameon=True)

        plt.figtext(
            0.5, -0.02,
            "Note: Attributions quantify local model sensitivity in standardized feature space; they do not establish clinical causality.",
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
    test_y = loaders['test'].dataset.targets.numpy()

    wrapper = load_federated_model('Federated_1D_AlexNet')
    explainer = LimeTabularExplainerWrapper(train_x)
    res = explainer.explain_instance(test_x[0], wrapper.predict_proba, num_features=10)
    print("LIME Single Case Explanation Result:")
    print(f"  Prediction: {res['predicted_label']} (P={res['prob_disease']:.4f})")
    print(f"  Top Features: {res['sorted_features'][:5]}")
    plot_path = explainer.plot_explanation(res, 'Hospital 1 (Cleveland)', 'Federated 1D AlexNet', 'sample_01', int(test_y[0]))
    print(f"  Plot saved to: {plot_path}")
