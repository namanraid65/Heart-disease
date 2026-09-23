"""
LIME vs. SHAP Explanation Comparison and Agreement Analysis Module
Computes Top-K feature overlap, Jaccard similarity, and sign consistency
between local LIME surrogate weights and SHAP Shapley values.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Any, Tuple
import numpy as np

from xai.config import TOP_K_FEATURES


def compute_explanation_agreement(
    lime_sorted_features: List[Tuple[str, float]],
    shap_sorted_features: List[Tuple[str, float]],
    top_k: int = TOP_K_FEATURES
) -> Dict[str, Any]:
    """
    Calculates quantitative agreement between LIME and SHAP rankings for a sample.

    Metrics:
      1. Top-K Overlap Ratio: |Top_K_LIME intersect Top_K_SHAP| / K
      2. Jaccard Similarity:  |Top_K_LIME intersect Top_K_SHAP| / |Top_K_LIME union Top_K_SHAP|
      3. Sign Consistency:    Proportion of common features that share the same attribution sign (+/-)
    """
    lime_top_k = [f for f, _ in lime_sorted_features[:top_k]]
    shap_top_k = [f for f, _ in shap_sorted_features[:top_k]]

    set_lime = set(lime_top_k)
    set_shap = set(shap_top_k)

    common_features = set_lime.intersection(set_shap)
    union_features = set_lime.union(set_shap)

    overlap_ratio = len(common_features) / top_k if top_k > 0 else 0.0
    jaccard_sim = len(common_features) / len(union_features) if len(union_features) > 0 else 0.0

    # Calculate sign consistency on common features
    lime_weights_dict = dict(lime_sorted_features)
    shap_weights_dict = dict(shap_sorted_features)

    sign_matches = 0
    for feat in common_features:
        l_sign = np.sign(lime_weights_dict.get(feat, 0.0))
        s_sign = np.sign(shap_weights_dict.get(feat, 0.0))
        if l_sign == s_sign and l_sign != 0:
            sign_matches += 1

    sign_consistency = sign_matches / len(common_features) if len(common_features) > 0 else 1.0

    return {
        'top_k': top_k,
        'lime_top_k': lime_top_k,
        'shap_top_k': shap_top_k,
        'common_features': list(common_features),
        'overlap_count': len(common_features),
        'overlap_ratio': float(overlap_ratio),
        'jaccard_similarity': float(jaccard_sim),
        'sign_consistency': float(sign_consistency)
    }


def format_feature_list(features: List[str], max_items: int = 5) -> str:
    """Helper to format feature lists cleanly for markdown tables."""
    return ", ".join(features[:max_items])
