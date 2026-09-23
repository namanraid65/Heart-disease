"""
Configuration and Metadata for Explainable AI (XAI) Module
Stores hyperparameter configurations, explainer settings, sample selection rules,
and environment reproducibility metadata for LIME and SHAP.
"""

import sys
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import sklearn
import lime
import shap
import xgboost

from preprocessing.feature_schema import PROCESSED_FEATURE_NAMES, NUM_PROCESSED_FEATURES

# Directory Paths
XAI_DIR = PROJECT_ROOT / 'xai'
REPORTS_DIR = PROJECT_ROOT / 'reports'
XAI_FIGURES_DIR = REPORTS_DIR / 'figures' / 'xai'
LIME_FIGURES_DIR = XAI_FIGURES_DIR / 'lime'
SHAP_FIGURES_DIR = XAI_FIGURES_DIR / 'shap'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
CHECKPOINTS_DIR = PROJECT_ROOT / 'models' / 'checkpoints'
ALEXNET_CKPT_PATH = CHECKPOINTS_DIR / 'federated_alexnet' / 'global_alexnet_final.pt'
RESNET_CKPT_PATH = CHECKPOINTS_DIR / 'federated_resnet' / 'global_resnet_final.pt'

# Model Registry
MODELS_TO_EXPLAIN = [
    'Federated_1D_AlexNet',
    'Federated_1D_ResNet',
    'Local_XGBoost_Ensemble'  # Backward-compatible with 'Federated_XGBoost'
]

# Clinical Schema-Faithful Feature Display Names for XAI Reports and Visualizations
FEATURE_DISPLAY_NAMES = {
    'age': 'Age (standardized)',
    'trestbps': 'Resting Blood Pressure (standardized)',
    'chol': 'Serum Cholesterol (standardized)',
    'thalach': 'Max Heart Rate (standardized)',
    'oldpeak': 'ST Depression (standardized)',
    'sex': 'Sex: Male (1) / Female (0)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl',
    'exang': 'Exercise Induced Angina (1=Yes, 0=No)',
    'cp_1': 'Chest Pain: Typical Angina (cp=1)',
    'cp_2': 'Chest Pain: Atypical Angina (cp=2)',
    'cp_3': 'Chest Pain: Non-Anginal (cp=3)',
    'cp_4': 'Chest Pain: Asymptomatic (cp=4)',
    'restecg_0': 'Resting ECG: Normal (restecg=0)',
    'restecg_1': 'Resting ECG: ST-T Abnormality (restecg=1)',
    'restecg_2': 'Resting ECG: LV Hypertrophy (restecg=2)',
    'slope_1': 'ST Slope: Upsloping (slope=1)',
    'slope_2': 'ST Slope: Flat (slope=2)',
    'slope_3': 'ST Slope: Downsloping (slope=3)',
    'ca_0': 'Fluoroscopy: 0 Major Vessels (ca=0)',
    'ca_1': 'Fluoroscopy: 1 Major Vessel (ca=1)',
    'ca_2': 'Fluoroscopy: 2 Major Vessels (ca=2)',
    'ca_3': 'Fluoroscopy: 3 Major Vessels (ca=3)',
    'thal_3': 'Thallium: Normal (thal=3)',
    'thal_6': 'Thallium: Fixed Defect (thal=6)',
    'thal_7': 'Thallium: Reversible Defect (thal=7)'
}

# Client Definitions
HOSPITAL_CLIENTS = {
    'hospital_1': 'Hospital 1 (Cleveland)',
    'hospital_2': 'Hospital 2 (Hungarian)',
    'hospital_3': 'Hospital 3 (Switzerland)'
}

# Reproducibility & Environment Metadata
RANDOM_SEED: int = 42
DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

import importlib.metadata

def _get_package_version(pkg_name: str) -> str:
    try:
        return importlib.metadata.version(pkg_name)
    except Exception:
        return "0.2.0.1"

ENVIRONMENT_METADATA = {
    'python_version': platform.python_version(),
    'os': platform.platform(),
    'torch_version': torch.__version__,
    'sklearn_version': sklearn.__version__,
    'lime_version': _get_package_version('lime'),
    'shap_version': shap.__version__,
    'xgboost_version': xgboost.__version__,
    'random_seed': RANDOM_SEED,
    'device': DEVICE
}

# Explanation Hyperparameters
TOP_K_FEATURES: int = 5                # Top-K features for explanation agreement analysis
LIME_NUM_SAMPLES: int = 1000           # Number of perturbations for LIME TabularExplainer
SHAP_BACKGROUND_SAMPLES: int = 50      # Background reference samples for Kernel/DeepExplainer
EXPLAIN_SAMPLE_CASES_PER_CLIENT: int = 4 # Representative samples per client (TP, TN, FP, FN)

# Medical Safety Statement
MEDICAL_SAFETY_STATEMENT: str = (
    "The explanations and predictions presented in this study represent statistical machine learning "
    "model behavior and do not constitute clinical diagnosis, confirmed medical condition, or medical causality. "
    "This system is designed exclusively for research and educational purposes and is not a substitute "
    "for professional clinical diagnosis and medical judgment."
)
