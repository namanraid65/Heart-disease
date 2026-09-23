"""
Configuration and Hyperparameter Management for Local Models
Stores architecture, training, optimization, and path settings for 1D AlexNet, 1D ResNet, and XGBoost.
"""

import sys
from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

# Data Paths
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
CHECKPOINTS_DIR = PROJECT_ROOT / 'models' / 'checkpoints'
ALEXNET_FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures' / 'alexnet'
RESNET_FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures' / 'resnet'
XGBOOST_FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures' / 'xgboost'
FIGURES_DIR = ALEXNET_FIGURES_DIR  # Default alias
REPORTS_DIR = PROJECT_ROOT / 'reports'

# Client Identifiers & Checkpoint Mappings
CLIENT_CONFIGS = {
    'hospital_1': {
        'name': 'Hospital 1 (Cleveland)',
        'alexnet_checkpoint': 'hospital_1_alexnet.pt',
        'resnet_checkpoint': 'hospital_1_resnet.pt',
        'xgboost_checkpoint': 'hospital_1_xgboost.json'
    },
    'hospital_2': {
        'name': 'Hospital 2 (Hungarian)',
        'alexnet_checkpoint': 'hospital_2_alexnet.pt',
        'resnet_checkpoint': 'hospital_2_resnet.pt',
        'xgboost_checkpoint': 'hospital_2_xgboost.json'
    },
    'hospital_3': {
        'name': 'Hospital 3 (Switzerland)',
        'alexnet_checkpoint': 'hospital_3_alexnet.pt',
        'resnet_checkpoint': 'hospital_3_resnet.pt',
        'xgboost_checkpoint': 'hospital_3_xgboost.json'
    }
}

# Reproducibility Seed
RANDOM_SEED: int = 42

# Device Configuration
DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# Common Model Hyperparameters
INPUT_FEATURES: int = 25  # Dynamically verified from schema
DROPOUT_RATE: float = 0.3

# Deep Learning Training Hyperparameters
LEARNING_RATE: float = 0.001
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 16
MAX_EPOCHS: int = 150
EARLY_STOPPING_PATIENCE: int = 25
MIN_DELTA: float = 1e-4
LR_SCHEDULER_PATIENCE: int = 10
LR_SCHEDULER_FACTOR: float = 0.5

# XGBoost Hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.01,
    'reg_lambda': 1.0,
    'random_state': RANDOM_SEED,
    'eval_metric': 'logloss',
    'early_stopping_rounds': 15
}
