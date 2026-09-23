"""
Configuration for Heterogeneous-Feature Federated Learning
Defines latent dimension, hyperparameter defaults, client registries, and output directories.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Any
import torch

# Paths
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
HETEROGENEOUS_CHECKPOINTS_DIR = PROJECT_ROOT / 'models' / 'checkpoints' / 'federated_heterogeneous'
HETEROGENEOUS_FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures' / 'federated_heterogeneous'
REPORTS_DIR = PROJECT_ROOT / 'reports'

# Participating Hospital Clients
HETEROGENEOUS_CLIENTS: Dict[str, Dict[str, Any]] = {
    'hospital_1': {
        'name': 'Hospital 1 (Cleveland)',
        'train_samples': 212,
        'val_samples': 45,
        'test_samples': 46
    },
    'hospital_2': {
        'name': 'Hospital 2 (Hungarian)',
        'train_samples': 205,
        'val_samples': 44,
        'test_samples': 44
    },
    'hospital_3': {
        'name': 'Hospital 3 (Switzerland)',
        'train_samples': 86,
        'val_samples': 18,
        'test_samples': 19
    }
}

TOTAL_TRAIN_SAMPLES = sum(c['train_samples'] for c in HETEROGENEOUS_CLIENTS.values())  # 503

# Common Latent Space
LATENT_DIM: int = 32

# Model Hidden Layer Architectures
ENCODER_HIDDEN_DIMS: List[int] = [64]
PREDICTOR_HIDDEN_DIMS: List[int] = [32]
DROPOUT_RATE: float = 0.2

# Federated Learning Hyperparameters
NUM_ROUNDS: int = 15
LOCAL_EPOCHS: int = 3
LOCAL_BATCH_SIZE: int = 16
LOCAL_LEARNING_RATE: float = 0.001
LOCAL_WEIGHT_DECAY: float = 1e-4
RANDOM_SEED: int = 42

# Device Configuration
DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# Strategy
STRATEGY_NAME: str = 'HeterogeneousFedAvg'
