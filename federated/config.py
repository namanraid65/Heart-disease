"""
Configuration for Federated Learning Simulation with 1D AlexNet
Defines communication rounds, client settings, local hyperparameters, and paths.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

# Directory Paths - AlexNet
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
FEDERATED_CHECKPOINTS_DIR = PROJECT_ROOT / 'models' / 'checkpoints' / 'federated_alexnet'
FEDERATED_FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures' / 'federated_alexnet'

# Directory Paths - ResNet
FEDERATED_RESNET_CHECKPOINTS_DIR = PROJECT_ROOT / 'models' / 'checkpoints' / 'federated_resnet'
FEDERATED_RESNET_FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures' / 'federated_resnet'
REPORTS_DIR = PROJECT_ROOT / 'reports'

# Client Registry
FEDERATED_CLIENTS = {
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

# Total Federated Training Population
TOTAL_TRAIN_SAMPLES = sum(c['train_samples'] for c in FEDERATED_CLIENTS.values())  # 503

# Federated Learning Common Hyperparameters
NUM_ROUNDS: int = 15                 # Number of Federated Communication Rounds
NUM_CLIENTS: int = 3                 # Number of participating hospital nodes
FRACTION_FIT: float = 1.0            # 100% client participation per round
LOCAL_EPOCHS: int = 3                # Local gradient epochs per communication round
LOCAL_BATCH_SIZE: int = 16           # Local batch size
LOCAL_LEARNING_RATE: float = 0.001   # Local Adam optimizer learning rate
LOCAL_WEIGHT_DECAY: float = 1e-4     # L2 regularization
DROPOUT_RATE: float = 0.3            # Dropout rate
RANDOM_SEED: int = 42                # Reproducibility seed

# ResNet Specific Constants
RESNET_NUM_ROUNDS: int = 15
RESNET_LOCAL_EPOCHS: int = 3
RESNET_LOCAL_BATCH_SIZE: int = 16
RESNET_LOCAL_LEARNING_RATE: float = 0.001
RESNET_LOCAL_WEIGHT_DECAY: float = 1e-4
RESNET_DROPOUT_RATE: float = 0.3

# Device
DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Dimension Constants
INPUT_FEATURES: int = 25             # Preprocessed feature dimension (from schema)

# Strategy
STRATEGY_NAME: str = 'FedAvg'        # Weighted Federated Averaging


