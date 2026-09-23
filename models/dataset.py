"""
PyTorch Dataset and DataLoader Modules for Federated Heart Disease Clients
Loads processed client partitions from data/processed/<client_id>/ and encapsulates
them into reusable PyTorch Dataset and DataLoader objects.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from models.config import PROCESSED_DATA_DIR, BATCH_SIZE


class HeartDiseaseDataset(Dataset):
    """
    PyTorch Dataset for Heart Disease tabular clinical features.
    """

    def __init__(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray
    ):
        if isinstance(X, pd.DataFrame):
            self.features = torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32)
        else:
            self.features = torch.tensor(X, dtype=torch.float32)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            self.labels = torch.tensor(y.to_numpy(dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
        else:
            self.labels = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        assert len(self.features) == len(self.labels), "Features and labels must have equal length."

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[1]


def load_client_raw_splits(client_id: str) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Loads DataFrames for a single client from disk without merging across hospitals.
    """
    client_dir = PROCESSED_DATA_DIR / client_id
    if not client_dir.exists():
        raise FileNotFoundError(f"Client data directory not found: {client_dir}")

    X_train = pd.read_csv(client_dir / 'train' / 'X_train.csv')
    y_train = pd.read_csv(client_dir / 'train' / 'y_train.csv').squeeze('columns')

    X_val = pd.read_csv(client_dir / 'validation' / 'X_val.csv')
    y_val = pd.read_csv(client_dir / 'validation' / 'y_val.csv').squeeze('columns')

    X_test = pd.read_csv(client_dir / 'test' / 'X_test.csv')
    y_test = pd.read_csv(client_dir / 'test' / 'y_test.csv').squeeze('columns')

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def get_client_datasets(client_id: str) -> Dict[str, HeartDiseaseDataset]:
    """
    Returns train, validation, and test PyTorch Datasets for a client.
    """
    splits = load_client_raw_splits(client_id)
    return {
        'train': HeartDiseaseDataset(*splits['train']),
        'val': HeartDiseaseDataset(*splits['val']),
        'test': HeartDiseaseDataset(*splits['test'])
    }


def get_client_dataloaders(
    client_id: str,
    batch_size: int = BATCH_SIZE,
    shuffle_train: bool = True
) -> Dict[str, DataLoader]:
    """
    Constructs train, validation, and test DataLoaders for a given hospital client.
    """
    datasets = get_client_datasets(client_id)

    train_loader = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False
    )
    val_loader = DataLoader(
        datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    test_loader = DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
        loaders = get_client_dataloaders(cid, batch_size=16)
        print(f"{cid} DataLoaders -> Train batches: {len(loaders['train'])}, Val batches: {len(loaders['val'])}, Test batches: {len(loaders['test'])}")
