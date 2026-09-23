"""
Heterogeneous Client Feature Schema Definitions
Defines explicit feature schemas, data types, and encoder specifications
per hospital client, supporting heterogeneous native feature spaces without
forcing artificial padding or assuming identical dimensionality across sites.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch

from preprocessing.feature_schema import PROCESSED_FEATURE_NAMES


@dataclass
class ClientFeatureSchema:
    """
    Defines the native feature schema and encoder configuration for a specific hospital node.
    """
    hospital_id: str
    hospital_name: str
    feature_names: List[str]
    feature_types: Dict[str, str] = field(default_factory=dict)
    encoder_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_dimension(self) -> int:
        """Derives input dimension dynamically from feature count."""
        return len(self.feature_names)

    def validate_tensor_shape(self, x: torch.Tensor) -> bool:
        """
        Validates that an incoming tensor matches the hospital's declared input dimension.
        Raises ValueError if dimensionality does not match.
        """
        if x.ndim < 2:
            raise ValueError(
                f"Client '{self.hospital_id}' schema validation failed: "
                f"expected at least 2D tensor [batch_size, features], got shape {list(x.shape)}."
            )
        feature_dim = x.shape[1]
        if feature_dim != self.input_dimension:
            raise ValueError(
                f"Client '{self.hospital_id}' schema validation failed: "
                f"tensor has {feature_dim} features, but schema declares {self.input_dimension} features."
            )
        return True

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validates that a DataFrame contains the expected number of features and column names.
        """
        if len(df.columns) != self.input_dimension:
            raise ValueError(
                f"Client '{self.hospital_id}' DataFrame validation failed: "
                f"DataFrame has {len(df.columns)} columns, expected {self.input_dimension}."
            )
        return True


# Default feature types mapping for the standard 25 processed features
_STANDARD_FEATURE_TYPES: Dict[str, str] = {}
for feat in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
    _STANDARD_FEATURE_TYPES[feat] = 'continuous'
for feat in ['sex', 'fbs', 'exang']:
    _STANDARD_FEATURE_TYPES[feat] = 'binary'
for feat in PROCESSED_FEATURE_NAMES:
    if feat not in _STANDARD_FEATURE_TYPES:
        _STANDARD_FEATURE_TYPES[feat] = 'categorical_onehot'


# Synthetic 30-feature schema definition for Hospital 4 (demonstrating future hospital support)
# Adds 5 additional native biomarkers: bmi, hba1c, crp, ldl, hdl
H4_ADDITIONAL_FEATURES: List[str] = ['bmi', 'hba1c', 'crp', 'ldl', 'hdl']
H4_30_FEATURE_NAMES: List[str] = list(PROCESSED_FEATURE_NAMES) + H4_ADDITIONAL_FEATURES
H4_FEATURE_TYPES: Dict[str, str] = dict(_STANDARD_FEATURE_TYPES)
for feat in H4_ADDITIONAL_FEATURES:
    H4_FEATURE_TYPES[feat] = 'continuous'


# Client Schema Registry
CLIENT_SCHEMAS: Dict[str, ClientFeatureSchema] = {
    'hospital_1': ClientFeatureSchema(
        hospital_id='hospital_1',
        hospital_name='Hospital 1 (Cleveland)',
        feature_names=list(PROCESSED_FEATURE_NAMES),
        feature_types=_STANDARD_FEATURE_TYPES,
        encoder_config={
            'hidden_dims': [64],
            'dropout_rate': 0.2,
            'activation': 'relu'
        }
    ),
    'hospital_2': ClientFeatureSchema(
        hospital_id='hospital_2',
        hospital_name='Hospital 2 (Hungarian)',
        feature_names=list(PROCESSED_FEATURE_NAMES),
        feature_types=_STANDARD_FEATURE_TYPES,
        encoder_config={
            'hidden_dims': [64],
            'dropout_rate': 0.2,
            'activation': 'relu'
        }
    ),
    'hospital_3': ClientFeatureSchema(
        hospital_id='hospital_3',
        hospital_name='Hospital 3 (Switzerland)',
        feature_names=list(PROCESSED_FEATURE_NAMES),
        feature_types=_STANDARD_FEATURE_TYPES,
        encoder_config={
            'hidden_dims': [64],
            'dropout_rate': 0.2,
            'activation': 'relu'
        }
    ),
    # Synthetic 4th Hospital schema (D4 = 30) for architectural verification
    'hospital_4_synthetic': ClientFeatureSchema(
        hospital_id='hospital_4_synthetic',
        hospital_name='Hospital 4 (Extended Biomarkers - Synthetic)',
        feature_names=H4_30_FEATURE_NAMES,
        feature_types=H4_FEATURE_TYPES,
        encoder_config={
            'hidden_dims': [64],
            'dropout_rate': 0.2,
            'activation': 'relu'
        }
    )
}


def get_client_schema(client_id: str) -> ClientFeatureSchema:
    """
    Retrieves the registered feature schema for a client.
    Raises KeyError if client_id is not registered.
    """
    if client_id not in CLIENT_SCHEMAS:
        raise KeyError(
            f"No schema registered for client '{client_id}'. "
            f"Available schemas: {list(CLIENT_SCHEMAS.keys())}"
        )
    return CLIENT_SCHEMAS[client_id]


def register_client_schema(schema: ClientFeatureSchema) -> None:
    """
    Registers or updates a client feature schema dynamically.
    """
    CLIENT_SCHEMAS[schema.hospital_id] = schema
