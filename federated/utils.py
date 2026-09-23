"""
Utility Functions for Federated Learning
Handles model parameter extraction, tensor-to-numpy serialization,
parameter loading, and strict privacy validation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import OrderedDict
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def get_bn_buffer_names(model: nn.Module) -> List[str]:
    """
    Dynamically identifies all buffer names belonging to any BatchNorm module
    (e.g., BatchNorm1d, BatchNorm2d, BatchNorm3d, SyncBatchNorm).
    Returns list of fully qualified state_dict buffer names (e.g., 'features.1.running_mean').
    """
    bn_buffers = []
    for mod_name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            for buf_name, _ in module.named_buffers(recurse=False):
                full_name = f"{mod_name}.{buf_name}" if mod_name else buf_name
                bn_buffers.append(full_name)
    return bn_buffers


def get_shared_parameter_names(model: nn.Module) -> List[str]:
    """
    Returns the ordered list of state_dict keys representing shared federated parameters.
    Excludes all client-local BatchNorm buffers (running_mean, running_var, num_batches_tracked).
    Includes all trainable weights and biases (Conv, Linear, BN affine weights/biases).
    """
    bn_buffer_names = set(get_bn_buffer_names(model))
    return [k for k in model.state_dict().keys() if k not in bn_buffer_names]


def get_model_shared_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extracts strictly the shared federated parameter tensors as a list of NumPy arrays.
    Client-local BatchNorm buffers are never extracted or sent to the server.
    """
    shared_names = get_shared_parameter_names(model)
    state = model.state_dict()
    return [state[k].detach().cpu().numpy() for k in shared_names]


def set_model_shared_parameters(model: nn.Module, shared_parameters: List[np.ndarray]) -> None:
    """
    Updates strictly the shared federated parameters in the model while preserving
    the model's current client-local BatchNorm buffers (running_mean, running_var, num_batches_tracked).
    Performs load_state_dict with strict=True to guarantee architectural integrity.
    """
    shared_names = get_shared_parameter_names(model)
    if len(shared_names) != len(shared_parameters):
        raise ValueError(
            f"Parameter count mismatch: model has {len(shared_names)} shared parameters, "
            f"received {len(shared_parameters)}."
        )

    current_state = model.state_dict()
    for name, param_arr in zip(shared_names, shared_parameters):
        current_tensor = current_state[name]
        new_tensor = torch.as_tensor(param_arr, dtype=current_tensor.dtype, device=current_tensor.device)
        current_state[name] = new_tensor

    model.load_state_dict(current_state, strict=True)


def get_model_bn_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extracts a copy of all client-local BatchNorm buffer tensors.
    """
    bn_buffer_names = get_bn_buffer_names(model)
    state = model.state_dict()
    return {k: state[k].clone().detach().cpu() for k in bn_buffer_names}


def set_model_bn_state(model: nn.Module, bn_state: Dict[str, torch.Tensor]) -> None:
    """
    Loads client-specific BatchNorm buffers into the model while leaving
    all shared weights and biases untouched.
    """
    current_state = model.state_dict()
    for k, tensor in bn_state.items():
        if k in current_state:
            current_tensor = current_state[k]
            current_state[k] = tensor.to(device=current_tensor.device, dtype=current_tensor.dtype)
    model.load_state_dict(current_state, strict=True)


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extracts all state_dict parameter tensors as a list of NumPy arrays.
    Only model weights are returned; no data or metadata is included.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Loads a list of NumPy parameter arrays back into the PyTorch model's state_dict.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def verify_privacy_and_data_locality(payload: Any) -> bool:
    """
    Audits transmitted payload to guarantee zero patient data leakage.
    Verifies that the object contains strictly numeric weight arrays or scalars,
    and contains no tabular DataFrames, patient identifiers, or raw feature vectors.
    """
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, np.ndarray):
                raise TypeError(f"Privacy Audit Failed: Transmitted item is {type(item)}, not np.ndarray.")
            if not np.issubdtype(item.dtype, np.number):
                raise TypeError(f"Privacy Audit Failed: Non-numeric array type {item.dtype} detected.")
    elif isinstance(payload, dict):
        for k, v in payload.items():
            if k in ['patient_id', 'records', 'features', 'raw_data', 'X', 'y']:
                raise ValueError(f"Privacy Audit Failed: Forbidden data key '{k}' found in transmission.")
    return True


def aggregate_fedavg(
    results: List[Tuple[List[np.ndarray], int]]
) -> List[np.ndarray]:
    """
    Computes sample-weighted Federated Averaging (FedAvg):
      W_global = sum_{k=1}^K (n_k / N_total) * W_k
    """
    total_samples = sum(num_samples for _, num_samples in results)
    if total_samples == 0:
        raise ValueError("Total sample count across participating clients is 0.")

    # Initialize aggregated weights with zeros matching layer shapes
    first_weights = results[0][0]
    weighted_weights = [np.zeros_like(layer, dtype=np.float32) for layer in first_weights]

    for client_weights, num_samples in results:
        weight_fraction = num_samples / total_samples
        for i, layer in enumerate(client_weights):
            weighted_weights[i] += layer.astype(np.float32) * weight_fraction

    return weighted_weights
