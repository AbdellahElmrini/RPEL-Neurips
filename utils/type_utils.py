"""
Type Utilities for ByzDL

This module provides utility functions for handling type conversions
between numpy arrays and torch tensors, preserving types, and extracting
device/dtype information.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, Any, Dict, Type

from utils import torch_tools


def convert_weights_to_type(
    weights: Union[list, torch.Tensor, np.ndarray],
    target_type: Any,
    reference_tensor: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert weights to the appropriate type (numpy or torch) based on target_type.
    
    Parameters
    ----------
    weights : list, torch.Tensor, or numpy.ndarray
        Weights to convert
    target_type : module
        Target type module (either np or torch_tools/torch)
    reference_tensor : torch.Tensor or numpy.ndarray, optional
        Reference tensor/array to extract device and dtype from.
        If None and target is torch, uses CPU and float32.
    
    Returns
    -------
    torch.Tensor or numpy.ndarray
        Converted weights in the target type
    """

    # Determine if target is numpy or torch
    is_torch_target = (target_type == torch_tools or target_type == torch)
    
    if isinstance(weights, list):
        if is_torch_target:
            if reference_tensor is not None and isinstance(reference_tensor, torch.Tensor):
                device = reference_tensor.device
            else:
                device = torch.device('cpu')
            return torch.tensor(weights, dtype=torch.float32, device=device)
        else:
            return np.array(weights, dtype=np.float32)
    
    elif isinstance(weights, np.ndarray):
        if is_torch_target:
            if reference_tensor is not None and isinstance(reference_tensor, torch.Tensor):
                device = reference_tensor.device
            else:
                device = torch.device('cpu')
            return torch.from_numpy(weights).float().to(device)
        else:
            # Already numpy array
            return weights
    
    elif isinstance(weights, torch.Tensor):
        if is_torch_target:
            # Ensure on correct device
            if reference_tensor is not None and isinstance(reference_tensor, torch.Tensor):
                device = reference_tensor.device
                return weights.to(device)
            else:
                return weights
        else:
            # Convert to numpy
            return weights.cpu().numpy()
    
    else:
        # Fallback: convert to appropriate type
        if is_torch_target:
            if reference_tensor is not None and isinstance(reference_tensor, torch.Tensor):
                device = reference_tensor.device
            else:
                device = torch.device('cpu')
            return torch.tensor(weights, dtype=torch.float32, device=device)
        else:
            return np.array(weights, dtype=np.float32)


def extract_device_and_dtype(
    tensor_or_array: Union[torch.Tensor, np.ndarray]
) -> Tuple[Optional[torch.device], Optional[torch.dtype]]:
    """
    Extract device and dtype from a tensor or array.
    
    Parameters
    ----------
    tensor_or_array : torch.Tensor or numpy.ndarray
        Tensor or array to extract information from
    
    Returns
    -------
    tuple (device, dtype)
        device: torch.device or None (if numpy array)
        dtype: torch.dtype or None (if numpy array)
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.device, tensor_or_array.dtype
    else:
        return None, None


def preserve_input_type(
    input_value: Union[torch.Tensor, np.ndarray],
    output_value: Union[torch.Tensor, np.ndarray],
    input_type: Optional[type] = None
) -> Union[torch.Tensor, np.ndarray]:
    """
    Preserve the input type in the output.
    
    If input was numpy, ensure output is numpy.
    If input was torch, ensure output is torch (on correct device).
    
    Parameters
    ----------
    input_value : torch.Tensor or numpy.ndarray
        Original input value (used to determine type)
    output_value : torch.Tensor or numpy.ndarray
        Output value that may need type conversion
    input_type : type, optional
        Explicit input type. If None, inferred from input_value.
    
    Returns
    -------
    torch.Tensor or numpy.ndarray
        Output value with same type as input
    """
    if input_type is None:
        is_numpy = isinstance(input_value, np.ndarray)
    else:
        is_numpy = (input_type == np.ndarray)
    
    if is_numpy:
        if isinstance(output_value, torch.Tensor):
            return output_value.cpu().numpy()
        return output_value
    else:
        # Input was torch, ensure output is on correct device
        if isinstance(output_value, torch.Tensor):
            if isinstance(input_value, torch.Tensor):
                reference_device = input_value.device
                return output_value.to(reference_device)
            return output_value
        else:
            # Output is numpy but input was torch - convert to torch
            if isinstance(input_value, torch.Tensor):
                reference_device = input_value.device
                return torch.from_numpy(output_value).float().to(reference_device)
            else:
                # Fallback: convert to torch on CPU
                return torch.from_numpy(output_value).float()


def ensure_scalar(value: Union[float, int, torch.Tensor, np.ndarray]) -> float:
    """
    Convert a tensor or array to a Python scalar.
    
    Parameters
    ----------
    value : float, int, torch.Tensor, or numpy.ndarray
        Value to convert to scalar
    
    Returns
    -------
    float
        Python scalar value
    """
    if isinstance(value, (float, int)):
        return float(value)
    elif isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else float(value)
    elif isinstance(value, np.ndarray):
        return float(value.item() if value.size == 1 else value)
    else:
        return float(value)


def convert_threshold_to_type(
    threshold: Union[float, int],
    tools: Any,
    reference_tensor: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Convert clipping threshold to appropriate type (numpy or torch).
    
    Handles conversion of float('inf') to np.inf and ensures device/dtype
    matching for torch tensors.
    
    Parameters
    ----------
    threshold : float or int
        Clipping threshold value (may be float('inf'))
    tools : module
        Target type module (either np or torch_tools/torch)
    reference_tensor : torch.Tensor or numpy.ndarray, optional
        Reference tensor/array to extract device and dtype from.
        If None and target is torch, uses CPU and float32.
    
    Returns
    -------
    float, numpy.ndarray, or torch.Tensor
        Threshold in the target type format
    """
    from byzfl.utils import torch_tools
    
    # Determine if target is numpy or torch
    is_torch_target = (tools == torch_tools or tools == torch)
    
    if is_torch_target:
        # Convert to torch tensor
        if threshold == float('inf'):
            # For infinity, we still create a tensor but use inf
            if reference_tensor is not None and isinstance(reference_tensor, torch.Tensor):
                device = reference_tensor.device
                dtype = reference_tensor.dtype
            else:
                device = torch.device('cpu')
                dtype = torch.float32
            return torch.tensor(float('inf'), device=device, dtype=dtype)
        else:
            # Regular float value
            if reference_tensor is not None and isinstance(reference_tensor, torch.Tensor):
                device = reference_tensor.device
                dtype = reference_tensor.dtype
            else:
                device = torch.device('cpu')
                dtype = torch.float32
            return torch.tensor(float(threshold), device=device, dtype=dtype)
    else:
        # Convert to numpy
        if threshold == float('inf'):
            return np.inf
        else:
            return np.float32(threshold)


def validate_gradients_type_consistency(
    gradients_dict: Dict[int, Union[torch.Tensor, np.ndarray]],
    context: str = "gradients"
) -> Tuple[bool, Optional[Type]]:
    """
    Validate that all gradients in a dictionary have the same type.
    
    Parameters
    ----------
    gradients_dict : Dict[int, Union[torch.Tensor, np.ndarray]]
        Dictionary mapping node_id to gradient
    context : str, optional
        Context string for error messages (e.g., "neighbor gradients", "honest gradients")
        Default "gradients"
    
    Returns
    -------
    tuple
        (is_numpy, first_gradient_type)
        is_numpy: bool - True if all gradients are numpy, False if all are torch
        first_gradient_type: type - Type of first gradient (for reference)
    
    Raises
    ------
    ValueError
        If gradients_dict is empty
    TypeError
        If gradients have mixed types
    """
    if not gradients_dict:
        raise ValueError(f"{context} dictionary cannot be empty")
    
    # Get first gradient to determine expected type
    first_node_id = next(iter(gradients_dict.keys()))
    first_grad = gradients_dict[first_node_id]
    first_gradient_type = type(first_grad)
    is_numpy = isinstance(first_grad, np.ndarray)
    
    # Validate all gradients have the same type
    for node_id, grad in gradients_dict.items():
        grad_is_numpy = isinstance(grad, np.ndarray)
        if grad_is_numpy != is_numpy:
            raise TypeError(
                f"All {context} must be the same type. "
                f"Found mixed types: node {first_node_id} has {first_gradient_type.__name__}, "
                f"node {node_id} has {type(grad).__name__}. "
                f"All must be either numpy.ndarray or torch.Tensor."
            )
    
    return is_numpy, first_gradient_type

