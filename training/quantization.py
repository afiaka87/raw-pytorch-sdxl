"""
Quantization utilities for memory-efficient training.

Supports INT8 and 4-bit quantization using PyTorch native quantization and bitsandbytes.
"""

import torch
import torch.nn as nn
from typing import Optional


def quantize_model(model: nn.Module, quantization_type: str, device: Optional[torch.device] = None):
    """
    Quantize model weights to reduce memory usage.

    Args:
        model: PyTorch model to quantize
        quantization_type: "int8" or "4bit"
        device: Target device (if None, uses current device)

    Returns:
        Quantized model

    Note:
        - INT8 uses PyTorch native dynamic quantization
        - 4-bit uses bitsandbytes if available, falls back to INT8
        - Only quantizes Linear layers in the base model (not LoRA)
        - LoRA weights remain in FP16/BF16 for training
    """
    if quantization_type not in ["int8", "4bit"]:
        raise ValueError(f"Unknown quantization type: {quantization_type}")

    print(f"Quantizing model to {quantization_type.upper()}...")

    if quantization_type == "int8":
        # Use PyTorch native INT8 dynamic quantization
        _apply_int8_quantization(model)
    elif quantization_type == "4bit":
        # Try bitsandbytes 4-bit, fall back to INT8 if not available
        try:
            import bitsandbytes as bnb
            _apply_bnb_4bit_quantization(model, device)
        except ImportError:
            print("Warning: bitsandbytes not available, falling back to INT8 quantization")
            _apply_int8_quantization(model)

    return model


def _apply_int8_quantization(model: nn.Module):
    """Apply PyTorch native INT8 dynamic quantization to Linear layers."""
    # Collect all Linear layers to quantize (excluding LoRA)
    layers_to_quantize = []

    def find_linear_layers(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and 'lora' not in full_name.lower():
                layers_to_quantize.append((module, name, child))
            else:
                find_linear_layers(child, full_name)

    find_linear_layers(model)

    # Quantize each layer
    for parent_module, layer_name, layer in layers_to_quantize:
        # Create quantized version
        quantized = torch.quantization.quantize_dynamic(
            layer,
            {nn.Linear},
            dtype=torch.qint8
        )
        setattr(parent_module, layer_name, quantized)

    print(f"  Quantized {len(layers_to_quantize)} Linear layers to INT8")


def _apply_bnb_4bit_quantization(model: nn.Module, device: Optional[torch.device] = None):
    """Apply bitsandbytes 4-bit quantization to Linear layers."""
    import bitsandbytes as bnb

    # Collect all Linear layers to quantize (excluding LoRA)
    layers_to_quantize = []

    def find_linear_layers(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and 'lora' not in full_name.lower():
                layers_to_quantize.append((module, name, child))
            else:
                find_linear_layers(child, full_name)

    find_linear_layers(model)

    # Replace each layer with 4-bit version
    for parent_module, layer_name, layer in layers_to_quantize:
        # Create 4-bit layer
        fourbit_layer = bnb.nn.Linear4bit(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_type='nf4',
        )

        # Copy weights
        if device is not None:
            fourbit_layer = fourbit_layer.to(device)

        with torch.no_grad():
            fourbit_layer.weight.data.copy_(layer.weight.data)
            if layer.bias is not None:
                fourbit_layer.bias.data.copy_(layer.bias.data)

        setattr(parent_module, layer_name, fourbit_layer)

    print(f"  Quantized {len(layers_to_quantize)} Linear layers to 4-bit (NF4)")


def get_model_memory_footprint(model: nn.Module) -> float:
    """
    Calculate model memory footprint in MB.

    Args:
        model: PyTorch model

    Returns:
        Memory usage in MB
    """
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem_total = mem_params + mem_buffers
    return mem_total / 1024**2  # Convert to MB
