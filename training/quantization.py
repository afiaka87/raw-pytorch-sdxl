"""
Quantization utilities for memory-efficient training.

Supports INT8 and 4-bit quantization using bitsandbytes.
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
        - Requires bitsandbytes library
        - Only quantizes Linear layers in the base model (not LoRA)
        - LoRA weights remain in FP16/BF16 for training
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for quantization. "
            "Install with: pip install bitsandbytes"
        )

    if quantization_type not in ["int8", "4bit"]:
        raise ValueError(f"Unknown quantization type: {quantization_type}")

    print(f"Quantizing model to {quantization_type.upper()}...")

    if quantization_type == "int8":
        # Replace Linear layers with 8-bit versions
        _replace_linear_with_int8(model)
    elif quantization_type == "4bit":
        # Replace Linear layers with 4-bit versions
        _replace_linear_with_4bit(model)

    return model


def _replace_linear_with_int8(model: nn.Module):
    """Replace nn.Linear layers with Int8 Linear layers."""
    import bitsandbytes as bnb

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and 'lora' not in name.lower():
            # Skip LoRA layers - they need full precision for training
            # Replace with 8-bit linear
            int8_layer = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False,
            )

            # Copy weight and bias to new layer
            int8_layer.weight = bnb.nn.Int8Params(
                module.weight.data,
                requires_grad=False,
                has_fp16_weights=False,
            )
            if module.bias is not None:
                int8_layer.bias = module.bias

            setattr(model, name, int8_layer)
        else:
            # Recursively replace in child modules
            _replace_linear_with_int8(module)


def _replace_linear_with_4bit(model: nn.Module):
    """Replace nn.Linear layers with 4-bit Linear layers."""
    import bitsandbytes as bnb

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and 'lora' not in name.lower():
            # Skip LoRA layers - they need full precision for training
            # Replace with 4-bit linear
            fourbit_layer = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type='nf4',
            )

            # The weights will be quantized automatically when moved to device
            with torch.no_grad():
                fourbit_layer.weight.data = module.weight.data
                if module.bias is not None:
                    fourbit_layer.bias = module.bias

            setattr(model, name, fourbit_layer)
        else:
            # Recursively replace in child modules
            _replace_linear_with_4bit(module)


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
