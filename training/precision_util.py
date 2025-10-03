"""
Mixed Precision Training Utilities

Focuses on BF16 (bfloat16) for stable mixed-precision training.
BF16 is preferred over FP16 for diffusion models due to better numerical stability.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Optional


class MixedPrecisionManager:
    """
    Manager for mixed precision training with BF16.

    BF16 advantages over FP16:
    - Same exponent range as FP32 (no overflow/underflow issues)
    - No loss scaling required
    - More stable for diffusion models
    - Supported on Ampere+ GPUs (RTX 30xx, A100, etc.)
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """
        Args:
            dtype: Precision dtype (torch.bfloat16 or torch.float16)
            device: Device to use
        """
        self.dtype = dtype
        self.device = device
        self.enabled = dtype != torch.float32

        # Check if BF16 is supported
        if dtype == torch.bfloat16 and device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available for BF16 training")
            if not torch.cuda.is_bf16_supported():
                print(
                    "WARNING: BF16 not supported on this GPU. "
                    "Falling back to FP32. Consider using FP16 or upgrading GPU."
                )
                self.dtype = torch.float32
                self.enabled = False

    @contextmanager
    def autocast(self):
        """
        Context manager for automatic mixed precision.

        Usage:
            with precision_manager.autocast():
                output = model(input)
                loss = criterion(output, target)
        """
        if self.enabled:
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                yield
        else:
            yield

    def convert_model(self, model: nn.Module) -> nn.Module:
        """
        Convert model to mixed precision.

        Args:
            model: PyTorch model

        Returns:
            Model with mixed precision
        """
        if self.enabled:
            model = model.to(dtype=self.dtype)
        return model

    def prepare_optimizer(self, optimizer):
        """
        Prepare optimizer for mixed precision (no-op for BF16).

        For FP16, this would set up GradScaler, but BF16 doesn't need it.
        """
        return optimizer

    def __repr__(self):
        return f"MixedPrecisionManager(dtype={self.dtype}, enabled={self.enabled})"


def convert_module_to_bf16(module: nn.Module) -> nn.Module:
    """
    Convert module to BF16.

    This is useful for converting specific submodules while keeping
    others in FP32 (e.g., VAE in FP32, UNet in BF16).
    """
    return module.to(dtype=torch.bfloat16)


def convert_module_to_fp32(module: nn.Module) -> nn.Module:
    """Convert module to FP32."""
    return module.to(dtype=torch.float32)


class SelectivePrecision:
    """
    Selective precision for different model components.

    Example:
        # VAE in FP32 for quality, UNet in BF16 for speed
        precision = SelectivePrecision(
            vae_dtype=torch.float32,
            unet_dtype=torch.bfloat16,
            text_encoder_dtype=torch.float16,
        )
    """

    def __init__(
        self,
        vae_dtype: torch.dtype = torch.float32,
        unet_dtype: torch.dtype = torch.bfloat16,
        text_encoder_dtype: torch.dtype = torch.float16,
    ):
        self.vae_dtype = vae_dtype
        self.unet_dtype = unet_dtype
        self.text_encoder_dtype = text_encoder_dtype

    def apply_to_vae(self, vae: nn.Module) -> nn.Module:
        """Apply precision to VAE."""
        return vae.to(dtype=self.vae_dtype)

    def apply_to_unet(self, unet: nn.Module) -> nn.Module:
        """Apply precision to UNet."""
        return unet.to(dtype=self.unet_dtype)

    def apply_to_text_encoder(self, text_encoder: nn.Module) -> nn.Module:
        """Apply precision to text encoder."""
        return text_encoder.to(dtype=self.text_encoder_dtype)


def get_dtype_from_str(dtype_str: str) -> torch.dtype:
    """
    Convert string to torch dtype.

    Args:
        dtype_str: "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"

    Returns:
        torch.dtype
    """
    dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    dtype_str = dtype_str.lower()
    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unknown dtype: {dtype_str}. "
            f"Supported: {list(dtype_map.keys())}"
        )
    return dtype_map[dtype_str]


# Memory management utilities
def print_memory_stats(device: str = "cuda:0"):
    """Print CUDA memory statistics."""
    if torch.cuda.is_available():
        print(f"\n=== GPU Memory Stats ({device}) ===")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"Reserved:  {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
        print("=" * 40 + "\n")


def clear_memory():
    """Clear CUDA cache and collect garbage."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# Gradient checkpointing utilities
def enable_gradient_checkpointing(model: nn.Module):
    """
    Enable gradient checkpointing on a model.

    This trades compute for memory by not storing intermediate activations.
    Reduces memory usage by ~30-40% at the cost of ~20% slower training.
    """
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    else:
        print(
            "WARNING: Model does not support gradient_checkpointing. "
            "Implement enable_gradient_checkpointing() method."
        )


def disable_gradient_checkpointing(model: nn.Module):
    """Disable gradient checkpointing."""
    if hasattr(model, "disable_gradient_checkpointing"):
        model.disable_gradient_checkpointing()
