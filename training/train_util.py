"""
Training utilities for SDXL fine-tuning.

Helper functions for training, validation, and checkpointing.
"""

import torch
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_model: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Save training checkpoint.

    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        ema_model: EMA model (optional)
        step: Global step
        epoch: Current epoch
        config: Training configuration
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
    }

    if ema_model is not None:
        checkpoint["ema"] = ema_model.state_dict()

    if config is not None:
        checkpoint["config"] = config

    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save checkpoint
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[Any] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        ema_model: EMA model to load (optional)
        device: Device to load to

    Returns:
        Dict with 'step', 'epoch', 'config'
    """
    checkpoint = torch.load(path, map_location=device)

    # Load model
    model.load_state_dict(checkpoint["model"])

    # Load optimizer if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load EMA if provided
    if ema_model is not None and "ema" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema"])

    print(f"Loaded checkpoint from {path}")
    print(f"  Step: {checkpoint.get('step', 0)}")
    print(f"  Epoch: {checkpoint.get('epoch', 0)}")

    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "config": checkpoint.get("config", {}),
    }


def latents_to_pil(latents: torch.Tensor, vae) -> Image.Image:
    """
    Convert latents to PIL Image.

    Args:
        latents: [B, 4, H, W] latents (scaled)
        vae: VAE model for decoding

    Returns:
        PIL Image
    """
    # Decode
    with torch.no_grad():
        images = vae.decode(latents)

    # Convert to PIL
    # images are in range [-1, 1]
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype(np.uint8)

    # Return first image
    pil_image = Image.fromarray(images[0])
    return pil_image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor to PIL Image.

    Args:
        tensor: [B, C, H, W] or [C, H, W] in range [-1, 1]

    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first in batch

    # Convert from [-1, 1] to [0, 255]
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    tensor = tensor.cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).round().astype(np.uint8)

    return Image.fromarray(tensor)


def make_grid(images: list[Image.Image], cols: int = 4) -> Image.Image:
    """
    Create a grid of images.

    Args:
        images: List of PIL Images
        cols: Number of columns

    Returns:
        Grid image
    """
    if not images:
        raise ValueError("No images provided")

    n = len(images)
    rows = (n + cols - 1) // cols

    # Get image size (assume all same size)
    w, h = images[0].size

    # Create grid
    grid = Image.new("RGB", (w * cols, h * rows), color=(255, 255, 255))

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * w, row * h))

    return grid


def setup_wandb(
    project_name: str,
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Setup Weights & Biases logging.

    Args:
        project_name: W&B project name
        run_name: Optional run name
        config: Configuration dict to log

    Returns:
        W&B run object
    """
    try:
        import wandb

        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
        )
        return run
    except ImportError:
        print("wandb not installed. Skipping W&B logging.")
        return None


def log_to_wandb(metrics: Dict[str, Any], step: int, wandb_run=None):
    """Log metrics to W&B."""
    if wandb_run is not None:
        try:
            import wandb

            wandb.log(metrics, step=step)
        except ImportError:
            pass


def create_output_directory(base_dir: str = "outputs") -> Path:
    """
    Create auto-incrementing output directory.

    Args:
        base_dir: Base directory for outputs

    Returns:
        Path to created directory
    """
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    # Find next available number
    existing_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()
    ]
    if existing_dirs:
        next_num = max(int(d.name) for d in existing_dirs) + 1
    else:
        next_num = 0

    # Create directory with 4-digit padding
    output_dir = base_path / f"{next_num:04d}"
    output_dir.mkdir(exist_ok=True)

    print(f"Created output directory: {output_dir}")
    return output_dir


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to JSON file."""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {path}")


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(path, "r") as f:
        config = json.load(f)
    return config


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Count model parameters.

    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """Set learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
