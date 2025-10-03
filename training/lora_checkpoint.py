"""
LoRA Checkpoint Management Utilities

Handles saving and loading of LoRA-only checkpoints for efficient training.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


def save_lora_checkpoint(
    checkpoint_path: str,
    unet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    config: Dict[str, Any],
    ema_model: Optional[Any] = None,
    get_lora_state_dict_fn: Optional[callable] = None,
) -> str:
    """
    Save a LoRA-only checkpoint.

    Args:
        checkpoint_path: Path to save checkpoint
        unet: The UNet model with LoRA applied
        optimizer: Optimizer with state to save
        global_step: Current training step
        epoch: Current epoch
        config: Training configuration
        ema_model: Optional EMA model
        get_lora_state_dict_fn: Function to extract LoRA weights

    Returns:
        Path to saved checkpoint
    """
    # Import here to avoid circular dependency
    if get_lora_state_dict_fn is None:
        from sdxl.lora import get_lora_state_dict
        get_lora_state_dict_fn = get_lora_state_dict

    # Get LoRA weights
    lora_state = get_lora_state_dict_fn(unet)

    # Build checkpoint
    checkpoint = {
        "lora_state_dict": lora_state,
        "optimizer": optimizer.state_dict(),
        "step": global_step,
        "epoch": epoch,
        "config": config,
        "checkpoint_version": "1.0",  # Version for future compatibility
        "is_lora_checkpoint": True,  # Explicit flag
    }

    # Add EMA LoRA weights if present
    if ema_model is not None:
        ema_lora_state = get_lora_state_dict_fn(ema_model.model)
        checkpoint["ema_lora"] = ema_lora_state

    # Save checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    # Report size
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    print(f"Saved LoRA checkpoint to {checkpoint_path} ({file_size:.2f} MB)")

    return checkpoint_path


def load_lora_checkpoint(
    checkpoint_path: str,
    unet: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[Any] = None,
    device: str = "cuda",
    strict: bool = True,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Load a LoRA checkpoint and resume training.

    Args:
        checkpoint_path: Path to checkpoint file
        unet: UNet model with LoRA already applied
        optimizer: Optional optimizer to restore state
        ema_model: Optional EMA model to restore
        device: Device to load checkpoint to
        strict: Whether to enforce strict state dict loading

    Returns:
        Tuple of (global_step, epoch, config)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if this is a LoRA checkpoint
    if not checkpoint.get("is_lora_checkpoint", False) and "lora_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not appear to be a LoRA checkpoint. "
            "Use the standard load_checkpoint function for full model checkpoints."
        )

    # Verify that the model has LoRA layers
    model_state = unet.state_dict()
    has_lora = any('lora' in key.lower() for key in model_state.keys())
    if not has_lora:
        raise ValueError(
            "Model does not have LoRA layers applied. "
            "Make sure to apply LoRA to the model before loading a LoRA checkpoint."
        )

    # Load LoRA weights
    lora_state = checkpoint["lora_state_dict"]

    # Create a mapping of LoRA weights to model state
    lora_keys_in_model = {k: v for k, v in model_state.items() if 'lora' in k.lower()}
    lora_keys_in_checkpoint = set(lora_state.keys())

    # Check for missing/unexpected keys
    missing_keys = set(lora_keys_in_model.keys()) - lora_keys_in_checkpoint
    unexpected_keys = lora_keys_in_checkpoint - set(lora_keys_in_model.keys())

    if missing_keys:
        msg = f"Missing LoRA keys in checkpoint: {missing_keys}"
        if strict:
            raise KeyError(msg)
        else:
            print(f"Warning: {msg}")

    if unexpected_keys:
        msg = f"Unexpected LoRA keys in checkpoint: {unexpected_keys}"
        if strict:
            raise KeyError(msg)
        else:
            print(f"Warning: {msg}")

    # Update model state with LoRA weights
    for key, value in lora_state.items():
        if key in model_state:
            model_state[key] = value.to(device)

    unet.load_state_dict(model_state, strict=False)  # False because we're only updating LoRA weights

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            # Move optimizer state to correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            print("Restored optimizer state")
        except Exception as e:
            print(f"Warning: Failed to restore optimizer state: {e}")

    # Load EMA LoRA weights if present
    if ema_model is not None and "ema_lora" in checkpoint:
        try:
            ema_lora_state = checkpoint["ema_lora"]
            ema_model_state = ema_model.model.state_dict()
            for key, value in ema_lora_state.items():
                if key in ema_model_state:
                    ema_model_state[key] = value.to(device)
            ema_model.model.load_state_dict(ema_model_state, strict=False)
            print("Restored EMA LoRA weights")
        except Exception as e:
            print(f"Warning: Failed to restore EMA state: {e}")

    # Extract metadata
    global_step = checkpoint.get("step", 0)
    epoch = checkpoint.get("epoch", 0)
    config = checkpoint.get("config", {})

    # Report loaded state
    num_lora_params = len([k for k in model_state.keys() if 'lora' in k.lower()])
    print(f"Successfully loaded LoRA checkpoint:")
    print(f"  LoRA parameters: {num_lora_params}")
    print(f"  Global step: {global_step}")
    print(f"  Epoch: {epoch}")
    print(f"  Checkpoint version: {checkpoint.get('checkpoint_version', 'unknown')}")

    return global_step, epoch, config


def is_lora_checkpoint(checkpoint_path: str) -> bool:
    """
    Check if a checkpoint file is a LoRA-only checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        True if it's a LoRA checkpoint, False otherwise
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        return checkpoint.get("is_lora_checkpoint", False) or "lora_state_dict" in checkpoint
    except Exception:
        return False


def get_latest_lora_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest LoRA checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    # Find all LoRA checkpoint files
    lora_files = []
    for file in checkpoint_dir.glob("lora_*.pt"):
        if is_lora_checkpoint(str(file)):
            lora_files.append(file)

    if not lora_files:
        return None

    # Sort by modification time and return latest
    latest = max(lora_files, key=lambda f: f.stat().st_mtime)
    return str(latest)


def create_resume_command(checkpoint_path: str, original_args: Dict[str, Any]) -> str:
    """
    Generate a command to resume training from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        original_args: Original training arguments from checkpoint

    Returns:
        Command string to resume training
    """
    # Build command with essential arguments
    cmd_parts = ["uv run python train_sdxl.py"]

    # Add resume path
    cmd_parts.append(f"--resume_from '{checkpoint_path}'")

    # Add essential arguments from original training
    essential_args = [
        "data_dir", "use_lora", "lora_rank", "lora_alpha", "lora_target_mode",
        "batch_size", "gradient_accumulation_steps", "num_epochs",
        "learning_rate", "min_snr_gamma", "warmup_steps",
        "precision", "image_size", "center_crop", "random_flip",
        "wandb_project", "device", "num_workers", "seed"
    ]

    for arg in essential_args:
        if arg in original_args:
            value = original_args[arg]
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{arg}")
            elif value is not None:
                if isinstance(value, str):
                    cmd_parts.append(f"--{arg} '{value}'")
                else:
                    cmd_parts.append(f"--{arg} {value}")

    return " \\\n    ".join(cmd_parts)


def inspect_checkpoint(checkpoint_path: str) -> None:
    """
    Print detailed information about a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print(f"\nCheckpoint Information: {checkpoint_path}")
    print("=" * 60)

    # Basic info
    is_lora = checkpoint.get("is_lora_checkpoint", False) or "lora_state_dict" in checkpoint
    print(f"Type: {'LoRA checkpoint' if is_lora else 'Full checkpoint'}")
    print(f"File size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")

    # Training state
    print(f"\nTraining State:")
    print(f"  Step: {checkpoint.get('step', 'unknown')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Contents
    print(f"\nContents:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            if key == "config":
                print(f"  {key}: {len(checkpoint[key])} config entries")
            elif key == "optimizer":
                print(f"  {key}: optimizer state dict")
            elif "lora" in key.lower():
                num_params = len(checkpoint[key])
                total_params = sum(v.numel() for v in checkpoint[key].values())
                print(f"  {key}: {num_params} tensors, {total_params:,} parameters")
            else:
                print(f"  {key}: dict with {len(checkpoint[key])} entries")
        else:
            print(f"  {key}: {type(checkpoint[key]).__name__}")

    # Config preview
    if "config" in checkpoint:
        config = checkpoint["config"]
        print(f"\nKey Configuration:")
        important_keys = ["use_lora", "lora_rank", "learning_rate", "batch_size",
                         "image_size", "precision", "data_dir"]
        for key in important_keys:
            if key in config:
                print(f"  {key}: {config[key]}")

    # Resume command
    if "config" in checkpoint:
        print(f"\nTo resume training:")
        print("-" * 40)
        resume_cmd = create_resume_command(checkpoint_path, checkpoint["config"])
        print(resume_cmd)
        print("-" * 40)