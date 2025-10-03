#!/usr/bin/env python3
"""
Extract LoRA Weights for Inference

Extract just the LoRA weights from a training checkpoint for use in inference.
This creates a minimal file containing only the LoRA parameters, without
optimizer state or training metadata.
"""

import argparse
import torch
from pathlib import Path


def extract_lora_weights(checkpoint_path: str, output_path: str = None) -> str:
    """
    Extract LoRA weights from a training checkpoint.

    Args:
        checkpoint_path: Path to the training checkpoint
        output_path: Optional output path (defaults to checkpoint_name_weights_only.pt)

    Returns:
        Path to the saved weights file
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Check if this is a LoRA checkpoint
    if "lora_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain LoRA weights (no 'lora_state_dict' key)")

    # Extract LoRA weights
    lora_weights = checkpoint["lora_state_dict"]

    # Also extract EMA weights if available and requested
    has_ema = "ema_lora" in checkpoint

    # Create output filename if not provided
    if output_path is None:
        checkpoint_name = Path(checkpoint_path).stem
        output_path = Path(checkpoint_path).parent / f"{checkpoint_name}_weights_only.pt"

    # Create minimal checkpoint with just weights
    minimal_checkpoint = {
        "lora_state_dict": lora_weights,
        "metadata": {
            "source_checkpoint": str(checkpoint_path),
            "original_step": checkpoint.get("step", None),
            "original_epoch": checkpoint.get("epoch", None),
        }
    }

    # Add EMA weights if present
    if has_ema:
        minimal_checkpoint["ema_lora"] = checkpoint["ema_lora"]
        print(f"  Including EMA weights")

    # Add some useful config info for inference
    if "config" in checkpoint:
        config = checkpoint["config"]
        minimal_checkpoint["metadata"]["lora_rank"] = config.get("lora_rank")
        minimal_checkpoint["metadata"]["lora_alpha"] = config.get("lora_alpha")
        minimal_checkpoint["metadata"]["lora_target_mode"] = config.get("lora_target_mode")

    # Save the minimal checkpoint
    torch.save(minimal_checkpoint, output_path)

    # Calculate sizes
    original_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)  # MB
    new_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    reduction = (1 - new_size / original_size) * 100

    print(f"\nExtracted LoRA weights:")
    print(f"  Original checkpoint: {original_size:.2f} MB")
    print(f"  Weights-only file: {new_size:.2f} MB")
    print(f"  Size reduction: {reduction:.1f}%")
    print(f"  Saved to: {output_path}")

    # Count parameters
    num_params = len(lora_weights)
    total_params = sum(p.numel() for p in lora_weights.values())
    print(f"\nLoRA parameters:")
    print(f"  Number of tensors: {num_params}")
    print(f"  Total parameters: {total_params:,}")

    return str(output_path)


def load_lora_for_inference(weights_path: str) -> dict:
    """
    Load LoRA weights for inference.

    Args:
        weights_path: Path to the weights file

    Returns:
        Dictionary containing LoRA weights and metadata
    """
    checkpoint = torch.load(weights_path, map_location="cpu")

    if "lora_state_dict" not in checkpoint:
        raise ValueError(f"File does not contain LoRA weights")

    return {
        "lora_weights": checkpoint["lora_state_dict"],
        "ema_weights": checkpoint.get("ema_lora", None),
        "metadata": checkpoint.get("metadata", {})
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract LoRA weights from training checkpoint for inference"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the training checkpoint"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for weights-only file (defaults to checkpoint_weights_only.pt)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Just show info about the checkpoint without extracting"
    )

    args = parser.parse_args()

    if args.info:
        # Just show information
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

        print(f"Checkpoint: {args.checkpoint}")
        print(f"Keys: {list(checkpoint.keys())}")

        if "lora_state_dict" in checkpoint:
            lora_weights = checkpoint["lora_state_dict"]
            num_params = len(lora_weights)
            total_params = sum(p.numel() for p in lora_weights.values())
            print(f"\nLoRA weights found:")
            print(f"  Tensors: {num_params}")
            print(f"  Parameters: {total_params:,}")

            # Show first few parameter names
            print(f"\n  First 5 parameters:")
            for i, key in enumerate(list(lora_weights.keys())[:5]):
                print(f"    {key}: {lora_weights[key].shape}")

        if "ema_lora" in checkpoint:
            print(f"\nEMA LoRA weights: Present")

        if "optimizer" in checkpoint:
            print(f"\nOptimizer state: Present")

        if "config" in checkpoint:
            config = checkpoint["config"]
            print(f"\nConfiguration:")
            print(f"  LoRA rank: {config.get('lora_rank')}")
            print(f"  LoRA alpha: {config.get('lora_alpha')}")
            print(f"  Target mode: {config.get('lora_target_mode')}")
            print(f"  Step: {checkpoint.get('step')}")
            print(f"  Epoch: {checkpoint.get('epoch')}")
    else:
        # Extract weights
        extract_lora_weights(args.checkpoint, args.output)


if __name__ == "__main__":
    main()