#!/usr/bin/env python3
"""
Resume Training Helper Script

Utilities for inspecting checkpoints and resuming training.
"""

import argparse
import sys
from pathlib import Path

from training.lora_checkpoint import (
    inspect_checkpoint,
    get_latest_lora_checkpoint,
    create_resume_command,
    is_lora_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(
        description="Resume SDXL training from checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a checkpoint
  python resume_training.py --inspect checkpoints/0034/lora_step000500_20241003_162048.pt

  # Find and inspect the latest checkpoint
  python resume_training.py --find-latest checkpoints/0034/

  # Generate resume command for a checkpoint
  python resume_training.py --resume checkpoints/0034/lora_step000500_20241003_162048.pt

  # Automatically find latest and generate resume command
  python resume_training.py --auto checkpoints/0034/
        """
    )

    parser.add_argument("--inspect", type=str, help="Inspect a checkpoint file")
    parser.add_argument("--find-latest", type=str, help="Find latest LoRA checkpoint in directory")
    parser.add_argument("--resume", type=str, help="Generate resume command for checkpoint")
    parser.add_argument("--auto", type=str, help="Auto find latest checkpoint and generate resume command")
    parser.add_argument("--execute", action="store_true", help="Execute the resume command (use with --resume or --auto)")

    args = parser.parse_args()

    # Count how many actions were specified
    actions = sum([
        args.inspect is not None,
        args.find_latest is not None,
        args.resume is not None,
        args.auto is not None,
    ])

    if actions == 0:
        parser.print_help()
        sys.exit(0)

    # Handle inspect
    if args.inspect:
        checkpoint_path = args.inspect
        if not Path(checkpoint_path).exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        inspect_checkpoint(checkpoint_path)
        print()

    # Handle find-latest
    if args.find_latest:
        checkpoint_dir = args.find_latest
        latest = get_latest_lora_checkpoint(checkpoint_dir)
        if latest:
            print(f"Latest LoRA checkpoint: {latest}")
            if not args.inspect:  # Don't double-inspect
                inspect_checkpoint(latest)
        else:
            print(f"No LoRA checkpoints found in {checkpoint_dir}")
            sys.exit(1)

    # Handle resume
    if args.resume:
        checkpoint_path = args.resume
        if not Path(checkpoint_path).exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        if not is_lora_checkpoint(checkpoint_path):
            print(f"Warning: {checkpoint_path} does not appear to be a LoRA checkpoint")
            print("This script is designed for LoRA checkpoints. Full model checkpoints can be resumed directly.")

        # Load checkpoint to get config
        import torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "config" not in checkpoint:
            print(f"Error: Checkpoint does not contain training configuration")
            sys.exit(1)

        resume_cmd = create_resume_command(checkpoint_path, checkpoint["config"])
        print("\nResume command:")
        print("=" * 60)
        print(resume_cmd)
        print("=" * 60)

        if args.execute:
            print("\nExecuting resume command...")
            import subprocess
            subprocess.run(resume_cmd, shell=True)

    # Handle auto (find latest and generate resume command)
    if args.auto:
        checkpoint_dir = args.auto
        latest = get_latest_lora_checkpoint(checkpoint_dir)
        if not latest:
            print(f"No LoRA checkpoints found in {checkpoint_dir}")
            sys.exit(1)

        print(f"Found latest checkpoint: {latest}\n")
        inspect_checkpoint(latest)

        # Load checkpoint to get config
        import torch
        checkpoint = torch.load(latest, map_location="cpu")
        if "config" not in checkpoint:
            print(f"Error: Checkpoint does not contain training configuration")
            sys.exit(1)

        resume_cmd = create_resume_command(latest, checkpoint["config"])
        print("\nResume command:")
        print("=" * 60)
        print(resume_cmd)
        print("=" * 60)

        if args.execute:
            print("\nExecuting resume command...")
            import subprocess
            subprocess.run(resume_cmd, shell=True)
        else:
            print("\nTip: Add --execute to automatically run this command")


if __name__ == "__main__":
    main()