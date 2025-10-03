"""
Download SDXL weights from Hugging Face Hub.

Uses huggingface-cli for efficient downloading with caching.
"""

import subprocess
import os
from pathlib import Path
from typing import Optional


def download_sdxl_weights(
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    local_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """
    Download SDXL weights using huggingface-cli.

    Args:
        model_id: HuggingFace model ID
        local_dir: Local directory to download to (optional)
        cache_dir: Cache directory (optional)

    Returns:
        Path to downloaded weights
    """
    if local_dir is None:
        local_dir = f"./weights/{model_id.split('/')[-1]}"

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id} to {local_dir}...")

    # Build huggingface-cli command
    cmd = [
        "huggingface-cli",
        "download",
        model_id,
        "--local-dir",
        str(local_path),
        "--local-dir-use-symlinks",
        "False",
    ]

    # Only download specific files
    cmd.extend(
        [
            "--include",
            "*.safetensors",
            "*.json",
            "*.txt",
        ]
    )

    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])

    # Run download
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"Downloaded weights to {local_path}")
        return local_path
    except subprocess.CalledProcessError as e:
        print(f"Error downloading weights: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def download_vae(
    model_id: str = "madebyollin/sdxl-vae-fp16-fix",
    local_dir: Optional[str] = None,
) -> Path:
    """
    Download improved SDXL VAE.

    The default SDXL VAE has issues with FP16. This version is fixed.

    Args:
        model_id: VAE model ID
        local_dir: Local directory

    Returns:
        Path to downloaded VAE
    """
    if local_dir is None:
        local_dir = "./weights/sdxl-vae-fp16-fix"

    return download_sdxl_weights(model_id=model_id, local_dir=local_dir)


def check_weights_exist(weights_dir: str) -> bool:
    """
    Check if weights directory contains necessary files.

    Args:
        weights_dir: Directory to check

    Returns:
        True if weights exist
    """
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        return False

    # Check for essential files
    required_patterns = [
        "unet/*.safetensors",
        "vae/*.safetensors",
        "text_encoder*/*.safetensors",
    ]

    for pattern in required_patterns:
        if not list(weights_path.glob(pattern)):
            return False

    return True


def main():
    """Download all necessary weights."""
    import argparse

    parser = argparse.ArgumentParser(description="Download SDXL weights")
    parser.add_argument(
        "--model-id",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--local-dir",
        default="./weights/sdxl-base-1.0",
        help="Local directory to save weights",
    )
    parser.add_argument(
        "--download-vae",
        action="store_true",
        help="Also download improved FP16 VAE",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for downloads",
    )

    args = parser.parse_args()

    # Download main model
    download_sdxl_weights(
        model_id=args.model_id,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
    )

    # Download VAE if requested
    if args.download_vae:
        download_vae()

    print("\nDownload complete!")
    print(f"Weights saved to: {args.local_dir}")


if __name__ == "__main__":
    main()
