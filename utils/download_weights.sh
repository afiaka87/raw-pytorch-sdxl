#!/bin/bash
set -e

# Download SDXL weights using huggingface-cli
# This script downloads stabilityai/stable-diffusion-xl-base-1.0

WEIGHTS_DIR="./weights"
mkdir -p "$WEIGHTS_DIR"

echo "Downloading SDXL base model..."
uv run huggingface-cli download \
    stabilityai/stable-diffusion-xl-base-1.0 \
    --local-dir "$WEIGHTS_DIR/sdxl-base-1.0" \
    --local-dir-use-symlinks False

echo ""
echo "Downloading improved VAE (optional but recommended)..."
uv run huggingface-cli download \
    madebyollin/sdxl-vae-fp16-fix \
    --local-dir "$WEIGHTS_DIR/sdxl-vae-fp16-fix" \
    --local-dir-use-symlinks False

echo ""
echo "✅ Download complete!"
echo ""
echo "Weights saved to:"
echo "  - $WEIGHTS_DIR/sdxl-base-1.0/"
echo "  - $WEIGHTS_DIR/sdxl-vae-fp16-fix/"
