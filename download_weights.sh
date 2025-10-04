#!/bin/bash
#
# Download SDXL weights using huggingface-cli
#
# This downloads stabilityai/stable-diffusion-xl-base-1.0 to ./weights/sdxl-base-1.0/
#

set -e

REPO_ID="stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_DIR="./weights/sdxl-base-1.0"

echo "=== Downloading SDXL Base 1.0 Weights ==="
echo "Repository: $REPO_ID"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if huggingface-cli is installed
if ! command -v uv run huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found"
    echo "Install with: pip install huggingface-hub[cli]"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Download the model
echo "Downloading model files..."
uv run huggingface-cli download "$REPO_ID" \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "=== Download Complete ==="
echo "Model downloaded to: $OUTPUT_DIR"
echo ""
echo "Directory contents:"
ls -lh "$OUTPUT_DIR"
