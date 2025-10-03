# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A minimal, pure PyTorch implementation for fine-tuning Stable Diffusion XL with LoRA. The project prioritizes clarity and hackability over abstraction, implementing SDXL training without relying on high-level libraries like diffusers/peft/accelerate for the core training loop.

**Key Philosophy**: This is an educational and research-focused implementation that provides complete visibility into the SDXL training process. The codebase adapts code from minSDXL (Apache 2.0 License) and prioritizes understanding over convenience.

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Download SDXL weights (stabilityai/stable-diffusion-xl-base-1.0 to ./weights/)
./download_weights.sh  # Note: This script doesn't exist yet, refer to README
```

### Training
```bash
# Basic LoRA fine-tuning (minimal VRAM, 512x512)
uv run python train_sdxl.py \
  --data_dir my_dataset \
  --use_lora \
  --lora_rank 4 \
  --lora_alpha 16 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_epochs 10 \
  --learning_rate 1e-4 \
  --output_dir ./outputs \
  --pretrained_model ./weights/sdxl-base-1.0 \
  --image_size 512
```

### Code Quality
```bash
# Format code
black .

# Lint
ruff check .

# Type checking
mypy .
```

## Architecture

### Core Module Organization

**`sdxl/`** - Pure PyTorch SDXL model components
- `unet.py`: SDXL UNet2DConditionModel implementation (main model)
- `lora.py`: Custom LoRA implementation with `LoRALinear` and `LoRALinearWrapper` classes
- `vae.py`: VAE encoder/decoder (custom implementation, but diffusers VAE used in practice)
- `text_encoders.py`: Dual CLIP text encoders (CLIP-L + OpenCLIP-G) with `SDXLTextEncoder` wrapper
- `diffusion.py`: Noise schedulers (DDPM/Euler/DDIM)

**`training/`** - Training infrastructure
- `train_loop.py`: Core training logic (`train_step`, `train_epoch`, `validate`)
- `data_loader.py`: Dataset loading for image-caption pairs
- `precision_util.py`: BF16/FP16/FP32 utilities
- `ema.py`: Exponential moving average (disabled by default due to VRAM)
- `train_util.py`: Checkpointing, W&B logging, utilities

**`sampling/`** - Inference
- `inference.py`: Image generation from trained models

**`minSDXL/`** - Reference implementation (vendored, Apache 2.0)
- Original minSDXL implementation used as reference
- Not actively used in main training pipeline

**`train_sdxl.py`** - Main training script entry point

### Critical Implementation Details

1. **LoRA Implementation** (`sdxl/lora.py`)
   - Custom LoRA without peft dependency
   - Freeze base model FIRST, then apply LoRA, then unfreeze LoRA params (order matters!)
   - LoRA layers must be moved to same device/dtype as base layer during init
   - `apply_lora_to_unet()` creates `LoRALinearWrapper` that wraps frozen Linear layers
   - `get_lora_state_dict()` extracts only LoRA weights for saving

2. **Gradient Checkpointing** (`train_sdxl.py:170-185`)
   - Applied to UNet down_blocks, mid_block, and up_blocks
   - Uses `torch.utils.checkpoint` with `use_reentrant=False`
   - Critical for fitting in <12GB VRAM

3. **VAE Handling** (`train_sdxl.py:199-210`)
   - **Always uses diffusers VAE**, not the custom implementation
   - Kept in FP32 for quality (frozen during training)
   - Custom VAE in `sdxl/vae.py` has architecture mismatch with pretrained weights

4. **Text Encoders** (`sdxl/text_encoders.py`)
   - Dual text encoders: CLIP-L (OpenAI) + OpenCLIP-G (laion)
   - Always frozen during training
   - Loads from HuggingFace transformers

5. **Precision Management**
   - Model: BF16 (recommended) or FP16
   - VAE: Always FP32 (quality)
   - Text encoders: FP16
   - Manual dtype conversion required in forward passes (train_loop.py)

6. **SDXL Conditioning** (`training/train_loop.py:82-97`)
   - Requires `add_time_ids` for original_size, crops_coords, target_size
   - `added_cond_kwargs` passed to UNet with text_embeds (pooled) and time_ids

## Memory Optimization

### VRAM Requirements (with LoRA)
- **<12GB VRAM**: 512x512, rank 4, BF16 + gradient checkpointing (~7.5GB)
- **16GB VRAM**: 768x768, rank 8, optional EMA
- **24GB+ VRAM**: 1024x1024, rank 16 or full fine-tuning

### Memory Breakdown (512x512, rank 4)
```
Model weights (BF16):       ~5.2GB
LoRA parameters:            ~9MB
Optimizer state (AdamW):    ~18MB
Gradients (LoRA only):      ~9MB
Activations (checkpointed): ~2.0GB
Temporary buffers:          ~0.3GB
Total:                      ~7.5GB
```

### Key Constraints
- Text encoders frozen (unfreezing requires more memory)
- VAE frozen (always FP32)
- EMA disabled by default (~2x VRAM)
- 512x512 resolution required for <12GB VRAM

## Dataset Format

### Regular Dataset (Directory)
Organize image-caption pairs:
```
my_dataset/
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
...
```

Each `.txt` file contains the caption for the corresponding image.

### WebDataset (Tar Shards)
Organize as tar archives:
```
data_dir/
├── shard_00000.tar  (contains: img_001.png, img_001.txt, img_002.png, img_002.txt, ...)
├── shard_00001.tar
├── shard_00002.tar
...
```

WebDataset format is **auto-detected** if .tar files are present in the data directory. Each tar shard contains paired .png/.jpg and .txt files with matching basenames.

## Key Training Arguments

**LoRA Configuration**:
- `--lora_rank`: Default 8, use 4 for <12GB VRAM (lower = fewer params)
- `--lora_alpha`: Default 32 (scaling factor)
- `--lora_target_mode`: attention/attention_out/all (which layers to target)

**Memory-Critical**:
- `--image_size`: 512 (for <12GB), 1024 (for 16GB+)
- `--precision`: bf16 (recommended), fp16, fp32
- `--use_ema`: Disabled by default (requires ~2x VRAM)

**Training**:
- `--batch_size`: Per-GPU batch size (default 1)
- `--gradient_accumulation_steps`: Effective batch = batch_size × this

## Output Files

Training produces:
- `lora_final.pt`: LoRA weights only (~9MB for rank 4)
- `checkpoint_final.pt`: Full checkpoint with optimizer state
- `config.json`: Training configuration

## Common Issues & Solutions

1. **OOM on 1024x1024**: Reduce to 512x512 or increase VRAM
2. **LoRA params not training**: Check freeze order (freeze base → apply LoRA → unfreeze LoRA)
3. **LoRA on CPU**: LoRA layers created before device move - fixed in current impl
4. **Dtype mismatches**: Ensure manual dtype conversion in forward passes
5. **VAE architecture mismatch**: Always use diffusers VAE, not custom implementation

## Development Notes

- SDXL requires dual text encoders (CLIP-L + OpenCLIP-G)
- Attention computation at 1024x1024 creates massive temporary matrices (256MB per layer × 70+ layers)
- Gradient checkpointing alone insufficient for <12GB VRAM at full resolution
- LoRA rank 4 with alpha 16 is the sweet spot (0.17% of model, sufficient expressiveness)
- BF16 more stable than FP16 (no loss scaling needed)

## References

- minSDXL implementation: `minSDXL/README.md`
- Implementation journey: `ADVANCED.md` (detailed lessons learned)
- Training script: `train_sdxl.py` (main entry point)
