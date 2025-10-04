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

# Resume from LoRA checkpoint
uv run python resume_training.py --auto checkpoints/latest/ --execute
```

### Inference & Generation
```bash
# Generate images with LoRA
uv run python generate_images.py \
  --lora-checkpoint lora_step_1800_lowest_loss.pt \
  --prompt-file captions.txt \
  --num-images 100 \
  --batch-size 4 \
  --image-size 512 \
  --output-dir ./generated_dataset

# Single prompt generation
uv run python generate_images.py \
  --lora-checkpoint my_lora.pt \
  --prompt "a beautiful landscape" \
  --num-images 10 \
  --image-size 768x512
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
- `inference.py`: Core generation functions with LoRA support, batch processing, and prompt file loading

**`minSDXL/`** - Reference implementation (vendored, Apache 2.0)
- Original minSDXL implementation used as reference
- Not actively used in main training pipeline

**Scripts**:
- `train_sdxl.py`: Main training script entry point
- `generate_images.py`: Image generation with LoRA, batch processing, COCO-style output
- `resume_training.py`: Checkpoint inspection and training resumption helper
- `extract_lora_weights.py`: Extract weights-only files from training checkpoints

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

### Training Outputs
Training produces:
- `lora_step{XXXXXX}_{timestamp}.pt`: LoRA checkpoints with optimizer state (~20MB for rank 16)
- `lora_epoch{XXXX}_{timestamp}.pt`: Epoch-based checkpoints
- `config.json`: Training configuration

### Generation Outputs (COCO-style)
```
output_dir/
├── 00000.png          # Generated image
├── 00000.txt          # Caption/prompt used
├── 00000.json         # Optional: metadata for this image
├── 00001.png
├── 00001.txt
├── 00001.json
├── 00002.png
├── 00002.txt
├── 00002.json
...
```

Each numbered basename groups related files (image, caption, metadata).

## Common Issues & Solutions

### Training Issues
1. **OOM on 1024x1024**: Reduce to 512x512 or increase VRAM
2. **LoRA params not training**: Check freeze order (freeze base → apply LoRA → unfreeze LoRA)
3. **LoRA on CPU**: LoRA layers created before device move - fixed in current impl
4. **Dtype mismatches**: Ensure manual dtype conversion in forward passes
5. **VAE architecture mismatch**: Always use diffusers VAE, not custom implementation
6. **Checkpoints not saving**: Ensure `--save_interval` is set for step-based saving with large datasets

### Inference Issues
1. **CUDA index errors with batch>1**: Create fresh scheduler for each batch (scheduler state gets modified)
2. **VAE decode returns DecoderOutput**: Access `.sample` attribute and divide by `vae.config.scaling_factor`
3. **Text encoder dtype issues**: Use `torch_dtype` parameter, not `dtype`
4. **Batch processing different prompts**: Use `generate_unique_batch()` not standard `generate()`
5. **LoRA not applying**: Ensure `apply_lora_to_unet()` called AFTER freezing base model

## Inference & Generation Pipeline

### Key Implementation Details

1. **LoRA Loading for Inference** (`sampling/inference.py`)
   - Load checkpoint with `torch.load(path, map_location="cpu")`
   - Apply LoRA structure to UNet AFTER freezing base weights
   - Load LoRA state dict with `load_lora_state_dict()`
   - Move to target device/dtype after loading

2. **Unique-Pair Batch Processing**
   - Standard `generate()`: Multiple images from SAME prompt
   - `generate_unique_batch()`: Each image gets DIFFERENT prompt
   - Enables efficient parallel processing of diverse prompts
   - Each image in batch gets unique random seed (base_seed + index)

3. **Scheduler State Management**
   - **CRITICAL**: Create fresh scheduler instance for each batch
   - Scheduler's internal state (sigmas, timesteps) gets modified during generation
   - Reusing scheduler across batches causes CUDA index errors

4. **VAE Decoding**
   - VAE returns `DecoderOutput` object, not raw tensor
   - Access decoded images via `.sample` attribute
   - Scale latents by `1 / vae.config.scaling_factor` before decoding
   - Keep VAE in FP32 for quality

5. **Prompt File Processing**
   - Supports line-separated text files (one prompt per line)
   - Can shuffle prompts with seed for reproducibility
   - Automatically cycles prompts if requesting more images than available prompts

## Development Notes

- SDXL requires dual text encoders (CLIP-L + OpenCLIP-G)
- Attention computation at 1024x1024 creates massive temporary matrices (256MB per layer × 70+ layers)
- Gradient checkpointing alone insufficient for <12GB VRAM at full resolution
- LoRA rank 4 with alpha 16 is the sweet spot (0.17% of model, sufficient expressiveness)
- BF16 more stable than FP16 (no loss scaling needed)
- Generation batch size 4 works well at 512x512 on 24GB VRAM
- COCO-style output enables direct use as training dataset

## References

- minSDXL implementation: `minSDXL/README.md`
- Implementation journey: `ADVANCED.md` (detailed lessons learned)
- Training script: `train_sdxl.py` (main entry point)
