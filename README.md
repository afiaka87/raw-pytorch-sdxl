# SDXL Fine-tuning in Pure PyTorch

A minimal, pure PyTorch implementation for fine-tuning Stable Diffusion XL with LoRA.

## Features

- Pure PyTorch implementation (no diffusers/peft/accelerate for training code)
- Custom LoRA implementation for parameter-efficient fine-tuning
- BF16 mixed precision training
- Gradient checkpointing for memory efficiency
- Supports GPUs with <12GB VRAM (at 512x512 resolution)
- Compatible with HuggingFace SDXL weights

## Quick Start

### 1. Download SDXL weights

```bash
./download_weights.sh
```

This downloads `stabilityai/stable-diffusion-xl-base-1.0` to `./weights/`

### 2. Prepare your dataset

Organize images and captions:
```
my_dataset/
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
...
```

### 3. Train

```bash
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

## Training Arguments

### Model
- `--pretrained_model`: Path to SDXL weights directory
- `--vae_model`: Optional improved VAE path

### LoRA
- `--use_lora`: Enable LoRA training
- `--lora_rank`: Rank (default: 8, lower = fewer params)
- `--lora_alpha`: Alpha scaling (default: 32)
- `--lora_target_mode`: Which layers to target (attention/attention_out/all)

### Training
- `--batch_size`: Batch size per GPU (default: 1)
- `--gradient_accumulation_steps`: Effective batch = batch_size * this
- `--num_epochs`: Training epochs
- `--learning_rate`: Learning rate (default: 1e-4)
- `--image_size`: Resolution (512 or 1024, default: 1024)
- `--precision`: bf16 (recommended), fp16, or fp32

### Checkpointing
- `--output_dir`: Where to save outputs
- `--save_interval`: Save every N epochs

## Memory Requirements

At 512x512 resolution with LoRA rank 4:
- ~7.5GB VRAM (BF16 + gradient checkpointing)
- Training: ~4.5M parameters (0.17% of model)

At 1024x1024 resolution:
- Requires >12GB VRAM (recommended: 16GB+)

## Output Files

Training produces:
- `lora_final.pt`: LoRA weights only (~9MB for rank 4)
- `checkpoint_final.pt`: Full training checkpoint with optimizer state
- `config.json`: Training configuration

## Project Structure

```
sdxl/
├── unet.py           # SDXL UNet implementation
├── lora.py           # Custom LoRA implementation
├── vae.py            # VAE encoder/decoder
├── text_encoders.py  # Dual CLIP text encoders
└── diffusion.py      # Noise schedulers (DDPM/Euler/DDIM)

training/
├── train_loop.py     # Main training logic
├── data_loader.py    # Dataset and data loading
├── precision_util.py # BF16 utilities
├── ema.py            # Exponential moving average
└── train_util.py     # Checkpointing and W&B

sampling/
└── inference.py      # Image generation

train_sdxl.py         # Main training script
```

## Limitations

- 512x512 resolution required for <12GB VRAM
- EMA disabled by default (requires ~2x VRAM)
- Text encoders frozen (unfreezing requires more memory)
- VAE frozen (always in FP32 for quality)

## Dependencies

- PyTorch >= 2.0.0
- transformers >= 4.35.0
- safetensors
- Pillow
- tqdm
- wandb (optional)

Install with: `uv sync`

## License

This project adapts code from minSDXL (Apache 2.0 License) and follows the same licensing.
