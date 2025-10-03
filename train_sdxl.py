#!/usr/bin/env python3
"""
SDXL Fine-Tuning Script

Train SDXL with LoRA on custom datasets.
"""

import argparse
import torch
import os
from pathlib import Path

# Import modules
from sdxl.unet import UNet2DConditionModel
from sdxl.vae import AutoencoderKL
from sdxl.text_encoders import SDXLTextEncoder
from sdxl.diffusion import DDPMScheduler
from sdxl.lora import apply_lora_to_unet, get_lora_state_dict

from training.data_loader import create_dataloader
from training.train_loop import train_epoch, validate
from training.ema import SimpleEMA
from training.precision_util import get_dtype_from_str, print_memory_stats
from training.train_util import (
    save_checkpoint,
    load_checkpoint,
    setup_wandb,
    create_output_directory,
    save_config,
    count_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SDXL with LoRA")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--val_data_dir", type=str, default=None, help="Validation data directory")

    # Model arguments
    parser.add_argument("--pretrained_model", type=str, default="./weights/sdxl-base-1.0",
                       help="Path to pretrained SDXL weights")
    parser.add_argument("--vae_model", type=str, default=None,
                       help="Path to VAE weights (optional)")

    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for training")
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=32.0,
                       help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                       help="LoRA dropout")
    parser.add_argument("--lora_target_mode", type=str, default="attention",
                       choices=["attention", "attention_out", "all"],
                       help="Which modules to target with LoRA")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,
                       help="AdamW weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="Adam beta2")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")

    # Precision arguments
    parser.add_argument("--precision", type=str, default="bf16",
                       choices=["fp32", "bf16", "fp16"],
                       help="Training precision")

    # Image arguments
    parser.add_argument("--image_size", type=int, default=1024,
                       help="Image resolution")
    parser.add_argument("--center_crop", action="store_true",
                       help="Use center crop")
    parser.add_argument("--random_flip", action="store_true", default=True,
                       help="Use random horizontal flip")

    # EMA arguments (disabled by default due to VRAM constraints)
    parser.add_argument("--use_ema", action="store_true", default=False,
                       help="Use EMA (requires ~2x VRAM)")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                       help="EMA decay rate")

    # Checkpointing arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--save_interval", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume from checkpoint")

    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="W&B run name")

    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Get device and dtype
    device = torch.device(args.device)
    dtype = get_dtype_from_str(args.precision)

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    save_config(vars(args), str(output_dir / "config.json"))

    # Setup W&B
    wandb_run = None
    if args.wandb_project:
        wandb_run = setup_wandb(
            project_name=args.wandb_project,
            run_name=args.wandb_run_name,
            config=vars(args),
        )

    # Initialize models
    print("Initializing models...")

    # UNet - load from pretrained if path provided
    if args.pretrained_model:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model,
            torch_dtype=dtype,
            device=device,
        )
    else:
        unet = UNet2DConditionModel().to(device, dtype=dtype)

    # Enable gradient checkpointing manually to save memory
    # Wrap down/up blocks in checkpointing
    from torch.utils.checkpoint import checkpoint

    def make_checkpointed(module):
        """Wrap a module's forward with gradient checkpointing"""
        original_forward = module.forward
        def checkpointed_forward(*args, **kwargs):
            return checkpoint(original_forward, *args, **kwargs, use_reentrant=False)
        module.forward = checkpointed_forward
        return module

    # Apply checkpointing to down/mid/up blocks
    for block in unet.down_blocks:
        make_checkpointed(block)
    make_checkpointed(unet.mid_block)
    for block in unet.up_blocks:
        make_checkpointed(block)

    print("Enabled gradient checkpointing on UNet blocks")

    # Apply LoRA if requested
    if args.use_lora:
        print(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
        unet = apply_lora_to_unet(
            unet,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_mode=args.lora_target_mode,
        )

    # VAE (keep in FP32 for quality)
    # Use diffusers VAE since it's frozen (not part of training)
    if args.pretrained_model:
        from diffusers import AutoencoderKL as DiffusersVAE
        vae = DiffusersVAE.from_pretrained(
            args.pretrained_model,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(device)
    else:
        vae = AutoencoderKL().to(device, dtype=torch.float32)
    vae.eval()
    vae.requires_grad_(False)

    # Text encoders (frozen) - these load from HF directly via transformers
    if args.pretrained_model:
        text_encoder = SDXLTextEncoder(
            model_path=args.pretrained_model,
            device=device,
            torch_dtype=torch.float16
        )
    else:
        text_encoder = SDXLTextEncoder(device=device, torch_dtype=torch.float16)
    text_encoder.freeze()

    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    # Print parameter counts
    total_params, trainable_params = count_parameters(unet)
    print(f"\nUNet Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%\n")

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )

    # EMA
    ema_model = None
    if args.use_ema:
        print(f"Initializing EMA (decay={args.ema_decay})...")
        ema_model = SimpleEMA(unet, decay=args.ema_decay)

    # Data loaders
    print("Creating data loaders...")
    train_dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        shuffle=True,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
    )

    val_dataloader = None
    if args.val_data_dir:
        val_dataloader = create_dataloader(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            shuffle=False,
            center_crop=args.center_crop,
            random_flip=False,
        )

    # Resume from checkpoint
    global_step = 0
    start_epoch = 0
    if args.resume_from:
        print(f"Resuming from {args.resume_from}...")
        checkpoint_info = load_checkpoint(
            args.resume_from,
            model=unet,
            optimizer=optimizer,
            ema_model=ema_model,
            device=device,
        )
        global_step = checkpoint_info["step"]
        start_epoch = checkpoint_info["epoch"]

    # Print memory stats
    if device.type == "cuda":
        print_memory_stats(device)

    # Training loop
    print("\nStarting training...\n")
    for epoch in range(start_epoch, args.num_epochs):
        print(f"=== Epoch {epoch + 1}/{args.num_epochs} ===")

        # Train
        avg_loss, global_step = train_epoch(
            dataloader=train_dataloader,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            ema_model=ema_model,
            device=device,
            dtype=dtype,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval,
            wandb_run=wandb_run,
            global_step=global_step,
        )

        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # Validate
        if val_dataloader is not None:
            val_loss = validate(
                dataloader=val_dataloader,
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                noise_scheduler=noise_scheduler,
                device=device,
                dtype=dtype,
            )
            print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")

            if wandb_run:
                from training.train_util import log_to_wandb
                log_to_wandb({"val/loss": val_loss}, step=global_step, wandb_run=wandb_run)

        # Save checkpoint
        if (epoch + 1) % max(1, args.save_interval // len(train_dataloader)) == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                path=str(checkpoint_path),
                model=unet,
                optimizer=optimizer,
                ema_model=ema_model,
                step=global_step,
                epoch=epoch + 1,
                config=vars(args),
            )

            # Save LoRA weights separately
            if args.use_lora:
                lora_path = output_dir / f"lora_epoch_{epoch + 1:04d}.pt"
                lora_state = get_lora_state_dict(unet)
                torch.save(lora_state, lora_path)
                print(f"Saved LoRA weights to {lora_path}")

    # Final checkpoint
    final_path = output_dir / "checkpoint_final.pt"
    save_checkpoint(
        path=str(final_path),
        model=unet,
        optimizer=optimizer,
        ema_model=ema_model,
        step=global_step,
        epoch=args.num_epochs,
        config=vars(args),
    )

    if args.use_lora:
        final_lora_path = output_dir / "lora_final.pt"
        lora_state = get_lora_state_dict(unet)
        torch.save(lora_state, final_lora_path)
        print(f"Saved final LoRA weights to {final_lora_path}")

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
