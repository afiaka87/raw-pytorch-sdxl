"""
SDXL Training Loop

Main training logic for fine-tuning SDXL with LoRA.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import time


def train_step(
    batch: Dict[str, Any],
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    text_encoder,
    noise_scheduler,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    min_snr_gamma: Optional[float] = None,
) -> torch.Tensor:
    """
    Single training step.

    Args:
        batch: Batch with 'images' and 'captions'
        unet: UNet model (with LoRA)
        vae: VAE model for encoding images
        text_encoder: Text encoder(s)
        noise_scheduler: Noise scheduler for diffusion
        device: Device to use
        dtype: Data type for computation

    Returns:
        Loss tensor
    """
    images = batch["images"].to(device, dtype=dtype)
    captions = batch["captions"]
    batch_size = images.shape[0]

    # Encode images to latents
    with torch.no_grad():
        # VAE in FP32 for better quality
        # Encode images to latents
        # Handle both custom VAE and diffusers VAE
        if hasattr(vae, 'encode'):
            encoded = vae.encode(images.to(torch.float32))
            if hasattr(encoded, 'latent_dist'):
                # Diffusers VAE returns AutoencoderKLOutput
                latents = encoded.latent_dist.sample()
            elif isinstance(encoded, torch.Tensor):
                # Custom VAE returns tensor directly
                latents = encoded
            else:
                latents = encoded
        else:
            raise ValueError("VAE must have encode method")

        # CRITICAL: Apply VAE scaling factor
        # Diffusers VAE does NOT automatically scale latents
        # SDXL VAE scaling factor is 0.13025
        vae_scale_factor = getattr(vae.config, 'scaling_factor', 0.13025)
        latents = latents * vae_scale_factor
        latents = latents.to(dtype)

    # Encode text
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = text_encoder(captions)
        prompt_embeds = prompt_embeds.to(dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype)

    # Sample noise
    noise = torch.randn_like(latents)

    # Sample timesteps
    timesteps = torch.randint(
        0,
        noise_scheduler.num_train_timesteps,
        (batch_size,),
        device=device,
    ).long()

    # Add noise to latents
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Prepare additional conditioning for SDXL
    # SDXL uses original_size, crops_coords, and target_size
    # Use actual image dimensions from the batch
    from sdxl.text_encoders import get_add_time_ids

    # Get actual image size from batch (images are [B, C, H, W])
    img_height, img_width = images.shape[2], images.shape[3]
    original_size = (img_height, img_width)
    crops_coords = (0, 0)
    target_size = (img_height, img_width)

    add_time_ids = get_add_time_ids(
        original_size,
        crops_coords,
        target_size,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
    )

    # Prepare added_cond_kwargs
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds,
        "time_ids": add_time_ids,
    }

    # Predict noise
    model_pred = unet(
        noisy_latents,
        timesteps,
        prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
    )[0]  # UNet returns list

    # Compute loss
    if noise_scheduler.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type: {noise_scheduler.prediction_type}")

    # MSE loss (per-sample, not reduced yet)
    loss = F.mse_loss(model_pred, target, reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape))))  # Mean over all dims except batch

    # Apply Min-SNR loss weighting if enabled
    # This helps stabilize training by preventing overfitting to certain timesteps
    # Reference: "Efficient Diffusion Training via Min-SNR Weighting Strategy"
    if min_snr_gamma is not None:
        # Compute SNR (Signal-to-Noise Ratio)
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device)
        snr = alphas_cumprod[timesteps] / (1.0 - alphas_cumprod[timesteps])

        # Min-SNR weighting: min(SNR, gamma) / SNR
        # This prevents the loss from being dominated by noisy timesteps
        mse_loss_weights = torch.stack([snr, min_snr_gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr

        # Apply weighting
        loss = loss * mse_loss_weights

    # Return mean loss
    loss = loss.mean()

    return loss


def train_epoch(
    dataloader,
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    text_encoder,
    noise_scheduler,
    optimizer: torch.optim.Optimizer,
    ema_model: Optional[Any] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    log_interval: int = 10,
    wandb_run: Optional[Any] = None,
    global_step: int = 0,
    min_snr_gamma: Optional[float] = None,
    lr_scheduler: Optional[Any] = None,
    validation_caption_file: Optional[str] = None,
    validation_interval: int = 0,
    validation_num_images: int = 4,
    validation_guidance_scale: float = 7.5,
    image_size: int = 512,
    save_interval: int = 0,
    save_callback: Optional[callable] = None,
) -> tuple[float, int]:
    """
    Train for one epoch.

    Args:
        dataloader: Training dataloader
        unet: UNet model
        vae: VAE model
        text_encoder: Text encoder
        noise_scheduler: Noise scheduler
        optimizer: Optimizer
        ema_model: EMA model (optional)
        device: Device
        dtype: Data type
        gradient_accumulation_steps: Steps to accumulate gradients
        max_grad_norm: Max gradient norm for clipping
        log_interval: Logging interval
        wandb_run: W&B run object
        global_step: Starting global step

    Returns:
        (avg_loss, final_global_step)
    """
    unet.train()
    total_loss = 0.0
    num_batches = 0

    # Save initial LoRA params for tracking changes
    initial_lora_params = {}
    for n, p in unet.named_parameters():
        if 'lora' in n.lower() and p.requires_grad:
            initial_lora_params[n] = p.data.clone()

    optimizer.zero_grad()

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Training step
        loss = train_step(
            batch,
            unet,
            vae,
            text_encoder,
            noise_scheduler,
            device=device,
            dtype=dtype,
            min_snr_gamma=min_snr_gamma,
        )

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Compute gradient norm before clipping (for monitoring)
            grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), float('inf'))

            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)

            # Debug: Check if LoRA parameters have gradients
            if global_step == 1:
                lora_params_with_grad = 0
                lora_params_without_grad = 0
                for n, p in unet.named_parameters():
                    if 'lora' in n.lower() and p.requires_grad:
                        if p.grad is not None:
                            lora_params_with_grad += 1
                        else:
                            lora_params_without_grad += 1
                print(f"\n  DEBUG: LoRA params with gradients: {lora_params_with_grad}")
                print(f"  DEBUG: LoRA params without gradients: {lora_params_without_grad}\n")

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Update learning rate scheduler
            if lr_scheduler is not None:
                lr_scheduler.step()

            # EMA update
            if ema_model is not None:
                ema_model.update(unet)

            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                avg_loss = total_loss / num_batches
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * batch["images"].shape[0] / elapsed

                print(
                    f"Step {global_step} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Grad Norm: {grad_norm:.4f} | "
                    f"Samples/sec: {samples_per_sec:.2f}"
                )

                # Log to W&B
                if wandb_run is not None:
                    from training.train_util import log_to_wandb, get_lr

                    log_to_wandb(
                        {
                            "train/loss": avg_loss,
                            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "train/samples_per_sec": samples_per_sec,
                            "train/lr": get_lr(optimizer),
                        },
                        step=global_step,
                        wandb_run=wandb_run,
                    )

            # Validation image generation
            # Skip step 0 validation (no training has occurred yet)
            if validation_interval > 0 and global_step > 0 and global_step % validation_interval == 0 and validation_caption_file:
                print(f"\nGenerating validation images at step {global_step}...")

                # Check if LoRA parameters have actually changed
                total_change = 0.0
                max_change = 0.0
                num_params_changed = 0
                for n, p in unet.named_parameters():
                    if 'lora' in n.lower() and p.requires_grad and n in initial_lora_params:
                        param_change = (p.data - initial_lora_params[n]).abs().max().item()
                        total_change += param_change
                        max_change = max(max_change, param_change)
                        if param_change > 1e-10:
                            num_params_changed += 1

                print(f"  LoRA param changes:")
                print(f"    Max change: {max_change:.10f}")
                print(f"    Avg change: {total_change / len(initial_lora_params):.10f}")
                print(f"    Params changed: {num_params_changed}/{len(initial_lora_params)}")

                from training.validation import generate_validation_images, log_validation_images, load_random_captions

                try:
                    # Load random captions for this validation run
                    random_prompts = load_random_captions(validation_caption_file, validation_num_images)
                    print(f"  Using {len(random_prompts)} random captions from {validation_caption_file}")
                    print(f"  CFG guidance scale: {validation_guidance_scale}")

                    val_images = generate_validation_images(
                        prompts=random_prompts,
                        unet=unet,
                        vae=vae,
                        text_encoder=text_encoder,
                        noise_scheduler=noise_scheduler,
                        height=image_size,
                        width=image_size,
                        num_inference_steps=20,
                        guidance_scale=validation_guidance_scale,  # Use CFG with text conditioning
                        device=device,
                        dtype=dtype,
                        seed=None,  # Different noise each time for variety
                    )

                    # Log to W&B
                    log_validation_images(
                        images=val_images,
                        prompts=random_prompts,
                        wandb_run=wandb_run,
                        step=global_step,
                        prefix="validation",
                    )

                    print(f"Generated {len(val_images)} validation images")

                    # Put model back in train mode
                    unet.train()

                except Exception as e:
                    print(f"Warning: Validation image generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    unet.train()  # Make sure we're back in train mode

            # Save checkpoint at step intervals
            if save_interval > 0 and global_step > 0 and global_step % save_interval == 0 and save_callback:
                print(f"\nSaving checkpoint at step {global_step}...")
                try:
                    save_callback(global_step=global_step, epoch=None)
                    print(f"Checkpoint saved successfully")
                except Exception as e:
                    print(f"Warning: Failed to save checkpoint: {e}")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


def validate(
    dataloader,
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    text_encoder,
    noise_scheduler,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_batches: int = 10,
) -> float:
    """
    Run validation.

    Args:
        dataloader: Validation dataloader
        unet: UNet model
        vae: VAE model
        text_encoder: Text encoder
        noise_scheduler: Noise scheduler
        device: Device
        dtype: Data type
        num_batches: Number of batches to validate

    Returns:
        Average validation loss
    """
    unet.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            loss = train_step(
                batch,
                unet,
                vae,
                text_encoder,
                noise_scheduler,
                device=device,
                dtype=dtype,
            )

            total_loss += loss.item()
            count += 1

    unet.train()
    avg_loss = total_loss / max(count, 1)
    return avg_loss
