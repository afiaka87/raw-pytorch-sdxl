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
        # latents are already scaled by VAE
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
    # For simplicity, assume all 1024x1024 with no cropping
    from sdxl.text_encoders import get_add_time_ids

    original_size = (1024, 1024)
    crops_coords = (0, 0)
    target_size = (1024, 1024)

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

    loss = F.mse_loss(model_pred, target, reduction="mean")

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
        )

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

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
                    f"Samples/sec: {samples_per_sec:.2f}"
                )

                # Log to W&B
                if wandb_run is not None:
                    from training.train_util import log_to_wandb, get_lr

                    log_to_wandb(
                        {
                            "train/loss": avg_loss,
                            "train/samples_per_sec": samples_per_sec,
                            "train/lr": get_lr(optimizer),
                        },
                        step=global_step,
                        wandb_run=wandb_run,
                    )

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
