"""
Validation Image Generation

Generate sample images during training to monitor progress.
"""

import torch
from typing import List, Optional, Any
from PIL import Image
import numpy as np
import random
from pathlib import Path


def load_random_captions(caption_file: str, num_captions: int) -> List[str]:
    """
    Load random captions from a file.

    Args:
        caption_file: Path to file with one caption per line
        num_captions: Number of captions to randomly select

    Returns:
        List of randomly selected captions
    """
    caption_path = Path(caption_file)
    if not caption_path.exists():
        raise FileNotFoundError(f"Caption file not found: {caption_file}")

    # Read all captions
    with open(caption_path, 'r', encoding='utf-8') as f:
        all_captions = [line.strip() for line in f if line.strip()]

    # Randomly sample
    if len(all_captions) < num_captions:
        # If we don't have enough, sample with replacement
        selected = random.choices(all_captions, k=num_captions)
    else:
        selected = random.sample(all_captions, num_captions)

    return selected


@torch.no_grad()
def generate_validation_images(
    prompts: List[str],
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    text_encoder,
    noise_scheduler,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,  # Typical SDXL CFG scale
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: Optional[int] = None,
) -> List[Image.Image]:
    """
    Generate validation images with Classifier-Free Guidance (CFG).

    Args:
        prompts: List of text prompts
        unet: UNet model
        vae: VAE decoder
        text_encoder: Text encoder
        noise_scheduler: Noise scheduler (must have set_timesteps and step methods)
        height: Image height
        width: Image width
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale (1.0 = no CFG, 7.5 is typical for SDXL)
        device: Device
        dtype: Data type
        seed: Random seed

    Returns:
        List of PIL Images
    """
    # Set to eval mode
    unet.eval()
    vae.eval()

    batch_size = len(prompts)

    # Set seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    # Encode conditional prompts
    prompt_embeds, pooled_embeds = text_encoder(prompts)
    prompt_embeds = prompt_embeds.to(device, dtype=dtype)
    pooled_embeds = pooled_embeds.to(device, dtype=dtype)

    # Encode unconditional prompts (empty strings) for CFG
    do_cfg = guidance_scale > 1.0
    if do_cfg:
        negative_prompts = [""] * batch_size
        negative_prompt_embeds, negative_pooled_embeds = text_encoder(negative_prompts)
        negative_prompt_embeds = negative_prompt_embeds.to(device, dtype=dtype)
        negative_pooled_embeds = negative_pooled_embeds.to(device, dtype=dtype)

        # Concatenate for classifier-free guidance
        # [uncond, cond]
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_embeds = torch.cat([negative_pooled_embeds, pooled_embeds], dim=0)

    # Prepare latents
    latent_channels = 4
    latent_height = height // 8
    latent_width = width // 8

    latents = torch.randn(
        (batch_size, latent_channels, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # Prepare additional conditioning
    from sdxl.text_encoders import get_add_time_ids

    add_time_ids = get_add_time_ids(
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        dtype=dtype,
        device=device,
        batch_size=batch_size,
    )

    # For CFG, duplicate the time_ids
    if do_cfg:
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    # Simple DDIM-like sampling (no scheduler switching needed)
    # We'll use a simplified Euler-like approach
    timesteps = torch.linspace(
        noise_scheduler.num_train_timesteps - 1,
        0,
        num_inference_steps,
        device=device,
    ).long()

    # Get alpha values for denoising
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    # Denoising loop
    for i, t in enumerate(timesteps):
        # Duplicate latents for CFG
        latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents

        # Prepare timestep
        t_batch = t.unsqueeze(0).repeat(latent_model_input.shape[0])

        # Prepare conditioning
        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": add_time_ids,
        }

        # Predict noise
        noise_pred = unet(
            latent_model_input,
            t_batch,
            prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        )[0]

        # Apply CFG
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # DDIM step (simple version)
        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)

        # Predict x0
        pred_original_sample = (latents - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)

        # Direction pointing to x_t
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev) * noise_pred

        # x_{t-1}
        latents = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

    # Decode latents
    # Remove VAE scaling factor before decoding
    vae_scale_factor = getattr(vae.config, 'scaling_factor', 0.13025)
    latents = latents / vae_scale_factor

    latents = latents.to(torch.float32)  # VAE in FP32

    # Decode
    if hasattr(vae, 'decode'):
        decoded = vae.decode(latents)
        if hasattr(decoded, 'sample'):
            # Diffusers VAE returns DecoderOutput
            images = decoded.sample
        else:
            images = decoded
    else:
        raise ValueError("VAE must have decode method")

    # Convert to PIL
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (images * 255).round().astype(np.uint8)

    pil_images = [Image.fromarray(img) for img in images]

    return pil_images


def create_image_grid(images: List[Image.Image], cols: int = 2) -> Image.Image:
    """
    Create a grid from list of images.

    Args:
        images: List of PIL Images
        cols: Number of columns

    Returns:
        Grid image
    """
    if not images:
        raise ValueError("No images provided")

    n = len(images)
    rows = (n + cols - 1) // cols

    w, h = images[0].size
    grid = Image.new("RGB", (w * cols, h * rows), color=(0, 0, 0))

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * w, row * h))

    return grid


def log_validation_images(
    images: List[Image.Image],
    prompts: List[str],
    wandb_run: Any,
    step: int,
    prefix: str = "validation",
):
    """
    Log validation images to Weights & Biases.

    Args:
        images: List of PIL Images
        prompts: List of prompts used
        wandb_run: W&B run object
        step: Current training step
        prefix: Prefix for logging
    """
    if wandb_run is None:
        return

    try:
        import wandb

        # Create grid
        grid = create_image_grid(images, cols=2)

        # Log individual images with captions
        wandb_images = [
            wandb.Image(img, caption=prompt)
            for img, prompt in zip(images, prompts)
        ]

        # Log to W&B
        wandb_run.log({
            f"{prefix}/images": wandb_images,
            f"{prefix}/grid": wandb.Image(grid),
        }, step=step)

    except Exception as e:
        print(f"Warning: Failed to log validation images to W&B: {e}")
