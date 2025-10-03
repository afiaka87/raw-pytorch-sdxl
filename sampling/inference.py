"""
SDXL Inference and Sampling

Generate images from text prompts using trained SDXL models.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, List
from PIL import Image
import numpy as np


@torch.no_grad()
def generate(
    prompt: Union[str, List[str]],
    unet: nn.Module,
    vae: nn.Module,
    text_encoder,
    scheduler,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: Optional[int] = None,
) -> List[Image.Image]:
    """
    Generate images from text prompts.

    Args:
        prompt: Text prompt(s)
        unet: UNet model
        vae: VAE decoder
        text_encoder: Text encoder
        scheduler: Noise scheduler (Euler, DDIM, etc.)
        height: Image height
        width: Image width
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        negative_prompt: Negative prompt(s)
        num_images_per_prompt: Number of images per prompt
        device: Device to use
        dtype: Data type
        seed: Random seed

    Returns:
        List of PIL Images
    """
    # Handle single vs batch prompts
    if isinstance(prompt, str):
        prompt = [prompt]
    batch_size = len(prompt)

    if isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt]

    # Set seed
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    # Encode prompts
    do_cfg = guidance_scale > 1.0

    if do_cfg:
        # Encode both positive and negative prompts
        if negative_prompt is None:
            negative_prompt = [""] * batch_size

        prompt_embeds_pos, pooled_embeds_pos = text_encoder.encode_prompt(
            prompt, num_images_per_prompt=num_images_per_prompt
        )
        prompt_embeds_neg, pooled_embeds_neg = text_encoder.encode_prompt(
            negative_prompt, num_images_per_prompt=num_images_per_prompt
        )

        # Concatenate for CFG
        prompt_embeds = torch.cat([prompt_embeds_neg, prompt_embeds_pos])
        pooled_embeds = torch.cat([pooled_embeds_neg, pooled_embeds_pos])
    else:
        prompt_embeds, pooled_embeds = text_encoder.encode_prompt(
            prompt, num_images_per_prompt=num_images_per_prompt
        )

    prompt_embeds = prompt_embeds.to(device, dtype=dtype)
    pooled_embeds = pooled_embeds.to(device, dtype=dtype)

    # Prepare latents
    latent_channels = 4
    latent_height = height // 8
    latent_width = width // 8

    latents_shape = (
        batch_size * num_images_per_prompt,
        latent_channels,
        latent_height,
        latent_width,
    )

    latents = torch.randn(
        latents_shape, generator=generator, device=device, dtype=dtype
    )

    # Scale initial noise
    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * scheduler.sigmas[0]

    # Prepare additional conditioning
    from sdxl.text_encoders import get_add_time_ids

    add_time_ids = get_add_time_ids(
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        dtype=dtype,
        device=device,
        batch_size=batch_size * num_images_per_prompt,
    )

    if do_cfg:
        add_time_ids = torch.cat([add_time_ids, add_time_ids])

    # Denoising loop
    for i, t in enumerate(scheduler.timesteps):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents, latents]) if do_cfg else latents

        # Prepare conditioning
        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": add_time_ids,
        }

        # Predict noise
        noise_pred = unet(
            latent_model_input,
            t,
            prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        )[0]

        # Classifier-free guidance
        if do_cfg:
            noise_pred_neg, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_neg + guidance_scale * (
                noise_pred_pos - noise_pred_neg
            )

        # Scheduler step
        latents = scheduler.step(noise_pred, i, latents)

    # Decode latents
    latents = latents.to(torch.float32)  # VAE in FP32
    images = vae.decode(latents)

    # Convert to PIL
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype(np.uint8)

    pil_images = [Image.fromarray(img) for img in images]

    return pil_images


@torch.no_grad()
def generate_batch(
    prompts: List[str],
    unet: nn.Module,
    vae: nn.Module,
    text_encoder,
    scheduler,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    batch_size: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: Optional[int] = None,
) -> List[Image.Image]:
    """
    Generate images for multiple prompts in batches.

    Args:
        prompts: List of prompts
        unet: UNet model
        vae: VAE decoder
        text_encoder: Text encoder
        scheduler: Noise scheduler
        height: Image height
        width: Image width
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        batch_size: Batch size for processing
        device: Device
        dtype: Data type
        seed: Random seed

    Returns:
        List of PIL Images (one per prompt)
    """
    all_images = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        # Adjust seed for each batch if provided
        batch_seed = seed + i if seed is not None else None

        images = generate(
            prompt=batch_prompts,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            scheduler=scheduler,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            device=device,
            dtype=dtype,
            seed=batch_seed,
        )

        all_images.extend(images)

    return all_images


def create_image_grid(images: List[Image.Image], cols: int = 4) -> Image.Image:
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
    grid = Image.new("RGB", (w * cols, h * rows), color=(255, 255, 255))

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * w, row * h))

    return grid
