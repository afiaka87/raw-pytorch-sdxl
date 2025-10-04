"""
SDXL Inference and Sampling

Generate images from text prompts using trained SDXL models.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Any, Tuple
from PIL import Image
import numpy as np
from pathlib import Path


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
    images = vae.decode(latents / vae.config.scaling_factor).sample

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


def load_and_apply_lora(
    unet: nn.Module,
    lora_checkpoint_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load LoRA weights from checkpoint and apply to UNet.

    Args:
        unet: Base UNet model
        lora_checkpoint_path: Path to LoRA checkpoint
        device: Device to load to
        dtype: Data type for model

    Returns:
        (unet_with_lora, checkpoint_info)
    """
    from sdxl.lora import apply_lora_to_unet, load_lora_state_dict

    # Load checkpoint
    checkpoint = torch.load(lora_checkpoint_path, map_location="cpu")

    # Verify it's a LoRA checkpoint
    if not checkpoint.get("is_lora_checkpoint", False):
        raise ValueError(f"Not a LoRA checkpoint: {lora_checkpoint_path}")

    # Extract config
    config = checkpoint.get("config", {})
    lora_rank = config.get("lora_rank", 8)
    lora_alpha = config.get("lora_alpha", 32)
    lora_target_mode = config.get("lora_target_mode", "attention")

    # Apply LoRA to UNet
    unet = apply_lora_to_unet(
        unet,
        rank=lora_rank,
        alpha=lora_alpha,
        target_mode=lora_target_mode,
    )

    # Load LoRA weights
    lora_state_dict = checkpoint["lora_state_dict"]
    load_lora_state_dict(unet, lora_state_dict, strict=False)

    # Move to device and dtype
    unet = unet.to(device, dtype=dtype)

    # Return info
    info = {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target_mode": lora_target_mode,
    }

    return unet, info


def load_prompts_from_file(
    prompt_file: str,
    num_prompts: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Load prompts from a text file (one per line).

    Args:
        prompt_file: Path to file with prompts
        num_prompts: Number of prompts to load (None = all)
        shuffle: Whether to shuffle prompts
        seed: Random seed for shuffling

    Returns:
        List of prompt strings
    """
    import random

    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    # Load all prompts
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    if len(prompts) == 0:
        raise ValueError(f"No prompts found in {prompt_file}")

    # Shuffle if requested
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(prompts)

    # Limit number if specified
    if num_prompts is not None:
        if num_prompts > len(prompts):
            # Repeat with cycling if we need more than available
            import itertools
            prompts = list(itertools.islice(itertools.cycle(prompts), num_prompts))
        else:
            prompts = prompts[:num_prompts]

    return prompts


def generate_unique_batch(
    prompts: List[str],
    unet: nn.Module,
    vae: nn.Module,
    text_encoder,
    scheduler,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: Optional[int] = None,
) -> Tuple[List[Image.Image], List[str]]:
    """
    Generate images with unique prompts per batch item.

    This processes multiple unique prompts in parallel for efficiency.
    Each image in the batch gets its own unique prompt.

    Args:
        prompts: List of unique prompts (one per image)
        unet: UNet model
        vae: VAE decoder
        text_encoder: Text encoder
        scheduler: Noise scheduler
        height: Image height
        width: Image width
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        negative_prompt: Negative prompt(s)
        device: Device
        dtype: Data type
        seed: Random seed (base seed, each image gets seed+index)

    Returns:
        (images, prompts) - images and their corresponding prompts
    """
    batch_size = len(prompts)

    # Handle negative prompts
    if isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * batch_size
    elif negative_prompt is None:
        negative_prompt = [""] * batch_size

    # Set up generators for each image
    if seed is not None:
        generators = [torch.Generator(device=device).manual_seed(seed + i) for i in range(batch_size)]
    else:
        generators = [None] * batch_size

    # Encode prompts for all images
    do_cfg = guidance_scale > 1.0

    if do_cfg:
        # Encode both positive and negative prompts for all images
        prompt_embeds_pos, pooled_embeds_pos = text_encoder.encode_prompt(prompts, num_images_per_prompt=1)
        prompt_embeds_neg, pooled_embeds_neg = text_encoder.encode_prompt(negative_prompt, num_images_per_prompt=1)

        # Concatenate for CFG [negative, positive]
        prompt_embeds = torch.cat([prompt_embeds_neg, prompt_embeds_pos])
        pooled_embeds = torch.cat([pooled_embeds_neg, pooled_embeds_pos])
    else:
        prompt_embeds, pooled_embeds = text_encoder.encode_prompt(prompts, num_images_per_prompt=1)

    prompt_embeds = prompt_embeds.to(device, dtype=dtype)
    pooled_embeds = pooled_embeds.to(device, dtype=dtype)

    # Prepare latents for all images at once
    latent_channels = 4
    latent_height = height // 8
    latent_width = width // 8

    latents_shape = (batch_size, latent_channels, latent_height, latent_width)

    # Generate initial noise for each image with its own generator
    latents_list = []
    for gen in generators:
        latent = torch.randn(
            (1, latent_channels, latent_height, latent_width),
            generator=gen,
            device=device,
            dtype=dtype
        )
        latents_list.append(latent)
    latents = torch.cat(latents_list, dim=0)

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
        batch_size=batch_size,
    )

    if do_cfg:
        add_time_ids = torch.cat([add_time_ids, add_time_ids])

    # Denoising loop - process all images in parallel
    for i, t in enumerate(scheduler.timesteps):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents, latents]) if do_cfg else latents

        # Prepare conditioning
        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": add_time_ids,
        }

        # Predict noise for all images at once
        noise_pred = unet(
            latent_model_input,
            t,
            prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        )[0]

        # Apply classifier-free guidance
        if do_cfg:
            noise_pred_neg, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_neg + guidance_scale * (noise_pred_pos - noise_pred_neg)

        # Scheduler step for all images
        latents = scheduler.step(noise_pred, i, latents)

    # Decode latents
    latents = latents.to(torch.float32)  # VAE in FP32
    images = vae.decode(latents / vae.config.scaling_factor).sample

    # Convert to PIL
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype(np.uint8)

    pil_images = [Image.fromarray(img) for img in images]

    return pil_images, prompts


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
