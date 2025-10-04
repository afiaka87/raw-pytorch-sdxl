#!/usr/bin/env python
"""
Generate images using SDXL with optional LoRA weights.

Supports:
- Loading LoRA checkpoints
- Batch generation with unique prompts
- COCO-style dataset output (numbered dirs with image + caption)
- Multiple resolutions
"""

import argparse
import torch
import os
from pathlib import Path
from typing import List, Tuple, Optional
import json
from tqdm import tqdm

from sampling.inference import (
    generate,
    generate_unique_batch,
    load_and_apply_lora,
    load_prompts_from_file,
)
from sdxl.unet import UNet2DConditionModel
from sdxl.text_encoders import SDXLTextEncoder
from sdxl.diffusion import EulerDiscreteScheduler, DDIMScheduler


def parse_image_size(size_str: str) -> Tuple[int, int]:
    """
    Parse image size string.

    Examples:
        "512" -> (512, 512)
        "1024" -> (1024, 1024)
        "512x768" -> (512, 768)
    """
    if 'x' in size_str:
        w, h = size_str.split('x')
        return int(w), int(h)
    else:
        size = int(size_str)
        return size, size


def save_coco_style(
    images: List,
    prompts: List[str],
    output_dir: Path,
    start_index: int = 0,
    metadata: Optional[dict] = None,
):
    """
    Save images in COCO-style format with numbered directories.

    Structure:
        output_dir/
            00000/
                image.png
                caption.txt
                metadata.json (optional)
            00001/
                ...
    """
    for i, (image, prompt) in enumerate(zip(images, prompts)):
        # Create numbered directory
        idx = start_index + i
        dir_name = f"{idx:05d}"
        item_dir = output_dir / dir_name
        item_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        image_path = item_dir / "image.png"
        image.save(image_path)

        # Save caption
        caption_path = item_dir / "caption.txt"
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(prompt)

        # Save metadata if provided
        if metadata:
            meta_path = item_dir / "metadata.json"
            item_metadata = {
                **metadata,
                "index": idx,
                "prompt": prompt,
                "image_path": str(image_path.name),
            }
            with open(meta_path, 'w') as f:
                json.dump(item_metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate images with SDXL + LoRA")

    # Model paths
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="./weights/sdxl-base-1.0",
        help="Path to pretrained SDXL model directory",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        help="Path to LoRA checkpoint file (.pt)",
    )

    # Prompts
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to generate",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=str,
        help="File with prompts (one per line)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt for CFG",
    )

    # Generation settings
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (images processed in parallel)",
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default="1024",
        help="Image size: 512, 768, 1024, or WxH (e.g., 512x768)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["euler", "ddim"],
        default="euler",
        help="Noise scheduler to use",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./generated_images",
        help="Output directory for images",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["coco", "simple"],
        default="coco",
        help="Output format: coco (numbered dirs) or simple (just images)",
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save generation metadata with each image",
    )

    # Hardware settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Model precision",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--shuffle-prompts",
        action="store_true",
        help="Shuffle prompts from file",
    )

    args = parser.parse_args()

    # Set precision
    if args.precision == "fp32":
        dtype = torch.float32
    elif args.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    # Parse image size
    width, height = parse_image_size(args.image_size)
    print(f"Generating at {width}x{height}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    if args.prompt_file:
        prompts = load_prompts_from_file(
            args.prompt_file,
            num_prompts=args.num_images,
            shuffle=args.shuffle_prompts,
            seed=args.seed,
        )
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    elif args.prompt:
        prompts = [args.prompt] * args.num_images
        print(f"Using single prompt: {args.prompt}")
    else:
        parser.error("Either --prompt or --prompt-file must be specified")

    # Load models
    print("Loading models...")

    # Load UNet
    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        device=args.device,
    )

    # Apply LoRA if provided
    lora_info = None
    if args.lora_checkpoint:
        print(f"Applying LoRA from {args.lora_checkpoint}...")
        unet, lora_info = load_and_apply_lora(
            unet,
            args.lora_checkpoint,
            device=args.device,
            dtype=dtype,
        )
        print(f"LoRA loaded: rank={lora_info['lora_rank']}, alpha={lora_info['lora_alpha']}")
        print(f"Checkpoint step={lora_info['step']}, epoch={lora_info['epoch']}")

    # Load VAE (always FP32 for quality)
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae = vae.to(args.device)
    vae.eval()

    # Load text encoders
    print("Loading text encoders...")
    text_encoder = SDXLTextEncoder(
        device=args.device,
        torch_dtype=torch.float16,
        model_path=args.pretrained_model,
    )

    # Prepare metadata
    metadata = None
    if args.save_metadata:
        metadata = {
            "model": args.pretrained_model,
            "lora_checkpoint": args.lora_checkpoint,
            "lora_info": lora_info,
            "width": width,
            "height": height,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "scheduler": args.scheduler,
            "seed": args.seed,
            "negative_prompt": args.negative_prompt,
        }

    # Generate images in batches
    print(f"\nGenerating {args.num_images} images in batches of {args.batch_size}...")

    all_images = []
    all_prompts = []

    with torch.inference_mode():
        for batch_start in tqdm(range(0, args.num_images, args.batch_size)):
            batch_end = min(batch_start + args.batch_size, args.num_images)
            batch_prompts = prompts[batch_start:batch_end]

            # Create fresh scheduler for each batch
            if args.scheduler == "euler":
                scheduler = EulerDiscreteScheduler()
            else:
                scheduler = DDIMScheduler()

            # Generate batch with unique prompts
            if args.batch_size == 1 or len(batch_prompts) == 1:
                # Single image generation
                images = generate(
                    prompt=batch_prompts[0],
                    unet=unet,
                    vae=vae,
                    text_encoder=text_encoder,
                    scheduler=scheduler,
                    height=height,
                    width=width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=args.negative_prompt,
                    num_images_per_prompt=1,
                    device=args.device,
                    dtype=dtype,
                    seed=(args.seed + batch_start) if args.seed else None,
                )
                used_prompts = [batch_prompts[0]]
            else:
                # Batch generation with unique prompts
                images, used_prompts = generate_unique_batch(
                    prompts=batch_prompts,
                    unet=unet,
                    vae=vae,
                    text_encoder=text_encoder,
                    scheduler=scheduler,
                    height=height,
                    width=width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=args.negative_prompt,
                    device=args.device,
                    dtype=dtype,
                    seed=(args.seed + batch_start) if args.seed else None,
                )

            all_images.extend(images)
            all_prompts.extend(used_prompts)

    # Save images
    print(f"\nSaving {len(all_images)} images to {output_dir}...")

    if args.output_format == "coco":
        # COCO-style numbered directories
        save_coco_style(
            all_images,
            all_prompts,
            output_dir,
            start_index=0,
            metadata=metadata,
        )
        print(f"Saved in COCO format: {output_dir}/00000/ to {output_dir}/{len(all_images)-1:05d}/")
    else:
        # Simple format - just save images
        for i, (image, prompt) in enumerate(zip(all_images, all_prompts)):
            image_path = output_dir / f"image_{i:05d}.png"
            image.save(image_path)

            # Optionally save prompt
            prompt_path = output_dir / f"image_{i:05d}.txt"
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(prompt)

        print(f"Saved images: {output_dir}/image_00000.png to {output_dir}/image_{len(all_images)-1:05d}.png")

    print("\nGeneration complete!")

    # Print summary
    if lora_info:
        print(f"\nLoRA Info:")
        print(f"  Rank: {lora_info['lora_rank']}")
        print(f"  Alpha: {lora_info['lora_alpha']}")
        print(f"  Step: {lora_info['step']}")
        print(f"  Epoch: {lora_info['epoch']}")


if __name__ == "__main__":
    main()