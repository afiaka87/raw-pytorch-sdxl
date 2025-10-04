#!/usr/bin/env python
"""
Generate images using SDXL with optional LoRA weights.

Supports:
- Loading LoRA checkpoints
- Batch generation with unique prompts
- COCO-style dataset output (numbered dirs with image + caption)
- Multiple resolutions
- Multi-GPU parallelization
"""

import argparse
import torch
import torch.multiprocessing as mp
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
    Save images in COCO-style format with numbered files.

    Structure:
        output_dir/
            00000.png
            00000.txt
            00000.json (optional metadata)
            00001.png
            00001.txt
            00001.json
            ...
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (image, prompt) in enumerate(zip(images, prompts)):
        # Generate numbered filename
        idx = start_index + i
        base_name = f"{idx:05d}"

        # Save image
        image_path = output_dir / f"{base_name}.png"
        image.save(image_path)

        # Save caption
        caption_path = output_dir / f"{base_name}.txt"
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(prompt)

        # Save metadata if provided
        if metadata:
            meta_path = output_dir / f"{base_name}.json"
            item_metadata = {
                **metadata,
                "index": idx,
                "prompt": prompt,
                "image_file": f"{base_name}.png",
                "caption_file": f"{base_name}.txt",
            }
            with open(meta_path, 'w') as f:
                json.dump(item_metadata, f, indent=2)


def worker_generate(
    gpu_id: int,
    prompts: List[str],
    args_dict: dict,
    output_dir: Path,
    start_index: int,
    metadata: Optional[dict],
):
    """
    Worker function to generate images on a specific GPU.

    Args:
        gpu_id: GPU device ID to use
        prompts: List of prompts for this worker
        args_dict: Dictionary of generation arguments
        output_dir: Output directory for images
        start_index: Starting index for naming files
        metadata: Metadata to save with images
    """
    # Set device for this worker
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    # Extract args
    pretrained_model = args_dict['pretrained_model']
    lora_checkpoint = args_dict.get('lora_checkpoint')
    width = args_dict['width']
    height = args_dict['height']
    batch_size = args_dict['batch_size']
    num_inference_steps = args_dict['num_inference_steps']
    guidance_scale = args_dict['guidance_scale']
    negative_prompt = args_dict['negative_prompt']
    scheduler_type = args_dict['scheduler']
    dtype = args_dict['dtype']
    seed = args_dict.get('seed')

    # Load models on this GPU
    print(f"[GPU {gpu_id}] Loading models...")

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model,
        torch_dtype=dtype,
        device=device,
    )

    # Apply LoRA if provided
    if lora_checkpoint:
        print(f"[GPU {gpu_id}] Applying LoRA from {lora_checkpoint}...")
        unet, lora_info = load_and_apply_lora(
            unet,
            lora_checkpoint,
            device=device,
            dtype=dtype,
        )

    # Load VAE (always FP32)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        pretrained_model,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae = vae.to(device)
    vae.eval()

    # Load text encoders
    text_encoder = SDXLTextEncoder(
        device=device,
        torch_dtype=torch.float16,
        model_path=pretrained_model,
    )

    print(f"[GPU {gpu_id}] Generating {len(prompts)} images...")

    # Generate images in batches
    all_images = []
    all_prompts = []

    with torch.inference_mode():
        for batch_start in tqdm(range(0, len(prompts), batch_size), desc=f"GPU {gpu_id}"):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            # Create fresh scheduler for each batch
            if scheduler_type == "euler":
                scheduler = EulerDiscreteScheduler()
            else:
                scheduler = DDIMScheduler()

            # Generate batch
            if batch_size == 1 or len(batch_prompts) == 1:
                images = generate(
                    prompt=batch_prompts[0],
                    unet=unet,
                    vae=vae,
                    text_encoder=text_encoder,
                    scheduler=scheduler,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=1,
                    device=device,
                    dtype=dtype,
                    seed=(seed + start_index + batch_start) if seed else None,
                )
                used_prompts = [batch_prompts[0]]
            else:
                images, used_prompts = generate_unique_batch(
                    prompts=batch_prompts,
                    unet=unet,
                    vae=vae,
                    text_encoder=text_encoder,
                    scheduler=scheduler,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    device=device,
                    dtype=dtype,
                    seed=(seed + start_index + batch_start) if seed else None,
                )

            all_images.extend(images)
            all_prompts.extend(used_prompts)

    # Save images from this worker
    print(f"[GPU {gpu_id}] Saving {len(all_images)} images...")
    save_coco_style(
        all_images,
        all_prompts,
        output_dir,
        start_index=start_index,
        metadata=metadata,
    )

    print(f"[GPU {gpu_id}] Complete!")


def main():
    parser = argparse.ArgumentParser(description="Generate images with SDXL + LoRA (Multi-GPU)")

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
        help="Device to use (ignored if --num-gpus > 1)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for parallel generation (default: auto-detect all available)",
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

    # Determine number of GPUs to use
    num_gpus_available = torch.cuda.device_count()
    if args.num_gpus is None:
        num_gpus = num_gpus_available
    else:
        num_gpus = min(args.num_gpus, num_gpus_available)

    if num_gpus == 0:
        parser.error("No GPUs available. This script requires CUDA.")

    print(f"\n=== Multi-GPU Configuration ===")
    print(f"Available GPUs: {num_gpus_available}")
    print(f"Using GPUs: {num_gpus}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"================================\n")

    # Prepare metadata
    metadata = None
    if args.save_metadata:
        metadata = {
            "model": args.pretrained_model,
            "lora_checkpoint": args.lora_checkpoint,
            "width": width,
            "height": height,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "scheduler": args.scheduler,
            "seed": args.seed,
            "negative_prompt": args.negative_prompt,
        }

    # Prepare arguments dict for workers
    args_dict = {
        'pretrained_model': args.pretrained_model,
        'lora_checkpoint': args.lora_checkpoint,
        'width': width,
        'height': height,
        'batch_size': args.batch_size,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'negative_prompt': args.negative_prompt,
        'scheduler': args.scheduler,
        'dtype': dtype,
        'seed': args.seed,
    }

    if num_gpus == 1:
        # Single GPU - use original sequential approach
        print("Using single GPU mode...")
        worker_generate(
            gpu_id=0,
            prompts=prompts,
            args_dict=args_dict,
            output_dir=output_dir,
            start_index=0,
            metadata=metadata,
        )
    else:
        # Multi-GPU parallel generation
        print(f"Splitting {len(prompts)} prompts across {num_gpus} GPUs...")

        # Split prompts among GPUs
        prompts_per_gpu = len(prompts) // num_gpus
        prompt_splits = []
        start_indices = []

        for i in range(num_gpus):
            start = i * prompts_per_gpu
            if i == num_gpus - 1:
                # Last GPU gets any remaining prompts
                end = len(prompts)
            else:
                end = (i + 1) * prompts_per_gpu

            prompt_splits.append(prompts[start:end])
            start_indices.append(start)
            print(f"  GPU {i}: {len(prompt_splits[-1])} prompts (indices {start} to {end-1})")

        # Launch workers
        print(f"\nLaunching {num_gpus} workers...")
        mp.set_start_method('spawn', force=True)

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=worker_generate,
                args=(
                    gpu_id,
                    prompt_splits[gpu_id],
                    args_dict,
                    output_dir,
                    start_indices[gpu_id],
                    metadata,
                )
            )
            p.start()
            processes.append(p)

        # Wait for all workers to complete
        for p in processes:
            p.join()

    print("\n=== Generation Complete! ===")
    print(f"Total images generated: {len(prompts)}")
    print(f"Output directory: {output_dir}")
    if args.output_format == "coco":
        print(f"Format: COCO-style ({output_dir}/00000.png to {output_dir}/{len(prompts)-1:05d}.png)")
    print("============================\n")


if __name__ == "__main__":
    main()