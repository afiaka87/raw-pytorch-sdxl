"""
SDXL Dual Text Encoders

Uses two CLIP models:
1. OpenAI CLIP-L/14 (768-dim)
2. OpenCLIP-bigG/14 (1280-dim)

Outputs are concatenated to produce 2048-dim text embeddings for cross-attention.
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Tuple, Optional


class SDXLTextEncoder(nn.Module):
    """
    Dual text encoder for SDXL.

    Combines CLIP-L and OpenCLIP-G to produce 2048-dim embeddings.
    """

    def __init__(
        self,
        clip_l_model="openai/clip-vit-large-patch14",
        clip_g_model="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        torch_dtype=torch.float32,
        device="cuda",
        model_path=None,
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype

        # If model_path is provided, load from local SDXL weights
        if model_path:
            from pathlib import Path
            model_path = Path(model_path)
            clip_l_model = str(model_path / "text_encoder")
            clip_g_model = str(model_path / "text_encoder_2")
            tokenizer_l_path = str(model_path / "tokenizer")
            tokenizer_g_path = str(model_path / "tokenizer_2")
        else:
            tokenizer_l_path = clip_l_model
            tokenizer_g_path = clip_g_model

        # Load CLIP-L (OpenAI)
        self.tokenizer_l = CLIPTokenizer.from_pretrained(tokenizer_l_path)
        self.text_encoder_l = CLIPTextModel.from_pretrained(
            clip_l_model, torch_dtype=torch_dtype
        ).to(device)

        # Load CLIP-G (OpenCLIP)
        # Note: Using the transformers-compatible version
        self.tokenizer_g = CLIPTokenizer.from_pretrained(
            tokenizer_g_path,
            # OpenCLIP uses different tokenizer
            # We'll use the laion model which is compatible
        )
        self.text_encoder_g = CLIPTextModel.from_pretrained(
            clip_g_model, torch_dtype=torch_dtype
        ).to(device)

        # Freeze by default
        self.freeze()

    def freeze(self):
        """Freeze both text encoders."""
        for param in self.text_encoder_l.parameters():
            param.requires_grad = False
        for param in self.text_encoder_g.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze both text encoders."""
        for param in self.text_encoder_l.parameters():
            param.requires_grad = True
        for param in self.text_encoder_g.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def encode_prompt(
        self,
        prompts: list[str],
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts using both CLIP models.

        Args:
            prompts: List of text prompts
            num_images_per_prompt: Number of images to generate per prompt
            do_classifier_free_guidance: If True, also encode empty prompts

        Returns:
            (prompt_embeds, pooled_prompt_embeds)
            - prompt_embeds: [B, seq_len, 2048] concatenated embeddings
            - pooled_prompt_embeds: [B, 1280] pooled embeddings from CLIP-G
        """
        batch_size = len(prompts)

        # Tokenize for both models
        tokens_l = self.tokenizer_l(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        tokens_g = self.tokenizer_g(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Encode with CLIP-L
        outputs_l = self.text_encoder_l(tokens_l, output_hidden_states=True)
        prompt_embeds_l = outputs_l.hidden_states[-2]  # [B, 77, 768]

        # Encode with CLIP-G
        outputs_g = self.text_encoder_g(tokens_g, output_hidden_states=True)
        prompt_embeds_g = outputs_g.hidden_states[-2]  # [B, 77, 1280]

        # Pooled embeddings from CLIP-G (using the <EOS> token)
        pooled_prompt_embeds = outputs_g.pooler_output  # [B, 1280]

        # Concatenate embeddings: [B, 77, 768] + [B, 77, 1280] -> [B, 77, 2048]
        prompt_embeds = torch.cat([prompt_embeds_l, prompt_embeds_g], dim=-1)

        # Duplicate for each image per prompt
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(
                num_images_per_prompt, dim=0
            )

        # For classifier-free guidance, encode empty prompts
        if do_classifier_free_guidance:
            negative_prompt_embeds = self.encode_prompt(
                [""] * batch_size, num_images_per_prompt=num_images_per_prompt
            )
            # Concatenate [negative, positive]
            prompt_embeds = torch.cat([negative_prompt_embeds[0], prompt_embeds])
            pooled_prompt_embeds = torch.cat(
                [negative_prompt_embeds[1], pooled_prompt_embeds]
            )

        return prompt_embeds, pooled_prompt_embeds

    def forward(self, prompts: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training (with gradients if unfrozen).

        Args:
            prompts: List of text prompts

        Returns:
            (prompt_embeds, pooled_prompt_embeds)
        """
        batch_size = len(prompts)

        # Tokenize
        tokens_l = self.tokenizer_l(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        tokens_g = self.tokenizer_g(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Encode
        outputs_l = self.text_encoder_l(tokens_l, output_hidden_states=True)
        prompt_embeds_l = outputs_l.hidden_states[-2]

        outputs_g = self.text_encoder_g(tokens_g, output_hidden_states=True)
        prompt_embeds_g = outputs_g.hidden_states[-2]
        pooled_prompt_embeds = outputs_g.pooler_output

        # Concatenate
        prompt_embeds = torch.cat([prompt_embeds_l, prompt_embeds_g], dim=-1)

        return prompt_embeds, pooled_prompt_embeds


def get_add_time_ids(
    original_size: Tuple[int, int],
    crops_coords_top_left: Tuple[int, int],
    target_size: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Get additional time embeddings for SDXL.

    SDXL uses additional conditioning on image size and crop coordinates.

    Args:
        original_size: (height, width) of original image
        crops_coords_top_left: (top, left) crop coordinates
        target_size: (height, width) of target image
        dtype: torch dtype
        device: torch device
        batch_size: batch size

    Returns:
        [B, 6] tensor with [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    """
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids


# Simplified single-encoder version for testing
class SimpleCLIPTextEncoder(nn.Module):
    """
    Single CLIP text encoder (for testing/debugging).
    """

    def __init__(
        self,
        model_name="openai/clip-vit-large-patch14",
        torch_dtype=torch.float32,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(device)

        # Freeze by default
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> torch.Tensor:
        """Encode prompts to embeddings."""
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        outputs = self.text_encoder(tokens, output_hidden_states=True)
        embeddings = outputs.hidden_states[-2]  # Use penultimate layer

        return embeddings

    def forward(self, prompts: list[str]) -> torch.Tensor:
        """Forward pass with gradients."""
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        outputs = self.text_encoder(tokens, output_hidden_states=True)
        embeddings = outputs.hidden_states[-2]

        return embeddings
