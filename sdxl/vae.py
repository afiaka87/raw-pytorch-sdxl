"""
SDXL VAE (Variational Autoencoder) in Pure PyTorch

Encodes images (3x1024x1024) to latents (4x128x128) with 8x spatial compression.
Based on the Stable Diffusion VAE architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResnetBlock(nn.Module):
    """ResNet block for VAE encoder/decoder."""

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return self.shortcut(x) + h


class AttnBlock(nn.Module):
    """Self-attention block for VAE."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels**-0.5

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]
        k = k.reshape(b, c, h * w)  # [B, C, HW]
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]

        attn = torch.bmm(q, k) * self.scale  # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)  # [B, HW, C]
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj_out(out)

        return x + out


class Downsample(nn.Module):
    """Downsampling by 2x."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling by 2x."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    """
    VAE Encoder: 3x1024x1024 -> 8x128x128 -> 4x128x128 (via bottleneck)

    Architecture:
    - 4 downsample stages (1024->512->256->128->128)
    - Channel progression: [128, 256, 512, 512]
    - Attention at 128x128 resolution
    """

    def __init__(
        self,
        in_channels=3,
        latent_channels=8,
        channels=128,
        channel_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[128],
    ):
        super().__init__()
        self.num_resolutions = len(channel_mult)

        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)

        # Downsampling
        self.down = nn.ModuleList()
        in_ch = channels
        for i, mult in enumerate(channel_mult):
            out_ch = channels * mult
            blocks = nn.ModuleList()

            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch

                # Add attention at specified resolutions
                # Resolution at stage i: 1024 / 2^i
                if 1024 // (2**i) in attn_resolutions:
                    blocks.append(AttnBlock(out_ch))

            # Downsample (except last)
            if i < self.num_resolutions - 1:
                blocks.append(Downsample(out_ch))

            self.down.append(blocks)

        # Middle
        self.mid = nn.ModuleList(
            [
                ResnetBlock(out_ch, out_ch),
                AttnBlock(out_ch),
                ResnetBlock(out_ch, out_ch),
            ]
        )

        # Output
        self.norm_out = nn.GroupNorm(32, out_ch, eps=1e-6)
        self.conv_out = nn.Conv2d(out_ch, latent_channels, 3, padding=1)

    def forward(self, x):
        # Initial conv
        h = self.conv_in(x)

        # Downsampling
        for blocks in self.down:
            for block in blocks:
                h = block(h)

        # Middle
        for block in self.mid:
            h = block(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class Decoder(nn.Module):
    """
    VAE Decoder: 4x128x128 -> 8x128x128 -> 3x1024x1024

    Symmetric to encoder with upsampling.
    """

    def __init__(
        self,
        out_channels=3,
        latent_channels=4,
        channels=128,
        channel_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[128],
    ):
        super().__init__()
        self.num_resolutions = len(channel_mult)

        # Initial conv
        out_ch = channels * channel_mult[-1]
        self.conv_in = nn.Conv2d(latent_channels, out_ch, 3, padding=1)

        # Middle
        self.mid = nn.ModuleList(
            [
                ResnetBlock(out_ch, out_ch),
                AttnBlock(out_ch),
                ResnetBlock(out_ch, out_ch),
            ]
        )

        # Upsampling
        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            mult = channel_mult[i]
            in_ch = channels * mult
            blocks = nn.ModuleList()

            # Upsample first (except last)
            if i < self.num_resolutions - 1:
                blocks.append(Upsample(out_ch))

            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(out_ch, in_ch))
                out_ch = in_ch

                # Add attention at specified resolutions
                if 1024 // (2**i) in attn_resolutions:
                    blocks.append(AttnBlock(in_ch))

            self.up.append(blocks)

        # Output
        self.norm_out = nn.GroupNorm(32, out_ch, eps=1e-6)
        self.conv_out = nn.Conv2d(out_ch, out_channels, 3, padding=1)

    def forward(self, z):
        # Initial conv
        h = self.conv_in(z)

        # Middle
        for block in self.mid:
            h = block(h)

        # Upsampling
        for blocks in self.up:
            for block in blocks:
                h = block(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class DiagonalGaussianDistribution:
    """
    Diagonal Gaussian distribution for VAE latent space.

    Takes encoder output [B, 8, H, W] and splits into mean and logvar.
    """

    def __init__(self, parameters: torch.Tensor):
        # Split into mean and logvar
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        """Sample from distribution using reparameterization trick."""
        noise = torch.randn_like(self.mean)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        """Return mean (mode of Gaussian)."""
        return self.mean

    def kl(self) -> torch.Tensor:
        """Compute KL divergence with standard normal."""
        kl = 0.5 * (self.var + self.mean**2 - 1.0 - self.logvar)
        return kl.sum(dim=[1, 2, 3])


class AutoencoderKL(nn.Module):
    """
    Variational Autoencoder with KL divergence.

    SDXL VAE configuration:
    - Input: 3x1024x1024 images
    - Latent: 4x128x128 (8x spatial compression)
    - Scaling factor: 0.13025
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        channels=128,
        channel_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[128],
        scaling_factor=0.13025,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels * 2,  # For mean and logvar
            channels=channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
        )

        self.decoder = Decoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            channels=channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
        )

        # Quant conv layers
        self.quant_conv = nn.Conv2d(latent_channels * 2, latent_channels * 2, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    @classmethod
    def from_pretrained(cls, pretrained_path, torch_dtype=None, device=None):
        """
        Load VAE from pretrained SDXL weights.

        Args:
            pretrained_path: Path to directory containing vae/diffusion_pytorch_model.safetensors
            torch_dtype: Target dtype (keep as FP32 for quality)
            device: Target device

        Returns:
            Loaded VAE model
        """
        from pathlib import Path
        from safetensors.torch import load_file

        # Create instance with default SDXL config
        model = cls()

        # Load weights
        weights_path = Path(pretrained_path) / "vae" / "diffusion_pytorch_model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"VAE weights not found at {weights_path}")

        print(f"Loading VAE weights from {weights_path}...")
        state_dict = load_file(str(weights_path))

        # Load into model
        model.load_state_dict(state_dict, strict=True)
        print("VAE weights loaded successfully")

        # Convert dtype if requested
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        # Move to device if requested
        if device is not None:
            model = model.to(device=device)

        return model

    def encode(self, x: torch.Tensor, sample=True) -> torch.Tensor:
        """
        Encode images to latents.

        Args:
            x: [B, 3, H, W] images in range [-1, 1]
            sample: If True, sample from distribution. If False, use mean.

        Returns:
            [B, 4, H//8, W//8] scaled latents
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        dist = DiagonalGaussianDistribution(h)

        if sample:
            z = dist.sample()
        else:
            z = dist.mode()

        # Scale latents
        z = z * self.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images.

        Args:
            z: [B, 4, H, W] scaled latents

        Returns:
            [B, 3, H*8, W*8] images in range [-1, 1]
        """
        # Unscale latents
        z = z / self.scaling_factor

        z = self.post_quant_conv(z)
        x = self.decoder(z)
        return x

    def forward(
        self, x: torch.Tensor, sample=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass.

        Returns:
            (reconstructed_images, kl_divergence)
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        dist = DiagonalGaussianDistribution(h)

        if sample:
            z = dist.sample()
        else:
            z = dist.mode()

        kl = dist.kl()

        # Decode
        z = self.post_quant_conv(z)
        recon = self.decoder(z)

        return recon, kl
