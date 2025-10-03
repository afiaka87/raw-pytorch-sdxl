"""
Diffusion Process and Noise Scheduler for SDXL

Implements:
- DDPM forward process (adding noise)
- DDIM/Euler samplers for reverse process (denoising)
- Beta schedule management
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Models (DDPM) noise scheduler.

    Implements the forward diffusion process and utilities for training.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
    ):
        """
        Args:
            num_train_timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: "linear" or "scaled_linear"
            prediction_type: "epsilon" (predict noise) or "v_prediction"
        """
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        # Create beta schedule
        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # SDXL uses scaled linear: sqrt of linear space
            betas = (
                np.linspace(
                    beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32
                )
                ** 2
            )
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.betas = torch.from_numpy(betas)

        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1), self.alphas_cumprod[:-1]]
        )

        # Variance schedule
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / self.alphas_cumprod - 1.0
        )

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0) = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise

        Args:
            original_samples: [B, C, H, W] clean samples
            noise: [B, C, H, W] random noise
            timesteps: [B] timesteps

        Returns:
            [B, C, H, W] noisy samples
        """
        # Get coefficients (move timesteps to scheduler device for indexing)
        timesteps_cpu = timesteps.cpu()
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps_cpu].to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps_cpu].to(
            device=original_samples.device, dtype=original_samples.dtype
        )

        # Reshape for broadcasting: [B, 1, 1, 1]
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Add noise
        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )

        return noisy_samples

    def get_velocity(
        self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity prediction target.

        v = sqrt(alpha) * noise - sqrt(1 - alpha) * sample
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(device=sample.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(
            device=sample.device
        )

        # Reshape
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


class EulerDiscreteScheduler:
    """
    Euler method for sampling.

    Fast, deterministic sampler that works well with 20-30 steps.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
    ):
        self.num_train_timesteps = num_train_timesteps

        # Create beta schedule
        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            betas = (
                np.linspace(
                    beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32
                )
                ** 2
            )
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        # Convert to sigmas
        sigmas = np.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        self.sigmas = torch.from_numpy(sigmas).float()

        self.timesteps = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """Set the timesteps for sampling."""
        self.num_inference_steps = num_inference_steps

        # Linearly spaced timesteps
        timesteps = np.linspace(
            0, self.num_train_timesteps - 1, num_inference_steps, dtype=np.float32
        )[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)

        # Get corresponding sigmas
        sigmas = self.sigmas.to(device)[self.timesteps.long()]
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        One step of Euler sampling.

        Args:
            model_output: Predicted noise from model
            timestep: Current timestep index
            sample: Current sample x_t

        Returns:
            Previous sample x_{t-1}
        """
        sigma = self.sigmas[timestep]
        sigma_next = self.sigmas[timestep + 1]

        # Predict x_0 from noise prediction
        pred_original_sample = sample - sigma * model_output

        # Euler step
        derivative = (sample - pred_original_sample) / sigma
        dt = sigma_next - sigma
        prev_sample = sample + derivative * dt

        return prev_sample


class DDIMScheduler:
    """
    Denoising Diffusion Implicit Models (DDIM) scheduler.

    Supports both deterministic (eta=0) and stochastic (eta>0) sampling.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        clip_sample: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample

        # Create beta schedule
        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            betas = (
                np.linspace(
                    beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32
                )
                ** 2
            )

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()

        self.timesteps = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """Set timesteps for sampling."""
        self.num_inference_steps = num_inference_steps

        # Linearly spaced timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (
            np.arange(0, num_inference_steps) * step_ratio
        ).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        One DDIM sampling step.

        Args:
            model_output: Predicted noise
            timestep: Current timestep
            sample: Current sample x_t
            eta: Stochasticity parameter (0 = deterministic)
            generator: Random generator for reproducibility

        Returns:
            Previous sample x_{t-1}
        """
        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0)
        )

        alpha_prod_t = alpha_prod_t.to(sample.device)
        alpha_prod_t_prev = alpha_prod_t_prev.to(sample.device)

        # Predict x_0
        pred_original_sample = (
            sample - torch.sqrt(1 - alpha_prod_t) * model_output
        ) / torch.sqrt(alpha_prod_t)

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )
        std_dev_t = eta * torch.sqrt(variance)

        # Direction pointing to x_t
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * model_output

        # x_{t-1} = sqrt(alpha_{t-1}) * x_0 + direction + noise
        prev_sample = (
            torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        )

        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn(
                sample.shape, generator=generator, device=sample.device, dtype=sample.dtype
            )
            prev_sample = prev_sample + std_dev_t * noise

        return prev_sample
