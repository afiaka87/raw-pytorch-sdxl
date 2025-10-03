"""
Exponential Moving Average (EMA) for model weights.

Maintains a shadow copy of model parameters that are updated with exponential decay.
EMA models typically produce better quality outputs than the raw trained model.
"""

import torch
import torch.nn as nn
from typing import Optional
from copy import deepcopy


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Usage:
        model = MyModel()
        ema = EMAModel(model, decay=0.9999)

        # Training loop
        for batch in dataloader:
            loss = train_step(model, batch)
            loss.backward()
            optimizer.step()
            ema.update(model)  # Update EMA after optimizer step

        # Use EMA for evaluation
        ema.apply_shadow()  # Temporarily replace model weights with EMA
        evaluate(model)
        ema.restore()  # Restore original weights
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Model to track
            decay: EMA decay rate (0.9999 for SDXL)
            device: Device to store shadow parameters
        """
        self.decay = decay
        self.device = device if device is not None else torch.device("cpu")

        # Create shadow parameters
        self.shadow_params = {}
        self.collected_params = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().to(self.device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA parameters.

        EMA update: shadow = decay * shadow + (1 - decay) * param
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow_params
                new_average = (
                    self.decay * self.shadow_params[name]
                    + (1.0 - self.decay) * param.data.to(self.device)
                )
                self.shadow_params[name] = new_average

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        """
        Replace model parameters with EMA shadow parameters.

        Call restore() to revert back to original parameters.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow_params
                self.collected_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name].to(param.device))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """Restore original model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.collected_params
                param.data.copy_(self.collected_params[name])
        self.collected_params = {}

    def state_dict(self):
        """Return EMA state dict for checkpointing."""
        return {
            "decay": self.decay,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow_params = state_dict["shadow_params"]

    def to(self, device: torch.device):
        """Move EMA parameters to device."""
        self.device = device
        for name in self.shadow_params:
            self.shadow_params[name] = self.shadow_params[name].to(device)
        return self


class SimpleEMA:
    """
    Simplified EMA that updates in-place (compatible with GLIDE finetune pattern).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        self.shadow.requires_grad_(False)

        # Move to same device as model
        device = next(model.parameters()).device
        self.shadow.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow model with EMA."""
        for ema_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            if model_param.requires_grad:
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1.0 - self.decay
                )

    def state_dict(self):
        """Return shadow model state dict."""
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        """Load shadow model state."""
        self.shadow.load_state_dict(state_dict)

    def to(self, device: torch.device):
        """Move shadow model to device."""
        self.shadow.to(device)
        return self
