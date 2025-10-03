"""
Learning Rate Schedulers

Implements warmup and various LR scheduling strategies.
"""

import torch
from torch.optim import Optimizer
from typing import Optional


class WarmupScheduler:
    """
    Learning rate warmup scheduler.

    Gradually increases learning rate from 0 to target LR over warmup steps.
    This helps stabilize training in the initial phase.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_lr: Optional[float] = None,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of steps for warmup
            base_lr: Base learning rate (if None, uses optimizer's current LR)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr or optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # After warmup, use base LR
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class CosineAnnealingWarmupScheduler:
    """
    Cosine annealing with warmup.

    Warms up LR, then decays using cosine schedule.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        base_lr: Optional[float] = None,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate at end
            base_lr: Base learning rate (if None, uses optimizer's current LR)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)))
            lr = lr.item() if isinstance(lr, torch.Tensor) else lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
