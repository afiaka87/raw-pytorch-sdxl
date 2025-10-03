"""
LoRA (Low-Rank Adaptation) for SDXL in Pure PyTorch

Implements parameter-efficient fine-tuning by adding trainable low-rank
decomposition matrices to linear layers while keeping base weights frozen.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import re


class LoRALinear(nn.Module):
    """
    LoRA-augmented Linear layer.

    Decomposes weight updates into low-rank matrices A and B:
        h = Wx + (B @ A)x * (alpha / rank)

    where W is frozen, and A, B are trainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA contribution: x @ A @ B * scaling"""
        x = self.lora_dropout(x)
        result = x @ self.lora_A @ self.lora_B * self.scaling
        return result


def replace_linear_with_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 32.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    prefix: str = "",
) -> nn.Module:
    """
    Replace Linear layers with LoRA-augmented versions.

    Args:
        module: PyTorch module to modify
        rank: Rank of LoRA decomposition
        alpha: LoRA scaling parameter
        dropout: Dropout probability for LoRA layers
        target_modules: List of module name patterns to target (regex)
        prefix: Current module path prefix (used internally)

    Returns:
        Modified module with LoRA layers
    """
    if target_modules is None:
        # Default: target attention Q, K, V projections
        target_modules = [r".*\.to_q$", r".*\.to_k$", r".*\.to_v$"]

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            # Check if this module matches target patterns
            if any(re.match(pattern, full_name) for pattern in target_modules):
                # Create LoRA-augmented layer
                lora_linear = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )

                # Wrap the original linear layer
                setattr(
                    module,
                    name,
                    LoRALinearWrapper(child, lora_linear),
                )
        else:
            # Recursively apply to children
            replace_linear_with_lora(
                child, rank=rank, alpha=alpha, dropout=dropout,
                target_modules=target_modules, prefix=full_name
            )

    return module


class LoRALinearWrapper(nn.Module):
    """
    Wrapper that combines a frozen linear layer with a LoRA layer.
    """

    def __init__(self, base_layer: nn.Linear, lora_layer: LoRALinear):
        super().__init__()
        self.base_layer = base_layer
        self.lora_layer = lora_layer

        # Move LoRA to same device and dtype as base layer
        self.lora_layer = self.lora_layer.to(
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype
        )

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Expose attributes for compatibility
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.bias = base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base + LoRA"""
        base_out = self.base_layer(x)
        lora_out = self.lora_layer(x)
        return base_out + lora_out


def get_full_name(module: nn.Module, name: str, prefix: str = "") -> str:
    """Helper to get full module name with hierarchy."""
    return f"{prefix}.{name}" if prefix else name


def find_lora_modules(model: nn.Module) -> List[str]:
    """
    Find all LoRA-augmented modules in the model.

    Returns:
        List of module names that have LoRA
    """
    lora_modules = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinearWrapper):
            lora_modules.append(name)
    return lora_modules


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA parameters from model.

    Returns:
        State dict containing only LoRA weights
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            lora_state_dict[name] = param.detach().cpu()
    return lora_state_dict


def load_lora_state_dict(
    model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = True
):
    """
    Load LoRA parameters into model.

    Args:
        model: Model with LoRA layers
        state_dict: State dict containing LoRA weights
        strict: Whether to strictly enforce key matching
    """
    # Filter to only LoRA parameters
    lora_keys = {k for k in state_dict.keys() if "lora_" in k}

    if strict:
        model_lora_keys = {k for k, _ in model.named_parameters() if "lora_" in k}
        missing = model_lora_keys - lora_keys
        unexpected = lora_keys - model_lora_keys

        if missing:
            raise ValueError(f"Missing LoRA keys: {missing}")
        if unexpected:
            raise ValueError(f"Unexpected LoRA keys: {unexpected}")

    # Load parameters
    model_state = model.state_dict()
    for key in lora_keys:
        if key in model_state:
            model_state[key].copy_(state_dict[key])


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into base model for inference.

    This creates a new model without LoRA layers by merging
    the low-rank updates into the base weights.

    Args:
        model: Model with LoRA layers

    Returns:
        New model with merged weights (no LoRA)
    """
    merged_model = type(model)()  # Create new instance
    merged_state = merged_model.state_dict()

    for name, module in model.named_modules():
        if isinstance(module, LoRALinearWrapper):
            # Compute merged weight: W + B @ A * scaling
            base_weight = module.base_layer.weight.data
            lora_A = module.lora_layer.lora_A.data
            lora_B = module.lora_layer.lora_B.data
            scaling = module.lora_layer.scaling

            merged_weight = base_weight + (lora_B @ lora_A).t() * scaling

            # Update merged model
            merged_state[f"{name}.weight"] = merged_weight
            if module.base_layer.bias is not None:
                merged_state[f"{name}.bias"] = module.base_layer.bias.data

    merged_model.load_state_dict(merged_state, strict=False)
    return merged_model


def count_lora_parameters(model: nn.Module) -> tuple[int, int, float]:
    """
    Count LoRA parameters vs total parameters.

    Returns:
        (trainable_params, total_params, trainable_percentage)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100.0 * trainable / total if total > 0 else 0.0

    return trainable, total, percentage


def apply_lora_to_unet(
    unet: nn.Module,
    rank: int = 8,
    alpha: float = 32.0,
    dropout: float = 0.0,
    target_mode: str = "attention",
) -> nn.Module:
    """
    Apply LoRA to SDXL UNet.

    Args:
        unet: UNet2DConditionModel instance
        rank: LoRA rank (4-16 typical)
        alpha: LoRA alpha (typically 2x-4x rank)
        dropout: LoRA dropout probability
        target_mode: Which modules to target:
            - "attention": Q, K, V in self and cross-attention (recommended)
            - "attention_out": Add output projections
            - "all": All linear layers (most parameters, slowest)

    Returns:
        UNet with LoRA applied
    """
    # First, freeze ALL parameters in the UNet
    for param in unet.parameters():
        param.requires_grad = False

    if target_mode == "attention":
        target_modules = [
            r".*\.attn1\.to_q$",
            r".*\.attn1\.to_k$",
            r".*\.attn1\.to_v$",
            r".*\.attn2\.to_q$",
            r".*\.attn2\.to_k$",
            r".*\.attn2\.to_v$",
        ]
    elif target_mode == "attention_out":
        target_modules = [
            r".*\.attn1\.to_q$",
            r".*\.attn1\.to_k$",
            r".*\.attn1\.to_v$",
            r".*\.attn1\.to_out\.0$",  # Output projection
            r".*\.attn2\.to_q$",
            r".*\.attn2\.to_k$",
            r".*\.attn2\.to_v$",
            r".*\.attn2\.to_out\.0$",
        ]
    elif target_mode == "all":
        target_modules = [r".*"]  # Match all
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    unet = replace_linear_with_lora(
        unet,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )

    # Ensure LoRA parameters are trainable
    for name, param in unet.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    # Print statistics
    trainable, total, percentage = count_lora_parameters(unet)
    print(f"LoRA Statistics:")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Total params: {total:,}")
    print(f"  Trainable %: {percentage:.2f}%")

    return unet
