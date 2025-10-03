# ADVANCED: Implementation Details and Lessons Learned

## Motivation

This project was born from a desire to understand and control the Stable Diffusion XL training process at the lowest level possible. The goal was to implement fine-tuning in pure PyTorch without relying on high-level abstractions from libraries like diffusers, peft, or accelerate. This would provide:

1. Complete visibility into the training process
2. Maximum flexibility for customization
3. Better understanding of memory usage and optimization opportunities
4. Educational value for those wanting to understand SDXL internals

## Implementation Journey

### Starting Point

We began with the ambitious goal of training full SDXL (2.6B parameters) on consumer GPUs with <12GB VRAM. The reference implementation from minSDXL provided a clean, minimal SDXL architecture without the complexity of the diffusers library.

### Major Challenges and Solutions

#### 1. Memory Crisis: The 12GB VRAM Wall

**Problem**: Full SDXL at 1024x1024 resolution consumed >11GB just for model weights, leaving no room for gradients or optimizer states.

**Initial Attempts**:
- BF16 precision: Saved ~30% memory but still insufficient
- LoRA implementation: Reduced trainable parameters from 2.6B to 4.5M
- Still hit OOM during attention computation

**Solution Evolution**:
1. First tried gradient checkpointing - helped but not enough
2. Finally had to reduce resolution to 512x512
3. Combination of BF16 + LoRA + gradient checkpointing + 512x512 finally fit in ~7.5GB

**Lesson**: SDXL's architecture is fundamentally designed for 16GB+ GPUs. Getting it to work on 12GB required significant compromises.

#### 2. LoRA Implementation Bugs

**Problem 1**: Initial LoRA implementation showed 100% of parameters as trainable.

**Cause**: We were freezing the base model AFTER applying LoRA, which also froze the LoRA parameters.

**Solution**: Freeze base model first, then apply LoRA, then explicitly unfreeze LoRA parameters.

**Problem 2**: LoRA parameters stayed on CPU while base model was on CUDA.

**Cause**: LoRA layers were created before moving to device.

**Solution**: Move LoRA layers to same device/dtype as base layer during initialization.

**Problem 3**: Module name matching failed - no LoRA layers were being created.

**Cause**: The recursive function wasn't building full module paths correctly.

**Solution**: Track the full module path through recursion with a prefix parameter.

#### 3. VAE Architecture Mismatch

**Problem**: Our custom VAE implementation had different layer names than the pretrained weights.

**Initial Approach**: Try to create a compatible VAE from scratch.

**Reality Check**: The VAE is frozen during training anyway - no need to reimplement.

**Solution**: Use diffusers' AutoencoderKL just for the VAE, keeping everything else pure PyTorch.

**Lesson**: Pragmatism beats purity when the component isn't being trained.

#### 4. Dtype Inconsistencies

**Problem**: Mixed dtype errors between FP32 timestep embeddings and BF16 model weights.

**Solution**: Ensure all tensors are converted to model dtype in forward passes.

**Lesson**: PyTorch's automatic mixed precision doesn't handle everything - manual dtype management is often needed.

## Principles We Had to Abandon

### 1. "No Diffusers Dependencies"

**Original Goal**: Zero dependencies on diffusers library.

**Reality**: We use diffusers for:
- VAE loading (frozen, not trained)
- Tokenizer configs

**Justification**: These components are peripheral to the training loop. The core training logic remains pure PyTorch.

### 2. "Full 1024x1024 Resolution"

**Original Goal**: Match SDXL's native 1024x1024 resolution.

**Reality**: 512x512 is the maximum that fits in 12GB VRAM with our setup.

**Impact**: Lower resolution training may affect quality, but it's a necessary trade-off for accessibility.

### 3. "EMA by Default"

**Original Goal**: Use Exponential Moving Average for better generation quality.

**Reality**: EMA requires a full copy of the model, doubling VRAM requirements.

**Solution**: Made EMA optional (disabled by default).

## Technical Insights

### Memory Breakdown (512x512, LoRA rank 4)

```
Model weights (BF16):          ~5.2GB
LoRA parameters:                ~9MB
Optimizer state (AdamW):        ~18MB
Gradients (LoRA only):          ~9MB
Activations (checkpointed):    ~2.0GB
Temporary buffers:             ~0.3GB
-----------------------------------
Total:                         ~7.5GB
```

### Why Gradient Checkpointing Wasn't Enough

Gradient checkpointing trades compute for memory by not storing intermediate activations. However, SDXL's attention mechanisms still require large temporary matrices:
- Q×K attention scores: [batch, heads, seq_len, seq_len]
- At 1024x1024: [1, 8, 4096, 4096] × 2 bytes = 256MB per attention layer
- With 70+ attention layers, this quickly exhausts memory

### The LoRA Sweet Spot

LoRA rank 4 with alpha 16 proved optimal:
- Only 4.5M trainable parameters (0.17% of model)
- ~9MB of LoRA weights
- Sufficient expressiveness for fine-tuning
- Minimal memory overhead

Higher ranks (8, 16) provided diminishing returns while significantly increasing memory usage.

## What Works Well

1. **Custom LoRA implementation**: Clean, hackable, and memory-efficient
2. **BF16 training**: More stable than FP16, no loss scaling needed
3. **Gradient checkpointing wrapper**: Simple but effective memory saver
4. **Mixed approach**: Pure PyTorch where it matters, pragmatic elsewhere

## What Could Be Better

1. **Flash Attention**: Could reduce memory usage further
2. **CPU offloading**: Could enable 1024x1024 on 12GB GPUs
3. **8-bit optimization**: Could reduce model size by another 50%
4. **Distributed training**: Currently single-GPU only

## Recommendations for Users

### For <12GB VRAM
- Use 512x512 resolution
- LoRA rank 4 or lower
- Disable EMA
- BF16 precision
- Batch size 1 with gradient accumulation

### For 16GB VRAM
- Can use 768x768 resolution
- LoRA rank 8 viable
- Consider enabling EMA
- Batch size 2-4 possible

### For 24GB+ VRAM
- Full 1024x1024 resolution
- LoRA rank 16 or full fine-tuning
- EMA recommended
- Larger batch sizes

## Final Thoughts

This implementation proves that SDXL fine-tuning is possible on consumer hardware, but with significant trade-offs. The journey revealed that while pure implementations are educational and provide control, pragmatic compromises are often necessary for practical use.

The key insight: Memory optimization is not just about reducing model size, but understanding where every byte goes during training. Even with aggressive optimizations, SDXL pushes the boundaries of what's possible on consumer GPUs.

For production use, consider:
- Using this implementation to understand the process
- Moving to cloud GPUs with more VRAM for serious training
- Using diffusers/accelerate for production (they've solved many of these problems)

The code remains valuable as a learning tool and for scenarios where maximum control is needed.
