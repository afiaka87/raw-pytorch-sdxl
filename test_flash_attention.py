#!/usr/bin/env python3
"""
Test Flash Attention implementation.

Verifies that Flash Attention produces similar outputs to standard attention.
"""

import torch
import torch.nn.functional as F
from sdxl.unet import Attention

def test_attention_equivalence():
    """Test that Flash Attention produces similar results to standard attention."""
    print("Testing Flash Attention equivalence...")

    # Setup
    batch_size = 2
    seq_len = 64
    inner_dim = 320
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Create test input
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, inner_dim, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(batch_size, 77, 2048, device=device, dtype=dtype)

    # Create two attention modules with same weights
    print(f"\nCreating attention modules on {device}...")
    attn_standard = Attention(inner_dim, cross_attention_dim=2048, use_flash_attention=False).to(device, dtype)
    attn_flash = Attention(inner_dim, cross_attention_dim=2048, use_flash_attention=True).to(device, dtype)

    # Copy weights to ensure they're identical
    attn_flash.load_state_dict(attn_standard.state_dict())

    # Run both
    print("Running standard attention...")
    with torch.no_grad():
        output_standard = attn_standard(hidden_states, encoder_hidden_states)

    print("Running flash attention...")
    with torch.no_grad():
        output_flash = attn_flash(hidden_states, encoder_hidden_states)

    # Compare outputs
    max_diff = (output_standard - output_flash).abs().max().item()
    mean_diff = (output_standard - output_flash).abs().mean().item()

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Output shape: {output_standard.shape}")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

    # Check if outputs are close enough (allowing for numerical differences)
    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"✓ PASSED: Outputs are equivalent (within {tolerance})")
        return True
    else:
        print(f"✗ FAILED: Outputs differ by more than {tolerance}")
        return False

def test_memory_usage():
    """Test that Flash Attention uses less memory."""
    if not torch.cuda.is_available():
        print("\nSkipping memory test (CUDA not available)")
        return

    print("\n" + "="*60)
    print("Testing memory usage...")
    print("="*60)

    batch_size = 4
    seq_len = 256  # Larger sequence to show memory difference
    inner_dim = 1280
    device = "cuda"
    dtype = torch.bfloat16

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Test standard attention
    attn_standard = Attention(inner_dim, use_flash_attention=False).to(device, dtype)
    hidden_states = torch.randn(batch_size, seq_len, inner_dim, device=device, dtype=dtype)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = attn_standard(hidden_states)
    mem_standard = torch.cuda.max_memory_allocated() / 1024**2  # MB

    del attn_standard
    torch.cuda.empty_cache()

    # Test flash attention
    attn_flash = Attention(inner_dim, use_flash_attention=True).to(device, dtype)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = attn_flash(hidden_states)
    mem_flash = torch.cuda.max_memory_allocated() / 1024**2  # MB

    print(f"Standard attention peak memory: {mem_standard:.2f} MB")
    print(f"Flash attention peak memory: {mem_flash:.2f} MB")
    print(f"Memory reduction: {(1 - mem_flash/mem_standard)*100:.1f}%")

if __name__ == "__main__":
    print("Flash Attention Test Suite")
    print("="*60)

    # Test equivalence
    success = test_attention_equivalence()

    # Test memory
    test_memory_usage()

    print("\n" + "="*60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*60)
