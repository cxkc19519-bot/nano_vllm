"""Triton implementation of Qwen3.5 RMSNorm and fused residual-add RMSNorm."""

from __future__ import annotations

import os
from pathlib import Path

import torch

# Locked-down Windows profiles may not permit Triton to create ``~/.triton``.
# Keep the default cache in the project while respecting an explicit override.
os.environ.setdefault("TRITON_CACHE_DIR", str(Path.cwd() / ".triton-cache"))

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in CPU-only installs
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _qwen3_5_add_rmsnorm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        output_ptr,
        residual_output_ptr,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        row_offsets = row * hidden_size + offsets

        # Qwen3.5 computes the residual accumulation and RMS reduction in
        # FP32, even when model activations are BF16/FP16.
        values = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
            values += residual
            tl.store(residual_output_ptr + row_offsets, values, mask=mask)

        variance = tl.sum(values * values, axis=0) / hidden_size
        inverse_rms = tl.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        normalized = values * inverse_rms * (1.0 + weight)
        tl.store(output_ptr + row_offsets, normalized, mask=mask)


def _torch_qwen3_5_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Numerically aligned fallback used on CPU and unsupported layouts."""
    input_dtype = x.dtype
    values = x.float()
    if residual is not None:
        values = values + residual.float()
        residual_output = values.to(dtype=residual.dtype)
    values = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + eps)
    output = (values * (1.0 + weight.float())).to(input_dtype)
    return (output, residual_output) if residual is not None else output


def can_use_fused_add_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> bool:
    """Return whether the Triton kernel supports this tensor layout."""
    if not TRITON_AVAILABLE or os.environ.get("NANOVLLM_DISABLE_FUSED_RMSNORM") == "1":
        return False
    if not x.is_cuda or not x.is_contiguous() or x.ndim < 1:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    hidden_size = x.shape[-1]
    if hidden_size == 0 or hidden_size > 65536:
        return False
    if weight.ndim != 1 or weight.numel() != hidden_size or not weight.is_contiguous():
        return False
    if weight.device != x.device:
        return False
    if residual is not None:
        if residual.shape != x.shape or residual.device != x.device:
            return False
        if residual.dtype != x.dtype or not residual.is_contiguous():
            return False
    return True


def qwen3_5_add_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3.5 ``(1 + weight)`` RMSNorm, fusing residual add when present."""
    if not can_use_fused_add_rmsnorm(x, weight, residual):
        return _torch_qwen3_5_rmsnorm(x, weight, eps, residual)

    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size
    output = torch.empty_like(x)
    residual_output = torch.empty_like(residual) if residual is not None else output
    block_size = triton.next_power_of_2(hidden_size)
    num_warps = 4 if block_size <= 2048 else 8
    _qwen3_5_add_rmsnorm_kernel[(rows,)](
        x,
        residual if residual is not None else x,
        weight,
        output,
        residual_output,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=block_size,
        HAS_RESIDUAL=residual is not None,
        num_warps=num_warps,
    )
    return (output, residual_output) if residual is not None else output
