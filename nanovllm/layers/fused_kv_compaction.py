"""Fused direct-copy kernel for paged K/V-cache compaction."""

from __future__ import annotations

import os
from pathlib import Path

import torch

os.environ.setdefault("TRITON_CACHE_DIR", str(Path.cwd() / ".triton-cache"))

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only installations
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _kv_cache_compaction_kernel(
        k_cache_ptr,
        v_cache_ptr,
        old_block_table_ptr,
        new_block_table_ptr,
        keep_indices_ptr,
        layer_stride,
        row_size: tl.constexpr,
        cache_block_size: tl.constexpr,
        BLOCK_ELEMENTS: tl.constexpr,
    ):
        output_token = tl.program_id(0)
        layer = tl.program_id(1)
        chunk = tl.program_id(2)

        source_position = tl.load(keep_indices_ptr + output_token)
        source_block = tl.load(old_block_table_ptr + source_position // cache_block_size)
        source_slot = source_block * cache_block_size + source_position % cache_block_size
        destination_block = tl.load(new_block_table_ptr + output_token // cache_block_size)
        destination_slot = destination_block * cache_block_size + output_token % cache_block_size

        columns = chunk * BLOCK_ELEMENTS + tl.arange(0, BLOCK_ELEMENTS)
        mask = columns < row_size
        source_offsets = layer * layer_stride + source_slot * row_size + columns
        destination_offsets = layer * layer_stride + destination_slot * row_size + columns

        # K and V share the same page mapping, so one program moves both and
        # avoids separate gather/scatter launches and temporary tensors.
        keys = tl.load(k_cache_ptr + source_offsets, mask=mask)
        values = tl.load(v_cache_ptr + source_offsets, mask=mask)
        tl.store(k_cache_ptr + destination_offsets, keys, mask=mask)
        tl.store(v_cache_ptr + destination_offsets, values, mask=mask)


def _physical_slots(
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    positions = positions.to(device=block_table.device, dtype=torch.long)
    table = block_table.to(dtype=torch.long)
    return table[positions // block_size] * block_size + positions % block_size


@torch.inference_mode()
def _torch_compact_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    old_block_table: torch.Tensor,
    new_block_table: torch.Tensor,
    keep_indices: torch.Tensor,
    block_size: int,
) -> None:
    """Reference gather/scatter implementation used as the safe fallback."""
    source_slots = _physical_slots(keep_indices, old_block_table, block_size)
    output_positions = torch.arange(keep_indices.numel(), device=new_block_table.device)
    destination_slots = _physical_slots(output_positions, new_block_table, block_size)
    layers = k_cache.shape[0]
    row_size = k_cache.shape[-2] * k_cache.shape[-1]
    # Advanced indexing already materializes an independent tensor, matching
    # the original implementation without adding a second redundant clone.
    keys = k_cache.view(layers, -1, row_size)[:, source_slots]
    values = v_cache.view(layers, -1, row_size)[:, source_slots]
    k_cache.view(layers, -1, row_size)[:, destination_slots] = keys
    v_cache.view(layers, -1, row_size)[:, destination_slots] = values


def can_use_fused_kv_compaction(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    old_block_table: torch.Tensor,
    new_block_table: torch.Tensor,
    keep_indices: torch.Tensor,
    block_size: int,
) -> bool:
    if not TRITON_AVAILABLE or os.environ.get("NANOVLLM_DISABLE_FUSED_KV_COMPACTION") == "1":
        return False
    if not k_cache.is_cuda or not v_cache.is_cuda:
        return False
    if k_cache.ndim != 5 or k_cache.shape != v_cache.shape:
        return False
    if not k_cache.is_contiguous() or not v_cache.is_contiguous():
        return False
    if k_cache.dtype != v_cache.dtype or k_cache.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return False
    if block_size < 1 or k_cache.shape[2] != block_size:
        return False
    if keep_indices.numel() == 0:
        return False
    tensors = (old_block_table, new_block_table, keep_indices)
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        return False
    if any(tensor.device != k_cache.device for tensor in tensors):
        return False
    # Avoid inspecting device values here: ``Tensor.item`` would introduce a
    # host synchronization into the compression hot path.  The scheduler owns
    # these tables and guarantees valid indices.
    required_new_blocks = (keep_indices.numel() + block_size - 1) // block_size
    return old_block_table.numel() > 0 and new_block_table.numel() >= required_new_blocks


@torch.inference_mode()
def compact_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    old_block_table: torch.Tensor,
    new_block_table: torch.Tensor,
    keep_indices: torch.Tensor,
    block_size: int,
) -> None:
    """Copy selected paged KV rows directly into a new block table in place."""
    if not can_use_fused_kv_compaction(
        k_cache, v_cache, old_block_table, new_block_table, keep_indices, block_size
    ):
        _torch_compact_kv_cache(
            k_cache, v_cache, old_block_table, new_block_table, keep_indices, block_size
        )
        return

    row_size = k_cache.shape[-2] * k_cache.shape[-1]
    block_elements = 256
    grid = (
        keep_indices.numel(),
        k_cache.shape[0],
        triton.cdiv(row_size, block_elements),
    )
    _kv_cache_compaction_kernel[grid](
        k_cache,
        v_cache,
        old_block_table,
        new_block_table,
        keep_indices,
        k_cache.stride(0),
        row_size=row_size,
        cache_block_size=block_size,
        BLOCK_ELEMENTS=block_elements,
        num_warps=4,
    )
