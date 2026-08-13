import os
import unittest

import torch

from nanovllm.layers.fused_kv_compaction import (
    TRITON_AVAILABLE,
    _torch_compact_kv_cache,
    can_use_fused_kv_compaction,
    compact_kv_cache,
)


def make_case(device: str, dtype: torch.dtype = torch.float32):
    torch.manual_seed(31)
    layers, blocks, block_size, heads, head_dim = 3, 8, 4, 2, 8
    k_cache = torch.randn(layers, blocks, block_size, heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    old_blocks = torch.tensor([1, 3, 5, 6], device=device, dtype=torch.int32)
    new_blocks = torch.tensor([0, 2, 4], device=device, dtype=torch.int32)
    keep = torch.tensor([0, 2, 5, 7, 8, 11, 13, 15, 3], device=device, dtype=torch.int32)
    return k_cache, v_cache, old_blocks, new_blocks, keep, block_size


class FusedKVCompactionTest(unittest.TestCase):
    def test_cpu_fallback_matches_reference(self):
        case = make_case("cpu")
        expected_k, expected_v = case[0].clone(), case[1].clone()
        actual_k, actual_v = case[0].clone(), case[1].clone()
        _torch_compact_kv_cache(expected_k, expected_v, *case[2:])
        compact_kv_cache(actual_k, actual_v, *case[2:])
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)

    def test_environment_flag_disables_kernel(self):
        case = make_case("cuda" if torch.cuda.is_available() else "cpu")
        previous = os.environ.get("NANOVLLM_DISABLE_FUSED_KV_COMPACTION")
        os.environ["NANOVLLM_DISABLE_FUSED_KV_COMPACTION"] = "1"
        try:
            self.assertFalse(can_use_fused_kv_compaction(*case))
        finally:
            if previous is None:
                os.environ.pop("NANOVLLM_DISABLE_FUSED_KV_COMPACTION", None)
            else:
                os.environ["NANOVLLM_DISABLE_FUSED_KV_COMPACTION"] = previous

    @unittest.skipUnless(torch.cuda.is_available() and TRITON_AVAILABLE, "requires CUDA and Triton")
    def test_cuda_bfloat16_is_bit_exact(self):
        case = make_case("cuda", torch.bfloat16)
        expected_k, expected_v = case[0].clone(), case[1].clone()
        actual_k, actual_v = case[0].clone(), case[1].clone()
        _torch_compact_kv_cache(expected_k, expected_v, *case[2:])
        compact_kv_cache(actual_k, actual_v, *case[2:])
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
