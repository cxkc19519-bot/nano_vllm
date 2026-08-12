import os
import unittest

import torch

from nanovllm.layers.fused_add_rmsnorm import (
    TRITON_AVAILABLE,
    _torch_qwen3_5_rmsnorm,
    can_use_fused_add_rmsnorm,
    qwen3_5_add_rmsnorm,
)


class FusedAddRMSNormTest(unittest.TestCase):
    def test_cpu_fallback_matches_reference_without_residual(self):
        torch.manual_seed(7)
        x = torch.randn(3, 16, dtype=torch.float32)
        weight = torch.randn(16, dtype=torch.float32)
        expected = _torch_qwen3_5_rmsnorm(x, weight, 1e-6, None)
        actual = qwen3_5_add_rmsnorm(x, weight, 1e-6)
        torch.testing.assert_close(actual, expected)

    def test_cpu_fallback_matches_reference_with_residual(self):
        torch.manual_seed(11)
        x = torch.randn(4, 32, dtype=torch.float32)
        residual = torch.randn_like(x)
        weight = torch.randn(32, dtype=torch.float32)
        expected_output, expected_residual = _torch_qwen3_5_rmsnorm(x, weight, 1e-6, residual)
        actual_output, actual_residual = qwen3_5_add_rmsnorm(x, weight, 1e-6, residual)
        torch.testing.assert_close(actual_output, expected_output)
        torch.testing.assert_close(actual_residual, expected_residual)

    def test_environment_flag_disables_kernel(self):
        previous = os.environ.get("NANOVLLM_DISABLE_FUSED_RMSNORM")
        os.environ["NANOVLLM_DISABLE_FUSED_RMSNORM"] = "1"
        try:
            self.assertFalse(can_use_fused_add_rmsnorm(torch.ones(2), torch.ones(2)))
        finally:
            if previous is None:
                os.environ.pop("NANOVLLM_DISABLE_FUSED_RMSNORM", None)
            else:
                os.environ["NANOVLLM_DISABLE_FUSED_RMSNORM"] = previous

    @unittest.skipUnless(torch.cuda.is_available() and TRITON_AVAILABLE, "requires CUDA and Triton")
    def test_cuda_bfloat16_matches_reference_with_residual(self):
        torch.manual_seed(19)
        x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
        expected_output, expected_residual = _torch_qwen3_5_rmsnorm(x, weight, 1e-6, residual)
        actual_output, actual_residual = qwen3_5_add_rmsnorm(x, weight, 1e-6, residual)
        torch.testing.assert_close(actual_output, expected_output, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() and TRITON_AVAILABLE, "requires CUDA and Triton")
    def test_cuda_bfloat16_matches_reference_without_residual(self):
        torch.manual_seed(23)
        # Q/K RMSNorm uses a three-dimensional [tokens, heads, head_dim]
        # tensor and exercises the no-residual specialization of the kernel.
        x = torch.randn(64, 8, 256, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        expected = _torch_qwen3_5_rmsnorm(x, weight, 1e-6, None)
        actual = qwen3_5_add_rmsnorm(x, weight, 1e-6)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
