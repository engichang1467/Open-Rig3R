# tests/test_amp.py
"""Autocast dtype selection: bf16 where it is native, fp16 + scaler where it is not."""
import sys
from pathlib import Path
from unittest.mock import patch

import torch

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from utils.amp import needs_grad_scaler, select_amp_dtype


def with_capability(capability):
    return patch("torch.cuda.get_device_capability", return_value=capability)


def test_ampere_and_newer_get_bfloat16():
    """sm_80+ runs bf16 natively, so no scaler is needed."""
    cuda = torch.device("cuda")
    for capability in [(8, 0), (8, 6), (9, 0), (12, 0)]:  # A100, RTX 30xx, H100, RTX 50xx
        with with_capability(capability):
            dtype = select_amp_dtype(cuda)
        assert dtype == torch.bfloat16, f"{capability} got {dtype}"
        assert not needs_grad_scaler(dtype)
    print("Ampere and newer select bf16 test passed!")


def test_pre_ampere_falls_back_to_scaled_fp16():
    """Below sm_80 bf16 is emulated, so pay for fp16 plus a scaler instead."""
    cuda = torch.device("cuda")
    for capability in [(7, 0), (7, 5)]:  # V100, T4 / RTX 20xx
        with with_capability(capability):
            dtype = select_amp_dtype(cuda)
        assert dtype == torch.float16, f"{capability} got {dtype}"
        assert needs_grad_scaler(dtype), "fp16 without a scaler is the bug being fixed"
    print("Pre-Ampere falls back to scaled fp16 test passed!")


def test_cpu_never_probes_cuda():
    """A CPU run must not call into the CUDA API to pick a dtype."""
    def explode(*args, **kwargs):
        raise AssertionError("queried CUDA capability on a CPU device")

    with patch("torch.cuda.get_device_capability", explode):
        dtype = select_amp_dtype(torch.device("cpu"))

    assert dtype == torch.bfloat16, dtype
    print("CPU selection does not touch CUDA test passed!")


def test_config_override_wins():
    """amp_dtype in the config forces the choice, e.g. to reproduce an old run."""
    cuda = torch.device("cuda")
    with with_capability((12, 0)):
        assert select_amp_dtype(cuda, "float16") == torch.float16
        assert select_amp_dtype(cuda, "float32") == torch.float32

    try:
        select_amp_dtype(cuda, "not_a_dtype")
    except ValueError:
        pass
    else:
        raise AssertionError("a bogus amp_dtype should not be accepted silently")
    print("Config override wins test passed!")


def test_bfloat16_holds_gradients_fp16_would_lose():
    """The reason for the whole change.

    fp16's smallest normal is 6.1e-05; below that it degrades into subnormals with
    shrinking precision, and under ~6e-08 it is zero outright. bf16 keeps fp32's
    exponent range, so the same values stay representable. Gradients that vanish do
    so silently - no NaN, no warning, just a loss curve that flattens early.
    """
    fp16, bf16 = torch.finfo(torch.float16), torch.finfo(torch.bfloat16)
    assert bf16.tiny < fp16.tiny / 1e30, "bf16 should have vastly more headroom"

    # below fp16's smallest subnormal, so it is gone entirely
    flushed = torch.tensor(1e-8)
    assert flushed.to(torch.float16) == 0, "expected fp16 to flush this to zero"
    assert flushed.to(torch.bfloat16) != 0, "bf16 must keep it"

    # inside fp16's subnormal range it survives, but precision is already degraded
    subnormal = torch.tensor(1e-7)
    assert subnormal.to(torch.float16) != 0
    fp16_error = (subnormal.to(torch.float16).float() - subnormal).abs() / subnormal
    bf16_error = (subnormal.to(torch.bfloat16).float() - subnormal).abs() / subnormal
    assert fp16_error > bf16_error, "fp16 subnormal should be the coarser of the two"

    print(f"fp16 min normal {fp16.tiny:.3e} (subnormals to {fp16.smallest_normal / 1024:.3e}), "
          f"bf16 min normal {bf16.tiny:.3e}")
    print(f"at 1e-7: fp16 rel error {fp16_error:.1%}, bf16 {bf16_error:.1%} test passed!")


if __name__ == "__main__":
    test_ampere_and_newer_get_bfloat16()
    test_pre_ampere_falls_back_to_scaled_fp16()
    test_cpu_never_probes_cuda()
    test_config_override_wins()
    test_bfloat16_holds_gradients_fp16_would_lose()
