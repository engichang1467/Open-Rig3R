"""Autocast dtype selection.

`torch.amp.autocast` defaults to fp16 on CUDA, whose smallest normal is 6.1e-05.
Anything below that flushes to zero in the backward pass, silently, with no NaN and
no warning - the loss curve just flattens earlier than it should. fp16 is only safe
paired with a GradScaler.

bf16 keeps fp32's 8 exponent bits (smallest normal 1.2e-38) and so needs no scaler,
but it wants Ampere or newer; below sm_80 it is emulated and slow.
"""

import torch

AMPERE = (8, 0)


def select_amp_dtype(device, override=None):
    """Pick the autocast dtype for this device.

    Args:
        device:   torch.device the run is on
        override: optional dtype name from config, e.g. "float16", to force a choice
    Returns:
        torch.dtype
    """
    if override is not None:
        dtype = getattr(torch, override, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"amp_dtype must name a torch dtype, got {override!r}")
        return dtype

    if device.type == "cuda" and torch.cuda.get_device_capability(device) < AMPERE:
        return torch.float16  # pre-Ampere: bf16 is emulated, so pay for a scaler instead

    return torch.bfloat16


def needs_grad_scaler(dtype):
    """Only fp16 underflows badly enough to need loss scaling."""
    return dtype == torch.float16
