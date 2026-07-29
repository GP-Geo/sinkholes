"""Device and memory-format selection shared by training and inference."""

import torch


def get_device() -> torch.device:
    """Best available device: CUDA, then Apple Silicon (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def memory_format_for(device) -> torch.memory_format:
    """channels_last where it helps (CUDA convnets), default layout on MPS.

    MPS cannot autograd through channels_last — the backward pass raises
    "view size is not compatible with input tensor's size and stride".
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    return torch.contiguous_format if device.type == "mps" else torch.channels_last
