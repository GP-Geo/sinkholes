"""
Device and memory-format selection, shared by the training and inference scripts.

Added 2026-07-27 so the project can use Apple Silicon GPUs. See CHANGELOG.md.

Background — what `channels_last` is
------------------------------------
A 4-D tensor (N, C, H, W) has a logical shape and a separate physical layout in
memory. The PyTorch default, `contiguous_format` (NCHW), stores one channel
plane at a time: all of channel 0, then all of channel 1, ... `channels_last`
(NHWC) instead stores all channels of one pixel adjacently. The logical shape,
indexing and results are identical; only the strides differ.

NHWC is what cuDNN's tensor-core convolution kernels want, so on modern NVIDIA
GPUs `.to(memory_format=torch.channels_last)` can be a large speedup for convnets
— which is why this repo applies it to both the model and every input batch.

It is not free everywhere. On CPU it made no measurable difference here
(107 vs 109 ms/step), and on MPS the backward pass raises

    RuntimeError: view size is not compatible with input tensor's size and
    stride (at least one dimension spans across two contiguous subspaces).

`memory_format_for()` therefore keeps NHWC on CUDA and CPU and falls back to the
default layout on MPS. Inference with NHWC inputs does work on MPS, so this only
matters for training.
"""
import torch


def get_device():
    """Best available device: CUDA, then Apple Silicon (MPS), then CPU.

    Previously every script hardcoded `'cuda' if torch.cuda.is_available() else 'cpu'`,
    so Macs always fell back to CPU. Measured on this UNet at 200x100, batch 1:
    CPU 107 ms/step vs MPS 14 ms/step.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def memory_format_for(device):
    """`channels_last` where it is supported and useful, default layout on MPS."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    return torch.contiguous_format if device.type == 'mps' else torch.channels_last
