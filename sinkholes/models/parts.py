"""U-Net building blocks (standard PyTorch-UNet parts).

Checkpoint note: existing checkpoints address these by submodule name and
Sequential index (``inc.double_conv.0.weight`` etc.), so attribute names and
layer ordering here are load-bearing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad so odd input sizes land back on the skip's exact H, W.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CentreCropOutput:
    """Mixin: crop a model's logits to ``predict_size`` before they leave forward.

    ``predict_size`` is a plain attribute, not a constructor parameter, so
    state-dict keys are untouched and every existing checkpoint keeps loading.
    ``None`` (the default) means "predict everything", which is the plain
    200x100 behaviour and is what an old checkpoint gets.

    The crop lives in the model on purpose: eval-scenes, test-patches, predict
    and the attention probe all call ``forward`` and would otherwise each need
    to remember the geometry. ``build_from_checkpoint`` sets this from the
    checkpoint's ``io_geometry``, so none of them can get it wrong.
    """

    predict_size = None

    def crop_output(self, logits):
        ps = getattr(self, "predict_size", None)
        if ps is None:
            return logits
        ph, pw = ps
        h, w = logits.shape[-2:]
        if (h, w) == (ph, pw):
            return logits
        dy, dx = h - ph, w - pw
        if dy < 0 or dx < 0:
            raise ValueError(f"predict_size {ps} is larger than the logits {(h, w)}")
        if dy % 2 or dx % 2:
            raise ValueError(f"logits {(h, w)} minus predict_size {ps} = {(dy, dx)}; "
                             "both must be even for a centred crop")
        return logits[..., dy // 2: dy // 2 + ph, dx // 2: dx // 2 + pw]
