"""Attention U-Net: attention gates on every skip connection, dropout in the convs.

Self-contained on purpose — its ``DoubleConv`` differs from the plain U-Net's
(an extra ``Dropout2d`` between the two convolutions), which shifts the
Sequential indices its checkpoints use (``double_conv.4/5`` instead of
``3/4``). That index shift is also how the checkpoint factory tells the two
architectures apart, so do not "deduplicate" these blocks into parts.py.
"""

from .parts import CentreCropOutput
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with Dropout2d after the first ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class AttentionGate(nn.Module):
    """Gate a skip connection by a mask computed from skip + gating signal."""

    def __init__(self, in_channels, gating_channels, reduction_ratio=0.5):
        super().__init__()
        reduced = max(1, int(in_channels * reduction_ratio))
        self.W_x = nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False)
        self.W_g = nn.Conv2d(gating_channels, reduced, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(reduced, 1, kernel_size=1, bias=False, groups=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=True)
        psi = self.sigmoid(self.psi(self.relu(x1 + g1)))
        return x * psi


class Up(nn.Module):
    """Upscale, attention-gate the skip, concatenate, double conv."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.attention = AttentionGate(out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.attention(x2, x1)
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


class AttentionUNet(CentreCropOutput, nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # Centre-crop to predict_size when this is a large-context model; a no-op otherwise.
        return self.crop_output(self.outc(x))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
