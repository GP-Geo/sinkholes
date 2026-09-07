"""Plain U-Net, optionally with channel self-attention at the bottleneck."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parts import CentreCropOutput, DoubleConv, Down, OutConv, Up


class ChannelSelfAttention(nn.Module):
    """Self-attention over channels of the pooled bottleneck.

    Spatial dims are average-pooled away before attention, so this reweights
    channels globally rather than attending over positions. It is the attention
    variant ``--add_attn`` has always trained (the original code assigned three
    candidates in a row and only this last one survived), and its parameter
    names are what those checkpoints contain.
    """

    def __init__(self, in_channels, add_positional_encoding=True):
        super().__init__()
        self.in_channels = in_channels
        self.add_positional_encoding = add_positional_encoding
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        if add_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, in_channels))
        self.output_proj = nn.Linear(in_channels, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        pooled = x.view(B, C, -1).mean(dim=-1)  # (B, C)
        if self.add_positional_encoding:
            pooled = pooled + self.pos_encoding

        Q, K, V = self.query(pooled), self.key(pooled), self.value(pooled)
        attn = torch.bmm(Q.unsqueeze(2), K.unsqueeze(1))  # (B, C, C)
        attn = F.softmax(attn / (self.in_channels**0.5), dim=-1)
        out = torch.bmm(attn, V.unsqueeze(2)).squeeze(-1)
        out = self.output_proj(out)

        out = out.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        return self.gamma * out + x


class UNet(CentreCropOutput, nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, add_attn=False):
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
        self.add_attn = add_attn
        if self.add_attn:
            self.attn = ChannelSelfAttention(in_channels=1024)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.add_attn:
            x5 = self.attn(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # Centre-crop to predict_size when this is a large-context model; a no-op otherwise.
        return self.crop_output(self.outc(x))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
