# --- path bootstrap: flat imports from any src/ subfolder. EDIT 2026-07-29, CHANGELOG.md #10 ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401,E402
# --- end bootstrap ---
import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_parts import DoubleConv, Down, Up, OutConv

# This module applies multi-head self attention to 2D feature maps.
# 
class MultiHeadSelfAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        Q = self.query(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # B x heads x HW x head_dim
        K = self.key(x).view(B, self.num_heads, self.head_dim, H * W)  # B x heads x head_dim x HW
        V = self.value(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # B x heads x HW x head_dim

        attn = torch.matmul(Q, K) / (self.head_dim ** 0.5)  # Scaled dot-product attention
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)  # B x heads x HW x head_dim
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)  # Merge heads back

        out = self.output_proj(out)
        return self.gamma * out + x  # Residual connection


import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSelfAttention(nn.Module):
    def __init__(self, in_channels, add_positional_encoding=True):
        super().__init__()
        self.in_channels = in_channels
        self.add_positional_encoding = add_positional_encoding

        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

        if add_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, in_channels))  # [1, C]

        self.output_proj = nn.Linear(in_channels, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        # Reshape to [B, C, N] where N = H*W (each channel is a token)
        x_flat = x.view(B, C, -1)  # [B, C, N]

        # Compute global average across spatial dims → [B, C]
        x_pooled = x_flat.mean(dim=-1)  # [B, C]

        if self.add_positional_encoding:
            x_pooled = x_pooled + self.pos_encoding  # Add positional encoding over channels

        # Linear projections across channels
        Q = self.query(x_pooled)  # [B, C]
        K = self.key(x_pooled)  # [B, C]
        V = self.value(x_pooled)  # [B, C]

        # Compute attention across channels
        attn = torch.bmm(Q.unsqueeze(2), K.unsqueeze(1))  # [B, C, C]
        attn = F.softmax(attn / (self.in_channels ** 0.5), dim=-1)

        out = torch.bmm(attn, V.unsqueeze(2)).squeeze(-1)  # [B, C]
        out = self.output_proj(out)  # [B, C]

        # Expand back to [B, C, H, W]
        out = out.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        return self.gamma * out + x  # Residual connection


class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        Q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        K = self.key(x).view(B, -1, H * W)
        V = self.value(x).view(B, -1, H * W)

        attn = torch.bmm(Q, K)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(V, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

class HistoryGate(nn.Module):
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps
        self.gate_head = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        # start conservative: small gate -> rely on current channel mostly
        with torch.no_grad():
            for m in self.gate_head.modules():
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, -2.0)  # sigmoid(-2)≈0.12

    def forward(self, x):            # x: (B, 1+k_prevs, H, W)
        if x.size(1) == 1:
            return x                 # no prevs
        cur  = x[:, :1]
        prev = x[:, 1:]
        g = self.gate_head(cur)                      # (B,1,H,W) in [0,1]
        g = self.eps + (1.0 - self.eps) * g         # floor so prevs never vanish
        prev = prev * g
        return torch.cat([cur, prev], dim=1)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,add_attn = False):
        super(UNet, self).__init__()

        # self.history_gate = HistoryGate(eps=0.1)
        # #
        # self.input_mix = nn.Sequential(
        #     nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(n_channels),
        #     nn.ReLU(inplace=True),
        # )
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.add_attn = add_attn
        if self.add_attn:
            self.attn = SelfAttention2D(1024)
            self.attn = MultiHeadSelfAttention2D(1024,8)
            self.attn = ChannelSelfAttention(in_channels=1024)

    def forward(self, x):
        # x = self.history_gate(x)
        # x = self.input_mix(x)  # <<— lets the net learn how to use history
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.add_attn:
            x5=self.attn(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def get_num_params(self):
        full_sum = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return full_sum

if __name__ == '__main__':
    unet = UNet(1,1,add_attn=True)
    unet_size = unet.get_num_params()
    print(unet_size)
