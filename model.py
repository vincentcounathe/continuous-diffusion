#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- tiny sinusoidal time embedding ---
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t in [0, 1]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), half, device=device)
        )
        ang = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

# --- simple ResBlock with time conditioning ---
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(t_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

# --- tiny UNet (28x28/32x32 friendly) ---
class TinyUNet(nn.Module):
    def __init__(self, in_ch=1, base=64, ch_mult=(1, 2, 2), t_dim=128):
        super().__init__()
        self.t_proj = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        c0 = base * ch_mult[0]
        c1 = base * ch_mult[1]
        c2 = base * ch_mult[2]

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.down1 = ResBlock(base, c0, t_dim)
        self.down2 = ResBlock(c0, c1, t_dim)
        self.down3 = ResBlock(c1, c2, t_dim)
        self.pool = nn.AvgPool2d(2)

        self.mid1 = ResBlock(c2, c2, t_dim)
        self.mid2 = ResBlock(c2, c2, t_dim)

        self.up3 = ResBlock(c2 + c1, c1, t_dim)
        self.up2 = ResBlock(c1 + c0, c0, t_dim)
        self.up1 = ResBlock(c0 + base, base, t_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )

    def forward(self, x, t):  # t in [0,1]
        t_emb = self.t_proj(t)

        x0 = self.in_conv(x)
        d1 = self.down1(x0, t_emb)
        p1 = self.pool(d1)
        d2 = self.down2(p1, t_emb)
        p2 = self.pool(d2)
        d3 = self.down3(p2, t_emb)
        p3 = self.pool(d3)

        m = self.mid2(self.mid1(p3, t_emb), t_emb)

        u3 = F.interpolate(m, scale_factor=2, mode="nearest")
        u3 = self.up3(torch.cat([u3, d3], dim=1), t_emb)

        u2 = F.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = self.up2(torch.cat([u2, d2], dim=1), t_emb)

        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = self.up1(torch.cat([u1, d1], dim=1), t_emb)

        return self.out(u1)
