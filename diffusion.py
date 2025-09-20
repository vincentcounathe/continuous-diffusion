#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch

# Beta schedules
def make_beta_schedule(T, schedule="cosine", max_beta=0.999):
    if schedule == "linear":
        betas = torch.linspace(1e-4, 0.02, T)
    elif schedule == "cosine":
        # from Nichol & Dhariwal improved DDPM
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(0, max_beta).float()
    else:
        raise ValueError("unknown schedule")
    return betas

class DDPM:
    def __init__(self, betas: torch.Tensor, device="cuda"):
        self.device = device
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.T = len(betas)

    def add_noise(self, x0, t, noise):
        # q(x_t | x_0) = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) eps
        a = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        b = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return a * x0 + b * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t, eta=0.0):  # DDIM when eta=0
        # Predict noise eps_theta(x_t, t)
        t_cont = (t.float() + 0.5) / self.T  # map to [0,1]
        eps = model(x_t, t_cont)

        alpha_t = extract(self.alphas, t, x_t.shape)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alpha = extract(self.sqrt_recip_alphas, t, x
