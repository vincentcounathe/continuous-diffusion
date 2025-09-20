#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from torchvision import utils as vutils

from model import TinyUNet
from diffusion import make_beta_schedule, DDPM

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to model .pt")
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", default="cosine")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--out", default="samples.png")
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--channels", type=int, default=1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyUNet(in_ch=args.channels).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    betas = make_beta_schedule(args.timesteps, args.schedule)
    diff = DDPM(betas, device=device)

    with torch.no_grad():
        imgs = diff.sample(model, (args.n, args.channels, args.img_size, args.img_size), eta=args.ddim_eta)
        vutils.save_image((imgs + 1) * 0.5, args.out, nrow=int(args.n ** 0.5))
    print("saved:", args.out)

if __name__ == "__main__":
    main()
