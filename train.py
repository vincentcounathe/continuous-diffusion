#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

from model import TinyUNet
from diffusion import make_beta_schedule, DDPM

def get_loader(dataset="MNIST", batch_size=128, img_size=32):
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),  # map to [-1,1]
    ])
    if dataset.upper() == "MNIST":
        ds = datasets.MNIST(root="./data", download=True, train=True, transform=tfm)
        channels = 1
    elif dataset.upper() == "FASHIONMNIST":
        ds = datasets.FashionMNIST(root="./data", download=True, train=True, transform=tfm)
        channels = 1
    else:
        raise ValueError("dataset must be MNIST or FashionMNIST")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True), channels

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    p.add_argument("--out", default="runs/continuous")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--ddim_eta", type=float, default=0.0)  # 0 = DDIM
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader, channels = get_loader(args.dataset, args.bs, args.img_size)
    model = TinyUNet(in_ch=channels, base=64, ch_mult=(1, 2, 2), t_dim=128).to(device)
    betas = make_beta_schedule(args.timesteps, args.schedule)
    diff = DDPM(betas, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        for x, _ in loader:
            x = x.to(device)

            # sample t and noise
            B = x.size(0)
            t = torch.randint(0, diff.T, (B,), device=device)
            noise = torch.randn_like(x)
            x_t = diff.add_noise(x, t, noise)

            # model predicts noise
            t_cont = (t.float() + 0.5) / diff.T
            pred = model(x_t, t_cont)

            loss = mse(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if global_step % 100 == 0:
                print(f"epoch {epoch} step {global_step} | loss {loss.item():.4f}")
            global_step += 1

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.out, f"unet_e{epoch}.pt"))
            # quick sample grid with DDIM
            model.eval()
            with torch.no_grad():
                imgs = diff.sample(model, (16, channels, args.img_size, args.img_size), eta=args.ddim_eta)
                vutils.save_image((imgs + 1) * 0.5, os.path.join(args.out, f"samples_e{epoch}.png"), nrow=4)
            model.train()

    print("Done. Models and samples saved in:", args.out)

if __name__ == "__main__":
    main()
