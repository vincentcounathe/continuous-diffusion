# Continuous Diffusion

A small diffusion model implementation for images.  
Trains a UNet on MNIST or FashionMNIST and generates samples using DDPM or DDIM.  
Designed to be simple and runnable on a single GPU or Colab.

---

## Usage

- **Training:** run `train.py` to fit the model.  
- **Sampling:** run `sample.py` with a checkpoint to generate images.  
- Results (checkpoints and sample grids) are saved in `runs/continuous/`.  
