# continuous-diffusion

small, single-GPU **DDPM** on MNIST/FashionMNIST.  
tiny UNet, cosine/linear beta schedules, DDPM/DDIM sampling. good for quick demos.

---

## install

```bash
git clone https://github.com/vincentcounathe/continuous-diffusion
cd continuous-diffusion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
