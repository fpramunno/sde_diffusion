# SDO Diffusion Model

This repository contains a PyTorch-based diffusion model designed for image-to-image translation using multi-channel solar observations from SDO (Solar Dynamics Observatory), including magnetograms, continuum intensity, and EUV wavelengths (AIA 171, 1600, 1700 Å). The model leverages a denoising diffusion probabilistic framework to reconstruct or forecast physical solar features.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/fpramunno/sdo_diffusion.git
cd sdo_diffusion
```

# Set up the environment
```bash
pip install -r requirements.txt
```
# Training
```bash
python train_sdo.py --config ./configs/config_256x256_sdohmi.json --batch-size 8 --name model_run_01 --dir-name my_test_run --wandb-project sdo_img2img --wandb-entity your_wandb_username --use_wandb True --wandb-save-model
```
# Output

Model checkpoints will be saved in:
```bash
model_<dir-name>/
```

Sample predictions and reconstructions in:
```bash
results_<dir-name>/
```

Logging optionally handled via Weights & Biases

