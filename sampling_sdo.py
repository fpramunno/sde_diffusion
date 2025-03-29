# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 09:39:46 2025

@author: pio-r
"""

import argparse
import os
os.chdir(r"/mnt/nas05/data01/francesco/sdo_img2img/k_diffusion")
# os.chdir(r"\\10.35.146.35\data01\francesco\sdo_img2img\k_diffusion")
# from pathlib import Path
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from copy import deepcopy
from functools import partial
import importlib.util
import math
import json
from pathlib import Path
import time
import numpy as np
import accelerate
import safetensors.torch as safetorch
import torch
import torch._dynamo
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import optim
from torch.utils import data #, flop_counter
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from astropy.io import fits

import k_diffusion as K
import pandas as pd
from dataset_functions import PairedFITSDataset, get_default_transforms, setup_logging, save_images
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())
        
p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--batch-size', type=int, default=64,
               help='the batch size')
p.add_argument('--checkpointing', action='store_true',
               help='enable gradient checkpointing')
p.add_argument('--clip-model', type=str, default='ViT-B/16',
               choices=K.evaluation.CLIPFeatureExtractor.available_models(),
               help='the CLIP model to use to evaluate')
p.add_argument('--compile', action='store_true',
               help='compile the model')
p.add_argument('--config', type=str, required=True,
               help='the configuration file')
p.add_argument('--demo-every', type=int, default=500,
               help='save a demo grid every this many steps')
p.add_argument('--dinov2-model', type=str, default='vitl14',
               choices=K.evaluation.DINOv2FeatureExtractor.available_models(),
               help='the DINOv2 model to use to evaluate')
p.add_argument('--dir-name', type=str, default='256_6to_1_unetcond',
               help='the directory name to use')  # <---- Added this line
p.add_argument('--end-step', type=int, default=None,
               help='the step to end training at')
p.add_argument('--evaluate-every', type=int, default=10000,
               help='evaluate every this many steps')
p.add_argument('--evaluate-n', type=int, default=2000,
               help='the number of samples to draw to evaluate')
p.add_argument('--evaluate-only', action='store_true',
               help='evaluate instead of training')
p.add_argument('--evaluate-with', type=str, default='inception',
               choices=['inception', 'clip', 'dinov2'],
               help='the feature extractor to use for evaluation')
p.add_argument('--gns', action='store_true',
               help='measure the gradient noise scale (DDP only, disables stratified sampling)')
p.add_argument('--grad-accum-steps', type=int, default=1,
               help='the number of gradient accumulation steps')
p.add_argument('--lr', type=float,
               help='the learning rate')
p.add_argument('--mixed-precision', type=str,
               help='the mixed precision type')
p.add_argument('--name', type=str, default='model',
               help='the name of the run')
p.add_argument('--num-workers', type=int, default=8,
               help='the number of data loader workers')
p.add_argument('--reset-ema', action='store_true',
               help='reset the EMA')
p.add_argument('--resume', type=str,
               help='the checkpoint to resume from')
p.add_argument('--resume-inference', type=str,
               help='the inference checkpoint to resume from')
p.add_argument('--sample-n', type=int, default=64,
               help='the number of images to sample for demo grids')
p.add_argument('--save-every', type=int, default=10000,
               help='save every this many steps')
p.add_argument('--seed', type=int,
               help='the random seed')
p.add_argument('--start-method', type=str, default='spawn',
               choices=['fork', 'forkserver', 'spawn'],
               help='the multiprocessing start method')
p.add_argument('--wandb-entity', type=str,
               help='the wandb entity name')
p.add_argument('--wandb-group', type=str,
               help='the wandb group name')
p.add_argument('--wandb-project', type=str,
               help='the wandb project name (specify this to enable wandb)')
p.add_argument('--wandb-save-model', action='store_true',
               help='save model to wandb')
# args = p.parse_args()
args = p.parse_args(["--config", "./configs/config_256x256_sdohmi.json"])

dir_path_res = f"results_{args.dir_name}"
dir_path_mdl = f"model_{args.dir_name}"

# if not os.path.exists(dir_path_res):
#     os.makedirs(dir_path_res)
    
# if not os.path.exists(dir_path_mdl):
#     os.makedirs(dir_path_mdl)

# mp.set_start_method(args.start_method)
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch._dynamo.config.automatic_dynamic_shapes = False
except AttributeError:
    pass

config = K.config.load_config(args.config)
model_config = config['model']
dataset_config = config['dataset']
opt_config = config['optimizer']
sched_config = config['lr_sched']
ema_sched_config = config['ema_sched']

# TODO: allow non-square input sizes
assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
size = model_config['input_size']

accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps, mixed_precision=args.mixed_precision)
# ensure_distributed() # useful if training on multiple gpus
device = accelerator.device
unwrap = accelerator.unwrap_model
print(f'Process {accelerator.process_index} using device: {device}', flush=True)
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    print(f'World size: {accelerator.num_processes}', flush=True)
    print(f'Batch size: {args.batch_size * accelerator.num_processes}', flush=True)

if args.seed is not None:
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
    torch.manual_seed(seeds[accelerator.process_index])
demo_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())
elapsed = 0.0

# Model definition

inner_model = K.config.make_model(config)
inner_model_ema = deepcopy(inner_model)

if args.compile:
    inner_model.compile()
    # inner_model_ema.compile()

if accelerator.is_main_process:
    print(f'Parameters: {K.utils.n_params(inner_model):,}')


# WANDB LOGGING
use_wandb = False # accelerator.is_main_process and args.wandb_project
if use_wandb:
    import wandb
    log_config = vars(args)
    log_config['config'] = config
    log_config['parameters'] = K.utils.n_params(inner_model)
    wandb.init(project="sdo_img2img", entity="francescopio", config=log_config, save_code=True)



assert ema_sched_config['type'] == 'inverse'
ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                              max_value=ema_sched_config['max_value'])
ema_stats = {}


# Create Dataloader

# Input:
dir2 = '/mnt/nas05/astrodata01/sdo_hmi/mag_24_filtered/'
# dir2 = '//10.35.146.35/astrodata01/sdo_hmi/mag_24_filtered/'
dir171 = '/mnt/nas05/astrodata01/sdo_aia/data_171_24_filtered/'
# dir171 = '//10.35.146.35/astrodata01/sdo_aia/data_171_24_filtered/'
dir1600 = '/mnt/nas05/astrodata01/sdo_aia/data_1600_24_filtered/'
# dir1600 = '//10.35.146.35/astrodata01/sdo_aia/data_1600_24_filtered/'
dir1700 = '/mnt/nas05/astrodata01/sdo_aia/data_1700_24_filtered/'
# dir1700 = '//10.35.146.35/astrodata01/sdo_aia/data_1700_24_filtered/'
dirdoppl = '/mnt/nas05/astrodata01/sdo_hmi/dopplergram_24_filtered/'
# dirdoppl = '//10.35.146.35/astrodata01/sdo_hmi/dopplergram_24_filtered/'
dircont = '/mnt/nas05/astrodata01/sdo_hmi/continuum_24_filtered/'
# dircont = '//10.35.146.35/astrodata01/sdo_hmi/continuum_24_filtered/'

# Flr
dir2_flr = '/mnt/nas05/astrodata01/sdo_hmi/mag_filtered/'
# dir2_flr = '//10.35.146.35/astrodata01/sdo_hmi/mag_filtered/'
dir171_flr = '/mnt/nas05/astrodata01/sdo_aia/data_171_filtered/'
# dir171_flr = '//10.35.146.35/astrodata01/sdo_aia/data_171_filtered/'
dir1600_flr = '/mnt/nas05/astrodata01/sdo_aia/data_1600_filtered/'
# dir1600_flr = '//10.35.146.35/astrodata01/sdo_aia/data_1600_filtered/'
dir1700_flr = '/mnt/nas05/astrodata01/sdo_aia/data_1700_filtered/'
# dir1700_flr = '//10.35.146.35/astrodata01/sdo_aia/data_1700_filtered/'
dirdoppl_flr = '/mnt/nas05/astrodata01/sdo_hmi/doppl_filter_effects_corr/'
# dirdoppl_flr = '//10.35.146.35/astrodata01/sdo_hmi/doppl_filter_effects_corr/'
dircont_flr = '/mnt/nas05/astrodata01/sdo_hmi/continuum_filtered/'
# dircont_flr = '//10.35.146.35/astrodata01/sdo_hmi/continuum_filtered/'


# transform_2 = get_default_transforms(
#     target_size=256, channel="magnetogram", mask_limb=False, radius_scale_factor=1.0)

transform_171 = get_default_transforms(
    target_size=256, channel="171A", mask_limb=False, radius_scale_factor=1.0)

transform_1600 = get_default_transforms(
    target_size=256, channel="1600A", mask_limb=False, radius_scale_factor=1.0)

transform_1700 = get_default_transforms(
    target_size=256, channel="1700A", mask_limb=False, radius_scale_factor=1.0)

# transform_cont = get_default_transforms(
#     target_size=256, channel="continuum", mask_limb=False, radius_scale_factor=1.0)

transform_doppl = transforms.Compose([
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomHorizontalFlip(p=1.0),  # This line adds a 90-degree rotation to the right
    transforms.Normalize(mean=(-2000), std=(4000)),
    transforms.Normalize(mean=(0.5), std=(0.5))
])

# path_df_mag = r"\\10.35.146.35\data01\francesco\sdo_img2img/mag_24_jsoc_stats_arcsinh.csv"
path_df_mag = r"/mnt/nas05/data01/francesco/sdo_img2img/mag_24_jsoc_stats_arcsinh.csv"
# path_df_dpl = r"\\10.35.146.35\data01\francesco\sdo_img2img/continuum_24_jsoc_stats_arcsinh_v2.csv"
path_df_dpl = r"/mnt/nas05/data01/francesco/sdo_img2img/continuum_24_jsoc_stats_arcsinh_v2.csv"



df_mag = pd.read_csv(r"/mnt/nas05/data01/francesco/sdo_img2img/mag_24_jsoc_stats_arcsinh.csv")
# df_mag = pd.read_csv(r"\\10.35.146.35\data01\francesco\sdo_img2img/mag_24_jsoc_stats_arcsinh.csv")

df_cnt = pd.read_csv(r"/mnt/nas05/data01/francesco/sdo_img2img/continuum_24_jsoc_stats_arcsinh_v2.csv")
# df_cnt = pd.read_csv(r"\\10.35.146.35\data01\francesco\sdo_img2img/continuum_24_jsoc_stats_arcsinh_v2.csv")

global_min = df_mag["Min"].min()
global_max = df_mag["Max"].max()

global_min_cont = df_cnt["Min"].min()
global_max_cont = df_cnt["Max"].max()

transform_2 = transforms.Compose([
    transforms.Normalize(mean=[global_min], std=[global_max - global_min]),
    transforms.Normalize(mean=(0.5), std=(0.5))
])

transform_cont = transforms.Compose([
    transforms.Normalize(mean=[global_min_cont], std=[global_max_cont - global_min_cont]),
    transforms.Normalize(mean=(0.5), std=(0.5))
])

# Training Dataset (2013, 2015, 2017, 2018, 2019)
train_dataset = PairedFITSDataset(
    dir2, dir171, dir1600, dir1700, dirdoppl, dircont, 
    dir2_flr, dir171_flr, dir1600_flr, dir1700_flr, dirdoppl_flr, dircont_flr, 
    train=True,  # Train mode
    transform2=transform_2, transform_171=transform_171, 
    transform_1600=transform_1600, transform_1700=transform_1700, 
    transform_doppl=transform_doppl, transform_cont=transform_cont, df=path_df_mag, df_doppl=path_df_dpl
)

total_samples = len(train_dataset)
train_size = int(0.7 * total_samples)  # Using 80% for training as an example
val_size = total_samples - train_size


torch.manual_seed(42)
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
import random
from torch.utils.data import DataLoader, Subset
# Randomly select 250 indices from the validation set
subset_indices = random.sample(range(len(val_dataset)), 250)

# Create a subset of the dataset
val_subset = Subset(val_dataset, subset_indices)

# Create a DataLoader from the subset
val_dl = DataLoader(val_subset, batch_size=1)

# train_dl = DataLoader(train_dataset, 1)

# val_dl = DataLoader(val_dataset, 1,)


# Define the model 

model = K.config.make_denoiser_wrapper(config)(inner_model)

ckpt_path = f"./{dir_path_mdl}/model_00104086.pth"

ckpt = torch.load(ckpt_path, map_location='cpu')
unwrap(model.inner_model).load_state_dict(ckpt['model_ema'])

from k_diffusion import sampling

def generate_samples(model, num_samples, device, inpt_cond, sampler="dpmpp_2m", step=50):
    """
    Generate samples using k-diffusion samplers.

    Args:
        model: The trained denoising model.
        num_samples (int): Number of samples to generate.
        device (torch.device): Device to run inference on.
        sampler (str): The sampler to use (default: "dpmpp_2m").

    Returns:
        Tensor: Generated samples.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        x = torch.randn(num_samples, 1, 256, 256, device=device)  # Start with noise
        # x = torch.cat([x, inpt_cond.reshape(num_samples, 6, 256, 256)], dim=1)
        extra_args = {
                        "unet_cond": inpt_cond,
                    }
        sigmas = sampling.get_sigmas_karras(n=step, sigma_min=1e-2, sigma_max=80, rho=7.0, device=device)


        # ✅ Choosing the correct sampler
        if sampler == "euler":
            samples = sampling.sample_euler(model, x, sigmas, extra_args=extra_args)
        elif sampler == "euler_ancestral":
            samples = sampling.sample_euler_ancestral(model, x, sigmas, extra_args=extra_args)
        elif sampler == "heun":
            samples = sampling.sample_heun(model, x, sigmas, extra_args=extra_args)
        elif sampler == "dpm_2":
            samples = sampling.sample_dpm_2(model, x, sigmas, extra_args=extra_args)
        elif sampler == "dpm_2_ancestral":
            samples = sampling.sample_dpm_2_ancestral(model, x, sigmas, extra_args=extra_args)
        elif sampler == "dpmpp_2m":
            samples = sampling.sample_dpmpp_2m(model, x, sigmas, extra_args=extra_args)
        elif sampler == "dpmpp_2m_sde":
            samples = sampling.sample_dpmpp_2m_sde(model, x, sigmas, extra_args=extra_args)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

    return samples  # Return generated images
from tqdm import tqdm
model.to(device)

from sunpy.net import hek
from sunpy.net import attrs as a
from util import extract_and_format_datetime
from util import reverse_scaling, compute_area, mask_outside_circle
from util import sun_earth_distance_in_meters, hpc_to_pixel, extract_and_format_datetime
from util import obtain_contour
from util import persistence_perd
from util import is_within_circle
from util import comput_jaccard_index
from util import plot_intersection_contour
from util import obtain_contour_all

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

psnr = PeakSignalNoiseRatio(data_range=1.0).to('cuda')
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
lpips_distance = lpips.LPIPS(net='alex').to('cuda')



def back_to_physical(global_min, global_max, alpha=0.1):
    return transforms.Compose([
        transforms.Lambda(lambda x: x * 0.5 + 0.5),  # [-1, 1] → [0, 1]
        transforms.Lambda(lambda x: x * (global_max - global_min) + global_min),  # [0, 1] → original
        transforms.Lambda(lambda x: torch.sinh(x) / alpha)  # undo arcsinh(alpha * x)
    ])

inverse_transform = back_to_physical(global_min, global_max, alpha=0.1)
flip = transforms.RandomVerticalFlip(p=1.0)

for f in range(1, 10):
    psnr_gen = []
    ssim_gen = []
    lpips_gen = []

    full_disk_tot = []
    full_disk_net = []
    full_disk_tot_samp = []
    full_disk_net_samp = []
    flux_diff_tot = []
    flux_diff_net = []
    flux_diff_tot_samp = []
    flux_diff_net_samp = []
    pv_full_disk_tot = []
    pv_full_disk_net = []
    pv_full_disk_tot_pers = []
    pv_full_disk_net_pers= []
    pv_flux_ev_fd_tot = []
    pv_flux_ev_fd_net = []
    pv_flux_ev_fd_tot_pers = []
    pv_flux_ev_fd_net_pers = []

    def create_dict():
        return {str(i): [] for i in range(len(val_subset))}

    # Create the dictionaries using the function
    ar_tot = create_dict()
    ar_net = create_dict()
    ar_tot_samp = create_dict()
    ar_net_samp = create_dict()
    pv_ar_tot = create_dict()
    pv_ar_net = create_dict()
    pv_ar_tot_ev = create_dict()
    pv_ar_net_ev = create_dict()
    size_ar = create_dict()
    size_ar_positive = create_dict()
    size_ar_negative = create_dict()
    orientation = create_dict()
    distance_center = create_dict()
    pv_ar_tot_pers = create_dict()
    pv_ar_net_pers = create_dict()
    pv_orientation_samp = create_dict()
    pv_orientation_pers = create_dict()
    jacc_samp = create_dict()
    jacc_per = create_dict()

    client_24 = hek.HEKClient()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dl, desc="Validation", disable=not accelerator.is_main_process)):
            inpt = (torch.cat(list((batch[0].values())), dim=1)).float().to(device)
            trgt = (batch[1]['dir2_flr']).float().to(device)
            time24, timepeak, cdelt1, cdelt2, crpix1, crpix2, rsun_obs, path_24, path_flr = batch[2:]
            
            original_size = (4096, 4096)  # Original image size
            new_size = (256, 256)        # New image size after resizing
            
            # Image scale in arcseconds per pixel
            scale_x = cdelt1[0].item() # CDELT1
            scale_y = cdelt2[0].item() # CDELT2
            
            # Reference point in pixel coordinates
            ref_pixel_x = crpix1[0].item() # CRPIX1
            ref_pixel_y = crpix2[0].item() # CRPIX2
            
            scale_img = 256 / 4096
            
            rsun_obs = rsun_obs[0].item() 
           
            # generated_imgs = []
            # samplers = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "dpmpp_2m", "dpmpp_2m_sde"]
            # for samp in tqdm(samplers):
            #     generated_imgs.append(generate_samples(model, 1, device, inpt_cond=inpt, sampler=samp, step=50))
            
            
            input_images = [inpt[0][i].reshape(1, 256, 256).permute(1, 2, 0).cpu().numpy() for i in range(6)]
            target_image = trgt[0].reshape(1, 256, 256).permute(1, 2, 0).cpu().numpy()
            
            # # Set up the figure with 2 rows and 7 columns
            # fig, axes = plt.subplots(2, 7, figsize=(20, 6))
            
            # # Plot the 6 input images and target image in the first row
            # for i in range(6):
            #     axes[0, i].imshow(input_images[i], origin='lower', cmap='gray')
            #     axes[0, i].set_title(f'Ch {i+1}')
            #     axes[0, i].axis('off')
            
            # axes[0, 6].imshow(target_image, origin='lower', cmap='gray')
            # axes[0, 6].set_title('Target')
            # axes[0, 6].axis('off')
            
            # # Plot the 7 generated images in the second row
            # for i in range(7):
            #     axes[1, i].imshow(generated_imgs[i][0][0].reshape(1, 256, 256).permute(1, 2, 0).cpu().numpy() , origin='lower', cmap='gray')
            #     axes[1, i].set_title(samplers[i])
            #     axes[1, i].axis('off')
            
            # # Adjust layout
            # plt.tight_layout()
            # plt.show()
            
            gen_img = generate_samples(model, 1, device, inpt_cond=inpt, sampler="dpmpp_2m_sde", step=50)
            # np.save(rf'\\10.35.146.35\data01\francesco\sdo_img2img\k_diffusion\sample_results\diff_unet_cond/{o}/gen_img_{i}.npy', gen_img.detach().cpu().numpy())
            np.save(rf'/mnt/nas05/data01/francesco/sdo_img2img/k_diffusion/sample_results/diff_unet_cond/{f}/gen_img_{i}.npy', gen_img.detach().cpu().numpy())
    
            img_rec_gen = gen_img.detach().cpu().float()
            tensor_img = trgt.detach().cpu().float()
            
            ## Mask 
            
            radius_pixels = (rsun_obs / scale_x) * scale_img #- 4
            
            # mask the images
            
            center_pix = ref_pixel_x * scale_img
            
            mask_samp, num_pix_true = mask_outside_circle(img_rec_gen, center_pix, center_pix, radius_pixels, torch.min(tensor_img).item())
            mask_24, num_pix_true = mask_outside_circle(inpt[0][0].detach().cpu().float(), center_pix, center_pix, radius_pixels, torch.min(inpt[0][0]).item())
            mask_true, num_pix_true = mask_outside_circle(tensor_img, center_pix, center_pix, radius_pixels, torch.min(tensor_img).item())
            
            # fig, axes = plt.subplots(2, 6, figsize=(20, 6))
            # colormaps = ['hmimag', 'sdoaia171', 'sdoaia1600', 'sdoaia1700', 'RdBu_r', 'gray']
            # titles = ['LoS Magnetogram', 'AIA 171', 'AIA 1600', 'AIA 1700', 'Dopplergram', 'Continuum']
            # # --- First row: 6 input images ---
            # for i in range(6):
                
            #     axes[0, i].imshow(input_images[i], origin='lower', cmap=colormaps[i])
            #     axes[0, i].set_title(f'{titles[i]}', fontsize=25)
            #     axes[0, i].axis('off')
            
            # # Leave the 7th column in row 0 blank
            # # axes[0, 6].axis('off')
            
            # # --- Second row: center the target and generated images ---
            # # Clear all second row axes first
            # for ax in axes[1]:
            #     ax.axis('off')
            
            # # Position target in column 2 and generated in column 4 (centered in row)
            # axes[1, 2].imshow(target_image, origin='lower', cmap='hmimag')
            # axes[1, 2].set_title("Target + 24h", fontsize=25)
            # axes[1, 2].axis('off')
            
            # # Assuming generated_imgs is shape (1, 1, 256, 256)
            # axes[1, 3].imshow(mask_samp[0].permute(1, 2, 0).cpu().numpy(), origin='lower', cmap='hmimag')
            # axes[1, 3].set_title("Generated + 24h", fontsize=25)
            # axes[1, 3].axis('off')
            
            # # Adjust layout
            # plt.tight_layout()
            # plt.show()
            
            # CS Metrics computation
        
            
            psnr_gen.append(psnr(tensor_img.reshape(1, 1, 256, 256).to(device), mask_samp.reshape(1, 1, 256, 256).to(device)).item())
            
            ssim_gen.append(ssim(tensor_img.reshape(1, 1, 256, 256).to(device), mask_samp.reshape(1, 1, 256, 256).to(device)).item())
            
            lpips_gen.append(lpips_distance(mask_samp.float().reshape(1, 1, 256, 256).to(device), tensor_img.reshape(1, 1, 256, 256).to(device)).item())
            
            
            # TRANSFORM TO GAUSS VALUES
            
            tensor_24_trans = flip(inverse_transform(mask_24[0]).reshape(1, 256, 256))
            tensor_img_trans = flip(inverse_transform(tensor_img)).reshape(1, 256, 256)
            gen_img_trans = flip(inverse_transform(mask_samp)).reshape(1, 256, 256)
            
            # Convert tensors to NumPy
            inp_img_np = tensor_24_trans.squeeze().cpu().numpy()
            trg_img_np = tensor_img_trans.squeeze().cpu().numpy()
            gen_img_np = gen_img_trans.squeeze().cpu().numpy()
            
            # # Set up figure
            # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # # Plot first image
            # axes[0].imshow(trg_img_np, cmap='hmimag', origin='lower')
            # axes[0].set_title('Original Image')
            # axes[0].axis('off')
            
            # # Plot second image
            # axes[1].imshow(gen_img_np, cmap='hmimag', origin='lower')
            # axes[1].set_title('Generated Image')
            # axes[1].axis('off')
            
            # # Show
            # plt.tight_layout()
            # plt.show()
            
            ##### COMPUTE THE FULL DISK TOTAL MAG FLUX AND NET FLUX
            
            tstart, tend = extract_and_format_datetime(timepeak[0])
            
            distance_earth = sun_earth_distance_in_meters(tstart)
            
            pix_area = compute_area(scale_x, distance_earth) # in meters
            
            num_pix_true = 256 * 256
            
            total_area = pix_area * num_pix_true
            
            total_flux_24 = torch.sum(torch.abs(mask_24)).item() * total_area
            
            net_flux_24 = torch.sum(mask_24).item() * total_area
            
            total_unsigned_flux = torch.sum(torch.abs(mask_true)).item() * total_area
            
            total_net_flux = torch.sum(mask_true).item() * total_area
            
            print('Total Flux = {}'.format(total_unsigned_flux))
            print('Net Flux = {}'.format(total_net_flux))
            
            total_unsigned_flux_samp = torch.sum(torch.abs(mask_samp)).item() * total_area
            
            total_net_flux_samp = torch.sum(mask_samp).item() * total_area
            
            print('Total Flux Samp = {}'.format(total_unsigned_flux_samp))
            print('Net Flux Samp = {}'.format(total_net_flux_samp))
            
            # perc_var_tot = ((total_unsigned_flux_samp - total_unsigned_flux) / total_unsigned_flux_samp) * 100
            
            # perc_var_net = ((total_net_flux_samp - total_net_flux) / total_net_flux_samp) * 100
            
            perc_var_tot = ((total_unsigned_flux - total_unsigned_flux_samp) / total_unsigned_flux) * 100
            
            perc_var_net = ((total_net_flux - total_net_flux_samp) / total_net_flux) * 100
            
            # SAVE VALUES 
            
            flux_diff_tot.append(total_unsigned_flux - total_flux_24)
            flux_diff_net.append(total_net_flux - net_flux_24)
            full_disk_tot.append(total_unsigned_flux)
            full_disk_net.append(total_net_flux)
            flux_diff_tot_samp.append(total_unsigned_flux_samp - total_flux_24)
            flux_diff_net_samp.append(total_net_flux_samp - net_flux_24)
            full_disk_tot_samp.append(total_unsigned_flux_samp)
            full_disk_net_samp.append(total_net_flux_samp)
            pv_full_disk_tot.append(perc_var_tot)
            pv_full_disk_net.append(perc_var_net)
            pv_flux_ev_fd_tot.append((((total_unsigned_flux_samp - total_flux_24) - (total_unsigned_flux - total_flux_24)) / (total_unsigned_flux_samp - total_flux_24)) * 100)
            pv_flux_ev_fd_net.append((((total_net_flux_samp - net_flux_24) - (total_net_flux - net_flux_24)) / (total_net_flux_samp - net_flux_24)) * 100)
            
            ## Persistence Model
            # TODO: CHANGE THE SCALING 
            
            # path_24 = path_24.replace(
            #         "//10.35.146.35/data01/francesco/sdo_data/final_dataset/mag_24_jsoc_4K",
            #         '//10.35.146.35/astrodata01/sdo_hmi/mag_24_filtered/'
            #     )
            
            # with fits.open(path_24[0]) as hdul:
            #     mag_4k_24 = hdul[1].data
            #     mag_4k_24 = np.nan_to_num(mag_4k_24, nan=np.nanmin(mag_4k_24))
                
            
            # alpha = 25
            # beta = 6
            # flattened_tensor1 = mag_4k_24.flatten()
            
            # mag_4k_24 = np.arcsinh(flattened_tensor1 / alpha) / beta
            
            # mag_4k_24 = flip_hor((rotate(torch.from_numpy(mag_4k_24).reshape(4096, 4096)))).float()
            
            original_tensor, persistence_pred = persistence_perd(path_24[0])
            
            # mask_perst, num_pix_true = mask_outside_circle(persistence_pred, center_pix, center_pix, radius_pixels, np.nanmin(tensor_24_trans.numpy()))
            
            # original_tensor = reverse_scaling(original_tensor)
            # mask_perst = reverse_scaling(persistence_pred)
            
            original_24 = original_tensor.reshape(1, 256, 256).permute(1, 2, 0).cpu().numpy()
            persistence_24 = persistence_pred.permute(1, 2, 0).cpu().numpy()
            persistence_24 = np.nan_to_num(persistence_24, nan=np.nanmin(trg_img_np))
    
            mask_perst = torch.from_numpy(persistence_24).reshape(1, 256, 256)    
        
            per_img_np = mask_perst.permute(1, 2, 0).cpu().numpy()
            
            # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
            # # Plot the original image in the first subplot
            # im1 = ax1.imshow(trg_img_np, cmap='gray')
            # ax1.set_title('Target', fontsize=36)
            # ax1.set_xticks([])
            # ax1.set_yticks([])
            
            # # Plot the EMA sampled image in the second subplot
            # im2 = ax2.imshow(gen_img_np, cmap='gray')
            # ax2.set_title('Diffusion model', fontsize=36)
            # ax2.set_xticks([])
            # ax2.set_yticks([])
            
            # # Plot the diffusion model image in the third subplot
            # im3 = ax3.imshow(per_img_np, cmap='gray')
            # ax3.set_title('Persistence model', fontsize=36)
            # ax3.set_xticks([])
            # ax3.set_yticks([])
            
            # # Adjust the spacing between subplots
            # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
            
            # # Show the plot
            # plt.show()
            
            # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
            # # Plot the original image in the first subplot
            # im1 = ax1.hist(trg_img_np.flatten(), bins=100)
            # ax1.set_title('Target', fontsize=36)
            # ax1.set_xticks([])
            # ax1.set_yticks([])
            
            # # Plot the EMA sampled image in the second subplot
            # im2 = ax2.hist(gen_img_np.flatten(), bins=100)
            # ax2.set_title('Diffusion model', fontsize=36)
            # ax2.set_xticks([])
            # ax2.set_yticks([])
            
            # # Plot the diffusion model image in the third subplot
            # im3 = ax3.hist(per_img_np.flatten(), bins=100)
            # ax3.set_title('Persistence model', fontsize=36)
            # ax3.set_xticks([])
            # ax3.set_yticks([])
            
            # # Adjust the spacing between subplots
            # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
            
            # # Show the plot
            # plt.show()
            # plt.close()
            
            # fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(15, 5))
    
            # # Plot the original image in the first subplot
    
            # # Plot the EMA sampled image in the second subplot
            # im2 = ax2.imshow(gt_peak)
            # ax2.set_title('Target', fontsize=25)
            # cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the second image
            
            # im3 = ax3.imshow(ema_samp - gt_peak)
            # ax3.set_title('Diffusion model Difference', fontsize=25)
            # cbar2 = fig.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the second image
            
            # im4 = ax4.imshow(persistence_24 - gt_peak)
            # ax4.set_title('Persistence model Difference', fontsize=25)
            # cbar4 = fig.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the second image
    
            # # Adjust the spacing between subplots
            # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
    
            # # Show the plot
            # plt.show()
            
            # Compute metrics with persistence model
            
            total_unsigned_flux_samp_per = torch.sum(torch.abs(mask_perst)).item() * total_area
            
            total_net_flux_samp_per = torch.sum(mask_perst).item() * total_area
            
            print('Total Flux Pers = {}'.format(total_unsigned_flux_samp_per))
            print('Net Flux Pers = {}'.format(total_net_flux_samp_per))
            
            # perc_var_tot_per = ((total_unsigned_flux_samp_per - total_unsigned_flux) / total_unsigned_flux_samp_per) * 100
            
            # perc_var_net_per = ((total_net_flux_samp_per - total_net_flux) / total_net_flux_samp_per) * 100
            
            perc_var_tot_per = ((total_unsigned_flux - total_unsigned_flux_samp_per) / total_unsigned_flux) * 100
            
            perc_var_net_per = ((total_net_flux - total_net_flux_samp_per) / total_net_flux) * 100
            
            pv_full_disk_tot_pers.append(perc_var_tot_per)
            pv_full_disk_net_pers.append(perc_var_net_per)
            pv_flux_ev_fd_tot_pers.append((((total_unsigned_flux_samp_per - total_flux_24) - (total_unsigned_flux - total_flux_24)) / (total_unsigned_flux_samp_per - total_flux_24)) * 100)
            pv_flux_ev_fd_net_pers.append((((total_net_flux_samp_per - net_flux_24) - (total_net_flux - net_flux_24)) / (total_net_flux_samp_per - net_flux_24)) * 100)
    
    
            ###### COMPUTE THE FULL DISK PER PIX #### IT IS THE SAME BECAUSE IT'S A LINEAR COMBINATION
            
            ###### COMPUTE THE ACTIVE REGION FLUX
            # 24 h INFO
            client_24 = hek.HEKClient()
            
            tstart24, tend24 = extract_and_format_datetime(time24[0])
            
            # Query the HEK for all events in the time range
            events_24 = client_24.search(a.Time(tstart24, tend24), a.hek.AR)
    
            
            # Dictionary to store HPC coordinates for each NOAA AR number
            ar_coordinates_24 = {}
            ar_number_24 = []
            
            for event in events_24:
                noaa_number = event.get('ar_noaanum')
                if noaa_number is not None:  # Check if the AR number is not None
                    hpc_coord = (event.get('hpc_x', 'N/A'), event.get('hpc_y', 'N/A'))
                    ar_coordinates_24[noaa_number] = hpc_coord
                    ar_number_24.append(noaa_number)
                    if noaa_number in ar_number_24:
                        continue
                    else:
                        ar_number_24.append(noaa_number)
                        
            ar_number_24 = list(set(ar_number_24))
            
            # PEAK INFO
            client_peak = hek.HEKClient()
            
            # Query the HEK for all events in the time range
            events_peak = client_peak.search(a.Time(tstart, tend), a.hek.AR)
    
            
            # Dictionary to store HPC coordinates for each NOAA AR number
            ar_coordinates = {}
            ar_number = []
            
            for event in events_peak:
                noaa_number = event.get('ar_noaanum')
                if noaa_number is not None:  # Check if the AR number is not None
                    hpc_coord = (event.get('hpc_x', 'N/A'), event.get('hpc_y', 'N/A'))
                    ar_coordinates[noaa_number] = hpc_coord
                    if noaa_number in ar_number:
                        continue
                    else:
                        ar_number.append(noaa_number)
    
            ar_number = list(set(ar_number))
                
            ar_fluxes = []
            
            indexes = [ar_number_24.index(item) for item in ar_number if item in ar_number_24]
            
            new_ar_number = [item for item in ar_number if item in ar_number_24]
            
            for j in range(len(new_ar_number)):
                
                # coordinate ar 24 h
                
                hpc_x_24 = ar_coordinates_24[ar_number_24[indexes[j]]][0]
                hpc_y_24 = ar_coordinates_24[ar_number_24[indexes[j]]][1]
                pixel_coordinates_24 = hpc_to_pixel(hpc_x_24, hpc_y_24, scale_x, scale_y, ref_pixel_x, ref_pixel_y, original_size, new_size)
                
                pixel_x_24 = pixel_coordinates_24[0]
                pixel_y_24 = pixel_coordinates_24[1]
                
                # coordinate ar peak
                
                hpc_x = ar_coordinates[new_ar_number[j]][0]
                hpc_y = ar_coordinates[new_ar_number[j]][1]
                pixel_coordinates = hpc_to_pixel(hpc_x, hpc_y, scale_x, scale_y, ref_pixel_x, ref_pixel_y, original_size, new_size)
                
                
                pixel_x = pixel_coordinates[0]
                pixel_y = pixel_coordinates[1]
                
                # Define the size of the box
                box_size_x = 50
                box_size_y = 40
                
                # Calculate the corners of the box 24h
                x_min_24 = int(pixel_x_24 - box_size_x / 2)
                x_max_24 = int(pixel_x_24 + box_size_x / 2)
                y_min_24 = int(pixel_y_24 - box_size_y / 2)
                y_max_24 = int(pixel_y_24 + box_size_y / 2)
                
                # Calculate the corners of the box
                x_min = int(pixel_x - box_size_x / 2)
                x_max = int(pixel_x + box_size_x / 2)
                y_min = int(pixel_y - box_size_y / 2)
                y_max = int(pixel_y + box_size_y / 2)
                
                gt_24 = inp_img_np
                gt_peak = trg_img_np
                ema_samp = gen_img_np
                
                # Extract the ROI from both images
                roi_gt_24 = gt_24[y_min_24:y_max_24, x_min_24:x_max_24]
                roi_gt_peak = gt_peak[y_min:y_max, x_min:x_max]
                roi_ema_samp = ema_samp[y_min:y_max, x_min:x_max]
                roi_ema_per = persistence_24[y_min:y_max, x_min:x_max]
                
                if roi_gt_peak.shape == (40, 50) and roi_gt_24.shape == (40, 50):
                    # print(j)
                
                    # import matplotlib.patches as patches
                    
                    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
                
                    # # Plot the original image in the first subplot
                    # im1 = ax1.imshow(gt_24, vmin=-250, vmax=250)
                    # ax1.set_title('Input Image', fontsize=25)
                    
                    # # Adding a box around the active region
                    # rect_24 = patches.Rectangle((x_min_24, y_min_24), box_size_x, box_size_y, linewidth=2, edgecolor='r', facecolor='none')
                    # rect = patches.Rectangle((x_min, y_min), box_size_x, box_size_y, linewidth=2, edgecolor='r', facecolor='none')
                    # rect2 = patches.Rectangle((x_min, y_min), box_size_x, box_size_y, linewidth=2, edgecolor='r', facecolor='none')
                    # rect3 = patches.Rectangle((x_min, y_min), box_size_x, box_size_y, linewidth=2, edgecolor='r', facecolor='none')
                    # ax1.add_patch(rect_24)
                    
                    # # Annotate the center point
                    # ax1.scatter(pixel_x_24, pixel_y_24, color='blue', s=10)  # Mark the center point
                    
                    # # Plot the EMA sampled image in the second subplot
                    # im2 = ax2.imshow(gt_peak, vmin=-250, vmax=250)
                    # ax2.set_title('Target', fontsize=25)
                    # ax2.add_patch(rect)
                    # # Annotate the center point
                    # ax2.scatter(pixel_x, pixel_y, color='blue', s=10)  # Mark the center point
                    
                    # im3 = ax3.imshow(ema_samp, vmin=-250, vmax=250)
                    # ax3.set_title('1 day prediction', fontsize=25)
                    # ax3.add_patch(rect2)
                    # # Annotate the center point
                    # ax3.scatter(pixel_x, pixel_y, color='blue', s=10)  # Mark the center point
                    
                    # im4 = ax4.imshow(persistence_24, vmin=-250, vmax=250)
                    # ax4.set_title('1 day Persistence', fontsize=25)
                    # ax4.add_patch(rect3)
                    # # Annotate the center point
                    # ax4.scatter(pixel_x, pixel_y, color='blue', s=10)  # Mark the center point
                
                    # # Adjust the spacing between subplots
                    # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
                    
                    # plt.tight_layout()
                    # plt.close()
                    # Show the plot
                    # plt.savefig(f'./final_sampling/mag_physics/{i}/{j}_full.png')
                    
                    # plt.close()
                    
                    # if 
                    
                    total_area = pix_area * (box_size_x * box_size_y)
                    
                    total_flux_24_ar = np.sum(np.abs(roi_gt_24)) * total_area
                    
                    total_unsigned_flux_ar = np.sum(np.abs(roi_gt_peak)) * total_area
                    
                    total_net_24_ar = np.sum(roi_gt_24) * total_area
                    
                    total_net_flux_ar = np.sum(roi_gt_peak) * total_area
                    
                    print('Total Flux = {}'.format(total_unsigned_flux_ar))
                    print('Net Flux = {}'.format(total_net_flux_ar))
                    
                    total_unsigned_flux_samp_ar = np.sum(np.abs(roi_ema_samp)).item() * total_area
                    
                    total_net_flux_samp_ar = np.sum(roi_ema_samp).item() * total_area
                    
                    print('Total Flux Samp = {}'.format(total_unsigned_flux_samp_ar))
                    print('Net Flux Samp = {}'.format(total_net_flux_samp_ar))
                    
                    # perc_tot_flux = ((total_unsigned_flux_samp_ar - total_unsigned_flux_ar) / total_unsigned_flux_samp_ar) * 100
                    
                    # perc_net_flux = ((total_net_flux_samp_ar - total_net_flux_ar) / total_net_flux_samp_ar) * 100
                    
                    perc_tot_flux = ((total_unsigned_flux_ar - total_unsigned_flux_samp_ar) / total_unsigned_flux_ar) * 100
                    
                    perc_net_flux = ((total_net_flux_ar - total_net_flux_samp_ar) / total_net_flux_ar) * 100
                    
                    print('Perc var Tot flux = {}'.format(perc_tot_flux))
                    print('Perc var Net flux = {}'.format(perc_net_flux))
                    
                    total_unsigned_flux_samp_ar_per = np.sum(np.abs(roi_ema_per)).item() * total_area
                    
                    total_net_flux_samp_ar_per = np.sum(roi_ema_per).item() * total_area
                    
                    print('Total Flux Pers = {}'.format(total_unsigned_flux_samp_ar_per))
                    print('Net Flux Pers = {}'.format(total_net_flux_samp_ar_per))
                    
                    # perc_tot_flux_per = ((total_unsigned_flux_samp_ar_per - total_unsigned_flux_ar) / total_unsigned_flux_samp_ar_per) * 100
                    
                    # perc_net_flux_per = ((total_net_flux_samp_ar_per - total_net_flux_ar) / total_net_flux_samp_ar_per) * 100
                    
                    perc_tot_flux_per = ((total_unsigned_flux_ar - total_unsigned_flux_samp_ar_per) / total_unsigned_flux_ar) * 100
                    
                    perc_net_flux_per = ((total_net_flux_ar - total_net_flux_samp_ar_per) / total_net_flux_ar) * 100
                    
                    
                    
                    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
        
                    # # Plot the original image in the first subplot
                    # im1 = ax1.imshow(roi_gt_24)
                    # ax1.set_title('Input', fontsize=36)
                    # cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                    # # Plot the EMA sampled image in the second subplot
                    # im2 = ax2.imshow(roi_gt_peak)
                    # ax2.set_title('Target', fontsize=36)
                    # cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                    # # Plot the EMA sampled image in the second subplot
                    # im3 = ax3.imshow(roi_ema_samp)
                    # ax3.set_title('1 day prediction', fontsize=36)
                    # cbar3 = fig.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                    
                    # im4 = ax4.imshow(roi_ema_per)
                    # ax4.set_title('1 day Persistence', fontsize=36)
                    # cbar4 = fig.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                    
                    # # Adjust the spacing between subplots
                    # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
                    
                    # # plt.figtext(0.5, 0.25, f'Percentage variation Total Flux = {perc_tot_flux:.2f}%', ha='center', fontsize=36)
                    # # plt.figtext(0.5, 0.20, f'Percentage variation Net Flux = {perc_net_flux:.2f}%', ha='center', fontsize=36)
        
                    # plt.tight_layout()
                    # plt.close()
                    # Show the plot
                    # plt.savefig(f'./final_sampling/mag_physics/{i}/{j}_crop.png')
                    
                    # fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(15, 5))
        
                
                    # # Plot the EMA sampled image in the second subplot
                    # im2 = ax2.imshow(roi_gt_peak)
                    # ax2.set_title('Target', fontsize=36)
                    # cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                    # # Plot the EMA sampled image in the second subplot
                    # im3 = ax3.imshow(roi_ema_samp - roi_gt_peak)
                    # ax3.set_title('1 day prediction', fontsize=36)
                    # cbar3 = fig.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                    
                    # im4 = ax4.imshow(roi_ema_per - roi_gt_peak)
                    # ax4.set_title('1 day Persistence', fontsize=36)
                    # cbar4 = fig.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                    
                    # # Adjust the spacing between subplots
                    # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
                    
                    # plt.figtext(0.5, 0.25, f'Average difference DM = {np.mean(np.abs(roi_ema_samp) - np.abs(roi_gt_peak)):.2f}%', ha='center', fontsize=36)
                    # plt.figtext(0.5, 0.20, f'Average difference PM = {np.mean(np.abs(roi_ema_per) - np.abs(roi_gt_peak)):.2f}%', ha='center', fontsize=36)
        
                    # plt.tight_layout()
                    
                    # plt.close()
                    
                    # Varition from the 1 day 
                    
                    var_tot_flux = np.abs(total_unsigned_flux_ar - total_flux_24_ar)
                    var_net_flux = np.abs(total_net_flux_ar - total_net_24_ar)
                    
                    var_tot_flux_samp = np.abs(total_unsigned_flux_samp_ar - total_flux_24_ar)
                    var_net_flux_samp = np.abs(total_net_flux_samp_ar - total_net_24_ar)
                    
                    # perc_var_ev_tot_ar = (var_tot_flux_samp - var_tot_flux) / var_tot_flux_samp * 100
                    # perc_var_ev_net_ar = (var_net_flux_samp - var_net_flux) / var_net_flux_samp * 100
                    
                    perc_var_ev_tot_ar = (var_tot_flux - var_tot_flux_samp) / var_tot_flux * 100
                    perc_var_ev_net_ar = (var_net_flux - var_net_flux_samp) / var_net_flux * 100
                    
                    var_tot_flux_samp_per = np.abs(total_unsigned_flux_samp_ar_per - total_flux_24_ar)
                    var_net_flux_samp_per = np.abs(total_net_flux_samp_ar_per - total_net_24_ar)
                    
                    # perc_var_ev_tot_ar_per = (var_tot_flux_samp_per - var_tot_flux) / var_tot_flux_samp_per * 100
                    # perc_var_ev_net_ar_per = (var_net_flux_samp_per - var_net_flux) / var_net_flux_samp_per * 100
                    
                    perc_var_ev_tot_ar_per = (var_tot_flux - var_tot_flux_samp_per) / var_tot_flux * 100
                    perc_var_ev_net_ar_per = (var_net_flux - var_net_flux_samp_per) / var_net_flux * 100
        
                    import cv2
    
                    if is_within_circle(tstart, hpc_x, hpc_y, degree=70):
                    
                        # Gaussian Filter
                        
                        fltr = 5
                        
                        width = 40
                        height = 50
                        
                        blur_gt_24 = cv2.GaussianBlur(roi_gt_24, (fltr, fltr), sigmaX=0).reshape(width, height, 1)
                        blur_gt_peak = cv2.GaussianBlur(roi_gt_peak, (fltr, fltr), sigmaX=0).reshape(width, height, 1)
                        blur_ema_samp = cv2.GaussianBlur(roi_ema_samp, (fltr, fltr), sigmaX=0).reshape(width, height, 1)
                        blur_ema_per = cv2.GaussianBlur(roi_ema_per, (fltr, fltr), sigmaX=0).reshape(width, height, 1)
                
                        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
                
                        # # Plot the original image in the first subplot
                        # im1 = ax1.imshow(blur_gt_24)
                        # ax1.set_title('Input', fontsize=36)
                        # cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                        # # Plot the EMA sampled image in the second subplot
                        # im2 = ax2.imshow(blur_gt_peak)
                        # ax2.set_title('Target', fontsize=36)
                        # cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                        # # Plot the EMA sampled image in the second subplot
                        # im3 = ax3.imshow(blur_ema_samp)
                        # ax3.set_title('1 day prediction', fontsize=36)
                        # cbar3 = fig.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                        
                        # im4 = ax4.imshow(blur_ema_per)
                        # ax4.set_title('1 day Persistence', fontsize=36)
                        # cbar4 = fig.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.2)  # Horizontal colorbar for the first image
                        
                        # # Adjust the spacing between subplots
                        # fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
                
                        # plt.tight_layout()    
                        # plt.close()
                
                        # Size of the active region
                        
                        contour_24 = obtain_contour_all(blur_gt_24)
                        contour_peak = obtain_contour_all(blur_gt_peak)
                        contour_samp = obtain_contour_all(blur_ema_samp)
                        contour_per = obtain_contour_all(blur_ema_per)
                        
                        import cv2
                        
                        # List of contours and corresponding images
                        contours = [contour_24, contour_peak, contour_samp, contour_per]
                        images = [blur_gt_24, blur_gt_peak, blur_ema_samp, blur_ema_per]
                        
                        # Create subplot structure
                        # fig, axs = plt.subplots(1, 4, figsize=(15, 5))
                        titles = ['24 Hours', 'Target', 'Prediction', 'Persistence']
                        ar_area = []
                        
                        roi_gt_peak = np.squeeze(roi_gt_peak)  # or keep 3D and expand mask if needed
                        
                        for image, cont, title in zip(images, contours, titles):
                            # ax.imshow(image, cmap='gray')
                            # ax.set_title(title, fontsize=36)
                        
                            total_pixel_count = 0
                        
                            for contour in cont:
                                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                                cv2.drawContours(mask, [contour], -1, 255, -1)
                        
                                masked_image = np.where(mask == 255, roi_gt_peak, 0)
                        
                                pixel_count = np.count_nonzero(mask)
                                total_pixel_count += pixel_count
                        
                                # ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='green', linewidth=3)
                        
                            ar_area.append(total_pixel_count)
                            # ax.set_xlabel(f'Total pixels: {total_pixel_count}', fontsize=25)
                        
                        # plt.tight_layout()
                        # plt.show()
    
                        # plt.close()
                        
                        
                        thrs_24 = 30 
                        thrs_gt = 30 
                        thrs_peak = 30 
                        thrs_pers = 30 
                    
                                  
                        # contour_24_white = obtain_contour_canny(blur_gt_24, thrs=thrs_24, white=True)
                        # contour_peak_white = obtain_contour_canny(blur_gt_peak, thrs=thrs_gt, white=True)
                        # contour_samp_white = obtain_contour_canny(blur_ema_samp, thrs=thrs_peak, white=True)
                        # contour_per_white = obtain_contour_canny(blur_ema_per, thrs=thrs_pers, white=True)
                                  
                        # contour_24_black = obtain_contour_canny(blur_gt_24, thrs=thrs_24, white=False)
                        # contour_peak_black = obtain_contour_canny(blur_gt_peak, thrs=thrs_gt, white=False)
                        # contour_samp_black = obtain_contour_canny(blur_ema_samp, thrs=thrs_peak, white=False)
                        # contour_per_black = obtain_contour_canny(blur_ema_per, thrs=thrs_pers, white=False)
                        
                        contour_24_white = obtain_contour(blur_gt_24, thrs=thrs_24, white=True)
                        contour_peak_white = obtain_contour(blur_gt_peak, thrs=thrs_gt, white=True)
                        contour_samp_white = obtain_contour(blur_ema_samp, thrs=thrs_peak, white=True)
                        contour_per_white = obtain_contour(blur_ema_per, thrs=thrs_pers, white=True)
                                  
                        contour_24_black = obtain_contour(blur_gt_24, thrs=thrs_24, white=False)
                        contour_peak_black = obtain_contour(blur_gt_peak, thrs=thrs_gt, white=False)
                        contour_samp_black = obtain_contour(blur_ema_samp, thrs=thrs_peak, white=False)
                        contour_per_black = obtain_contour(blur_ema_per, thrs=thrs_pers, white=False)
                        
                        
                        
                        # List of contours and corresponding images for both white and black
                        white_contours = [contour_24_white, contour_peak_white, contour_samp_white, contour_per_white]
                        black_contours = [contour_24_black, contour_peak_black, contour_samp_black, contour_per_black]
                        images = [blur_gt_24, blur_gt_peak, blur_ema_samp, blur_ema_per]
                        
                        # Create subplot structure
                        # fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Adjusted for two rows for white and black contours
                        
                        # Titles for subplots
                        titles = ['24 Hours', 'Target', 'Prediction', 'Persistence']
                        
                        white_areas = []
                        black_areas = []
                        white_mask_cont = []
                        black_mask_cont = []
                        # Iterate over each image and contour set
                        for o in range(4):
                            image = images[o]
                            white_cont = white_contours[o]
                            black_cont = black_contours[o]
                        
                            # Process white contours
                            white_mask = np.zeros(image.shape, dtype=np.uint8)
                            for contour in white_cont:
                                cv2.drawContours(white_mask, [contour], -1, 255, -1)
                            white_area = np.count_nonzero(white_mask)
                            white_mask_cont.append(white_mask)
                            white_areas.append(white_area)
                            
                            # Process black contours
                            black_mask = np.zeros(image.shape, dtype=np.uint8)
                            for contour in black_cont:
                                cv2.drawContours(black_mask, [contour], -1, 255, -1)
                            black_area = np.count_nonzero(black_mask)
                            black_mask_cont.append(black_mask)
                            black_areas.append(black_area)
                        
                            # Display the image with white contours
                        #     axs[0, o].imshow(image, cmap='gray')
                        #     axs[0, o].set_title(titles[o], fontsize=36)
                        #     for contour in white_cont:
                        #         axs[0, o].plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)  # Red for white contours
                        #     axs[0, o].set_xlabel(f'White pixels: {white_area}', fontsize=25)
                            
                        #     # Display the image with black contours
                        #     axs[1, o].imshow(image, cmap='gray')
                        #     for contour in black_cont:
                        #         axs[1, o].plot(contour[:, 0, 0], contour[:, 0, 1], 'b', linewidth=2)  # Blue for black contours
                        #     axs[1, o].set_xlabel(f'Black pixels: {black_area}', fontsize=25)
                        
                        # # Adjust layout
                        # plt.tight_layout()
                        # plt.show()
                        # plt.close()
                        
                        # ar_area = [a + b for a, b in zip(white_areas, black_areas)]
                        
                        contour_24_white = obtain_contour(blur_gt_24, thrs=thrs_24, white=True)
                        contour_peak_white = obtain_contour(blur_gt_peak, thrs=thrs_gt, white=True)
                        contour_samp_white = obtain_contour(blur_ema_samp, thrs=thrs_peak, white=True)
                        contour_per_white = obtain_contour(blur_ema_per, thrs=thrs_pers, white=True)
                                  
                        contour_24_black = obtain_contour(blur_gt_24, thrs=thrs_24, white=False)
                        contour_peak_black = obtain_contour(blur_gt_peak, thrs=thrs_gt, white=False)
                        contour_samp_black = obtain_contour(blur_ema_samp, thrs=thrs_peak, white=False)
                        contour_per_black = obtain_contour(blur_ema_per, thrs=thrs_pers, white=False)
                        
                        # Obtain Union and intersection for morphology determination
                        
                        int_pixels_samp, un_pixels_samp, jaccard_index_samp_white = comput_jaccard_index(contour_peak_white, contour_samp_white, blur_gt_peak)
                        
                        print(f"Number of pixels in the intersection: {int_pixels_samp}")
                        print(f"Number of pixels in the union: {un_pixels_samp}")
                        print(f"Jaccard index white: {jaccard_index_samp_white}")
                        
                        int_pixels_per, un_pixels_per, jaccard_index_per_white = comput_jaccard_index(contour_peak_white, contour_per_white, blur_gt_peak)
                        
                        print(f"Number of pixels in the intersection: {int_pixels_per}")
                        print(f"Number of pixels in the union: {un_pixels_per}")
                        print(f"Jaccard index white: {jaccard_index_per_white}")
                        
                        int_pixels_samp, un_pixels_samp, jaccard_index_samp_black = comput_jaccard_index(contour_peak_black, contour_samp_black, blur_gt_peak)
                        
                        print(f"Number of pixels in the intersection: {int_pixels_samp}")
                        print(f"Number of pixels in the union: {un_pixels_samp}")
                        print(f"Jaccard index black: {jaccard_index_samp_black}")
                        
                        int_pixels_per, un_pixels_per, jaccard_index_per_black = comput_jaccard_index(contour_peak_black, contour_per_black, blur_gt_peak)
                        
                        print(f"Number of pixels in the intersection: {int_pixels_per}")
                        print(f"Number of pixels in the union: {un_pixels_per}")
                        print(f"Jaccard index black: {jaccard_index_per_black}")
                        
                        jaccard_index_samp = np.mean([jaccard_index_samp_white, jaccard_index_samp_black])
                        jaccard_index_per = np.mean([jaccard_index_per_white, jaccard_index_per_black])
                        
                        print(f'Jaccard index samp: {jaccard_index_samp}')
                        print(f'Jaccard index pers: {jaccard_index_per}')
                        
                        
                        
                        # PLOT the intersection region
                        
                        # plot_intersection_contour(white_contours, black_contours, roi_gt_peak)
                                
                        
                        # COMPUTE POLARITY INVERSION LINE
                        
                        kernel = np.ones((4,4), np.uint8)  # Example kernel size, adjust as necessary
                        
                        ropi = []
                        
                        # Create subplot structure
                        # fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Adjusted for two rows for white and black contours
                        
                        # Titles for subplots
                        titles = ['24 Hours', 'Target', 'Prediction', 'Persistence']
                        
                        # Iterate over each image and contour set
                        for o in range(4):
                            image = images[o]
                            
                            # Dilate white and black masks
                            dilated_white_mask = cv2.dilate(white_mask_cont[o], kernel, iterations=1)
                            dilated_black_mask = cv2.dilate(black_mask_cont[o], kernel, iterations=1)
                            
                            # Find intersections (RoPI)
                            intersection_mask = cv2.bitwise_and(dilated_white_mask, dilated_black_mask)
                            
                            intersection_mask[intersection_mask == 255] = 1
        
                            
                            ropi.append(intersection_mask.reshape(40, 50, 1))
                            # Display the image with white contours
                            # axs[0, o].imshow(image, cmap='gray')
                            # axs[0, o].set_title(titles[o], fontsize=36)
                            
                            # # Display the image with black contours
                            # axs[1, o].imshow(intersection_mask, cmap='gray')
                        
                        # Adjust layout
                        # plt.tight_layout()
                        # plt.show()
                        # plt.close()
                        
                        # RoPI field strength magnitude filter
                        
                        ropi_filtered = [ropi[0]*blur_gt_24, ropi[1]*blur_gt_peak, ropi[2]*blur_ema_samp, ropi[3]*blur_ema_per]
                        
                        # fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Adjusted for two rows for white and black contours
                        
                        # Titles for subplots
                        titles = ['24 Hours', 'Target', 'Prediction', 'Persistence']
                        
                        for o in range(4):
                            image = ropi_filtered[o]
                            
        
                            
                            ropi.append(intersection_mask.reshape(40, 50, 1))
                            # Display the image with white contours
                        #     axs[0, o].imshow(image, cmap='gray')
                        #     axs[0, o].set_title(titles[o], fontsize=36)
                            
                        #     # Display the image with black contours
                        #     axs[1, o].imshow(ropi[o], cmap='gray')
                            
                        #     # Display the image with black contours
                        # # Adjust layout
                        # plt.tight_layout()
                        # plt.show()
                        # plt.close()
                        
                        import numpy as np
                        import matplotlib.pyplot as plt
                        from scipy.ndimage import label
                        
                    
                        # This function will find distinct regions in the mask and sum the pixel values in these regions
                        def sum_masked_regions(masked_image, pix_area):
                            # Label all the connected regions in the image
                            labeled_array, num_features = label(masked_image)
                            sums = []
                            
                            # Sum the pixel values for each region
                            for region in range(1, num_features + 1):
                                region_sum = np.sum(np.abs(masked_image[labeled_array == region]))
                                num_pix = len(masked_image[labeled_array == region])
                                total_area = pix_area * num_pix
                                sums.append(region_sum * total_area)
                            
                            norm_sum = []
                            for value in sums:
                                norm_sum.append(value/sum(sums))
                            
                            return norm_sum
                        
                        
                        
                        # Calculate the sum of pixel values for each masked image's regions
                        sums_per_image = [sum_masked_regions(image, pix_area) for image in ropi_filtered]
                        
                        # Magnetic field filtering
                        
                        labeled_array_list = [label(image)[0] for image in ropi_filtered]
                        
                        label_sum = [*zip(labeled_array_list, sums_per_image)]
                        
                        
                        def filter_labeled_regions_and_sums(labeled_array, sums):
                            # Get the indices of the regions that have sum less than 0.8
                            sums_to_remove_indices = [i for i, sum in enumerate(sums) if sum < 0.8]
                        
                            # Check if all sums are less than 0.8
                            if len(sums_to_remove_indices) == len(sums):
                                # If all sums are less than 0.8, find the max sum and its index
                                max_sum_index, max_sum = max(enumerate(sums), key=lambda x: x[1])
                                # Keep only the region with the max sum
                                for i, sum in enumerate(sums):
                                    if i != max_sum_index:
                                        labeled_array[labeled_array == i+1] = 0
                                # Update the filtered sums to only include the max sum
                                filtered_sums = [max_sum]
                            else:
                                # Set regions to zero in the labeled array for sums less than 0.8
                                for region in sums_to_remove_indices:
                                    labeled_array[labeled_array == region+1] = 0
                                # Filter sums list to remove values less than 0.8
                                filtered_sums = [sum for i, sum in enumerate(sums) if i not in sums_to_remove_indices]
                        
                            return labeled_array, filtered_sums
        
                        # Use the function on all image-label pairs
                        new_label_sum = [filter_labeled_regions_and_sums(label, sums) if len(sums) >= 2 else (label, sums) for label, sums in label_sum]
                                    
                        lab = [value for value, _ in new_label_sum]
                        # Plotting the masked images with the labels
                        # fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Adjust the size as needed
                        # for ax, image, labeled_array in zip(axes, ropi_filtered, lab):
                        #     ax.imshow(image, cmap='gray')
                        #     # Overlay the labeled regions
                        #     for region in range(1, labeled_array.max() + 1):
                        #         # Find the coordinates of the region's center
                        #         center = np.mean(np.argwhere(labeled_array == region), axis=0)
                        #         ax.text(center[1], center[0], str(region), color='red', ha='center', va='center')
                        #     ax.axis('off')
                        
                        # plt.tight_layout()
                        # plt.show()
                        # plt.close()
                        
                        from skimage.morphology import skeletonize
                        from scipy.ndimage import label
                        
                        pil = []
                        lab = [value for value, _ in new_label_sum]
                        images = [roi_gt_24, roi_gt_peak, roi_ema_samp, roi_ema_per]
                        
                        titles = ['Input', 'Ground Truth', 'Predicted', 'Persistence Model']
                        
                        lab = lab[1:]
                        images = images[1:]
                        
                        # fig, axs = plt.subplots(1, 3, figsize=(20, 20))  # Set up a 2x2 grid of plots
                        # axs = axs.flatten()  # Flatten the array to make indexing easier
                        
                        for h in range(len(lab)):
                            # Same processing as before
                            skeleton = skeletonize(lab[h])
                            pil.append(skeleton)
                            labeled_skeleton, num_features = label(skeleton)
                            final_PILs = np.zeros(labeled_skeleton.shape, dtype=bool)
                            
                            for feature in range(1, num_features + 1):
                                PIL_size = np.sum(labeled_skeleton == feature)
                                if PIL_size >= 0:
                                    final_PILs[labeled_skeleton == feature] = True
                            
                            skeleton_2d = np.squeeze(skeleton)
                            y, x = np.where(skeleton_2d != 0)
                            
                            # Plot on the current subplot
                        #     axs[h].imshow(images[h], cmap='gray')
                        #     axs[h].scatter(x, y, color='red')
                        #     axs[h].set_title(f'{titles[h]}')
                        #     axs[h].axis('off')  # Optionally remove the axis for a cleaner look
                        
                        # plt.tight_layout()  # Adjusts subplot params so that subplots fit in the figure area
                        # plt.show()
                        
                        for h in range(len(lab)):
                            # Assuming ropi_topR is a binary image representing top R% of magnetic flux RoPIs
                            skeleton = skeletonize(lab[h])
                            pil.append(skeleton)
                            # Label the skeleton image
                            labeled_skeleton, num_features = label(skeleton)
                            # Initiate an empty array with same shape for final PIL result
                            final_PILs = np.zeros(labeled_skeleton.shape, dtype=bool)
                            
                            # Loop through each isolated PIL and check if its size is greater or equal than Lth
                            for feature in range(1, num_features + 1):
                                PIL_size = np.sum(labeled_skeleton == feature)
                                if PIL_size >= 0:
                                    final_PILs[labeled_skeleton == feature] = True
                        
                            skeleton_2d = np.squeeze(skeleton)
                            # Get coordinates where skeleton is equal to 1
                            y, x = np.where(skeleton_2d != 0)
                            # Plot original image and skeleton lines
                            # plt.figure(figsize=(10,10))
                            # plt.imshow(images[h], cmap='gray')
                            # plt.scatter(x, y, color='red')    # use scatter plot to draw red dots on skeleton
                            # plt.title(f'{titles[h]}')
                            # plt.show()
                        #     # plt.close()
                        
                        pil_length = [np.count_nonzero(pil[0]), np.count_nonzero(pil[1]),
                                      np.count_nonzero(pil[2]), np.count_nonzero(pil[3])]
                        
                        
                        pv_ar_length_samp =  (((pil_length[1] - pil_length[2]))/pil_length[1]) * 100 if pil_length[1] != 0 else 0
                        pv_ar_length_pers =  (((pil_length[1] - pil_length[3]))/pil_length[1]) * 100 if pil_length[1] != 0 else 0
                        
                        
                    else:
                        # pv_ar_orientation_samp = -20000
                        # pv_ar_orientation_pers = -20000
                        ar_area = -20000
                        white_areas = -20000
                        black_areas = -20000
                        pil_length = []
                        pv_ar_length_samp = -20000
                        pv_ar_length_pers = -20000
                        jaccard_index_samp = -20000
                        jaccard_index_per = -20000
                        # angle_degrees_original = -20000
                        # angle_degrees_pred = -20000
                        # angle_degrees_pers = -20000
            
                
                ar_fluxes.append([total_unsigned_flux_ar, total_net_flux_ar, total_unsigned_flux_samp_ar,
                                  total_net_flux_samp_ar, perc_tot_flux, perc_net_flux,
                                  perc_var_ev_tot_ar, perc_var_ev_net_ar, ar_area, white_areas, black_areas,
                                  pil_length[1:],
                                  perc_tot_flux_per, perc_net_flux_per,
                                  pv_ar_length_samp, pv_ar_length_pers, jaccard_index_samp, jaccard_index_per])
            
            for k in range(len(ar_fluxes)):
                ar_tot['{}'.format(i)].append(ar_fluxes[k][0])
                ar_net['{}'.format(i)].append(ar_fluxes[k][1])
                ar_tot_samp['{}'.format(i)].append(ar_fluxes[k][2])
                ar_net_samp['{}'.format(i)].append(ar_fluxes[k][3])
                pv_ar_tot['{}'.format(i)].append(ar_fluxes[k][4])
                pv_ar_net['{}'.format(i)].append(ar_fluxes[k][5])
                pv_ar_tot_ev['{}'.format(i)].append(ar_fluxes[k][6])
                pv_ar_net_ev['{}'.format(i)].append(ar_fluxes[k][7])
                size_ar['{}'.format(i)].append(ar_fluxes[k][8])
                size_ar_positive['{}'.format(i)].append(ar_fluxes[k][9])
                size_ar_negative['{}'.format(i)].append(ar_fluxes[k][10])
                orientation['{}'.format(i)].append(ar_fluxes[k][11])
                # distance_center['{}'.format(i)].append(ar_fluxes[k][10])
                pv_ar_tot_pers['{}'.format(i)].append(ar_fluxes[k][12])
                pv_ar_net_pers['{}'.format(i)].append(ar_fluxes[k][13])
                pv_orientation_samp['{}'.format(i)].append(ar_fluxes[k][14])
                pv_orientation_pers['{}'.format(i)].append(ar_fluxes[k][15])
                jacc_samp['{}'.format(i)].append(ar_fluxes[k][16])
                jacc_per['{}'.format(i)].append(ar_fluxes[k][17])
                
    df = pd.DataFrame(columns=['PSNR', 'SSIM', 'LPIPS', 'Flux FD TOT', 'Flux FD NET', 'Flux FD TOT SAMP', 'Flux FD NET SAMP',
                               'Perc Var FD TOT', 'Perc Var FD NET','Perc Var FD TOT PERS', 'Perc Var FD NET PERS',
                               'Evolution FD TOT Samp', 'Evolution FD NET Samp', 'Evolution FD TOT Pers', 'Evolution FD NET Pers',
                               'Flux AR TOT', 'Flux AR NET',
                               'Flux AR TOT SAMP', 'Flux AR NET SAMP', 'Perc Var AR TOT', 'Perc Var AR NET',
                               'Perc Var AR TOT PERS', 'Perc Var AR NET PERS', 'Size AR', 'Size Positive', 'Size Negative',
                               'PIL_length',
                               'Perc Var length Samp', 'Perc Var length Pers', 'Jaccard_samp', 'Jaccard_per'])
    df['PSNR'] = psnr_gen
    df['SSIM'] = ssim_gen
    df['LPIPS'] = lpips_gen
    
    df['Flux FD TOT'] = full_disk_tot
    df['Flux FD NET'] = full_disk_net
    df['Flux FD TOT SAMP'] = full_disk_tot_samp
    df['Flux FD NET SAMP'] = full_disk_net_samp
    df['Perc Var FD TOT'] = pv_full_disk_tot
    df['Perc Var FD NET'] = pv_full_disk_net
    df['Perc Var FD TOT PERS'] = pv_full_disk_tot_pers
    df['Perc Var FD NET PERS'] = pv_full_disk_net_pers
    df['Evolution FD TOT Samp'] = pv_flux_ev_fd_tot
    df['Evolution FD NET Samp'] = pv_flux_ev_fd_net
    df['Evolution FD TOT Pers'] = pv_flux_ev_fd_tot_pers
    df['Evolution FD NET Pers'] = pv_flux_ev_fd_net_pers
    df['Flux AR TOT'] = [(value) for key, value in dict(list(ar_tot.items())[:]).items()]# ar_tot[:5].items()]
    df['Flux AR NET'] = [(value) for key, value in dict(list(ar_net.items())[:]).items()]#  ar_net[:5].items()]
    df['Flux AR TOT SAMP'] = [(value) for key, value in dict(list(ar_tot_samp.items())[:]).items()]#  ar_tot_samp[:5].items()]
    df['Flux AR NET SAMP'] = [(value) for key, value in dict(list(ar_net_samp.items())[:]).items()]# ar_net_samp[:5].items()]
    df['Perc Var AR TOT'] = [(value) for key, value in dict(list(pv_ar_tot.items())[:]).items()]# pv_ar_tot[:5].items()]
    df['Perc Var AR NET'] = [(value) for key, value in dict(list(pv_ar_net.items())[:]).items()]# pv_ar_net[:5].items()]
    df['Size AR'] = [(value) for key, value in dict(list(size_ar.items())[:]).items()]# size_ar[:5].items()]
    df['Size Positive'] = [(value) for key, value in dict(list(size_ar_positive.items())[:]).items()]# size_ar[:5].items()]
    df['Size Negative'] = [(value) for key, value in dict(list(size_ar_negative.items())[:]).items()]# size_ar[:5].items()]
    df['PIL_length'] = [(value) for key, value in dict(list(orientation.items())[:]).items()]# orientation[:5].items()]
    df['Perc Var AR TOT PERS'] = [(value) for key, value in dict(list(pv_ar_tot_pers.items())[:]).items()]# pv_ar_tot[:5].items()]
    df['Perc Var AR NET PERS'] = [(value) for key, value in dict(list(pv_ar_net_pers.items())[:]).items()]# pv_ar_net[:5].items()]
    df['Perc Var length Samp'] = [(value) for key, value in dict(list(pv_orientation_samp.items())[:]).items()]
    df['Perc Var length Pers'] = [(value) for key, value in dict(list(pv_orientation_pers.items())[:]).items()]
    df['Jaccard_samp'] = [(value) for key, value in dict(list(jacc_samp.items())[:]).items()]
    df['Jaccard_per'] = [(value) for key, value in dict(list(jacc_per.items())[:]).items()]
    
    
    # df.to_csv(fr'\\10.35.146.35\data01\francesco\sdo_img2img\k_diffusion\sample_results\diff_unet_cond\df_tot_mag_metrics_V2_{o}.csv')
    df.to_csv(fr'/mnt/nas05/data01/francesco/sdo_img2img/k_diffusion/sample_results/diff_unet_cond/df_tot_mag_metrics_V2_{f}.csv')
    