# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:53:07 2025

@author: pio-r
"""
import re
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from tqdm import tqdm
from astropy.io import fits
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from sunpy.visualization.colormaps import cm
from torchvision.transforms import Compose, Resize, Normalize, Lambda

# Define years for training and validation
train_years = {2013, 2015, 2017, 2018, 2019}
val_years = {2014, 2016, 2020}

rotate = transforms.Compose([
    
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomHorizontalFlip(p=1.0),])

to_tensor = transforms.ToTensor()

res = Resize((256, 256))

def save_images(img, path):
    V = []
    imgs = []
    for i in range(img.shape[0]):
        imgs.append(img[i])
    for images in imgs:
        images = images.permute(1, 2, 0)
        images = np.squeeze(images.cpu().numpy())
        # v = vis(images, channel_to_map(171))
        v = Image.fromarray(images)
        V.append(v)
    for value in V:
        value.save(path)

# Utility functions
def arcsinh_transform(image, alpha=0.001):
    """
    Apply arcsinh transformation to reduce peaks.
    Alpha controls the compression strength.
    """
    image = torch.tensor(image)
    return torch.asinh(alpha * image), alpha


def extract_year(filename):
    match = re.search(r"_(20\d{2})\.", filename)
    return int(match.group(1)) if match else None

def extract_timestamp(filename):
    match = re.search(r"_(20\d{2})\.(\d{2})\.(\d{2})_(\d{2})-(\d{2})-(\d{2})_TAI", filename)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return datetime(year, month, day, hour, minute, second)
    return None

def filter_files(directory, train):
    if not os.path.isdir(directory):
        print(f"[WARNING] Directory {directory} does not exist. Skipping...")
        return []
    
    fits_files = [f for f in os.listdir(directory) if f.endswith(".fits")]
    filtered_files = [os.path.join(directory, f) for f in fits_files if extract_year(f) in (train_years if train else val_years)]
    return sorted(filtered_files)

class PairedFITSDataset(Dataset):
    def __init__(self, dir2, dir_171, dir_1600, dir_1700, dir_doppl, dir_cont, 
                 dir2_flr, dir_171_flr, dir_1600_flr, dir_1700_flr, dir_doppl_flr, dir_cont_flr, 
                 train=True,  # Flag to determine if using training or validation set
                 transform2=None, transform_171=None, transform_1600=None, 
                 transform_1700=None, transform_doppl=None, transform_cont=None, df=None, df_doppl=None):
        super(PairedFITSDataset, self).__init__()

        # Store non-FLR directories
        self.directories = {
            "dir2": dir2,
            "dir_171": dir_171,
            "dir_1600": dir_1600,
            "dir_1700": dir_1700,
            "dir_doppl": dir_doppl,
            "dir_cont": dir_cont
        }

        # Store corresponding FLR directories
        self.flr_directories = {
            "dir2_flr": dir2_flr,
            "dir_171_flr": dir_171_flr,
            "dir_1600_flr": dir_1600_flr,
            "dir_1700_flr": dir_1700_flr,
            "dir_doppl_flr": dir_doppl_flr,
            "dir_cont_flr": dir_cont_flr
        }
        
        self.df = pd.read_csv(df)
        
        self.df_doppl = pd.read_csv(df_doppl)
        
        self.global_min = self.df["Min"].min()
        self.global_max = self.df["Max"].max()
        
        self.global_min_cont = self.df_doppl["Min"].min()
        self.global_max_cont = self.df_doppl["Max"].max()

        # Cache directory listings and timestamps
        self.filtered_files = {key: [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.fits')] for key, path in self.directories.items()}
        self.timestamps_dir2 = [self.extract_timestamp(os.path.basename(f)) for f in self.filtered_files["dir2"]]
        self.flr_files = {key: [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.fits')] for key, path in self.flr_directories.items()}
        # self.filtered_files = {key: filter_files(path, train) for key, path in self.directories.items()}
        # self.timestamps_dir2 = [extract_timestamp(os.path.basename(f)) for f in self.filtered_files["dir2"]]
        
        # # Match FLR directories (no optimizations here)
        # self.flr_files = {}
        # for flr_key, flr_path in tqdm(self.flr_directories.items()):
        #     flr_all_files = sorted([
        #         (extract_timestamp(f), os.path.join(flr_path, f))
        #         for f in os.listdir(flr_path) if f.endswith(".fits") and extract_timestamp(f)
        #     ])
            
        #     matched_files = []
        #     used_files = set()
            
        #     for t in tqdm(self.timestamps_dir2, desc=f"Matching {flr_key}"):
        #         closest_file = None
        #         min_time_diff = timedelta(days=1, minutes=45)

        #         for t3, f3 in flr_all_files:
        #             if f3 in used_files:
        #                 continue

        #             time_diff = abs(t3 - t)
        #             if time_diff <= min_time_diff:
        #                 min_time_diff = time_diff
        #                 closest_file = f3

        #         if closest_file:
        #             matched_files.append(closest_file)
        #             used_files.add(closest_file)
        #         else:
        #             print(f"[WARNING] No matching file found in {flr_key} for {t}")

        #     self.flr_files[flr_key] = matched_files
        
        # Ensure dataset alignment
        num_files = len(self.filtered_files["dir2"])
        for key, file_list in self.filtered_files.items():
            assert len(file_list) == num_files, f"Mismatch in {key}: {len(file_list)} vs {num_files}"
        for key, file_list in self.flr_files.items():
            assert len(file_list) == num_files, f"Mismatch in {key}: {len(file_list)} vs {num_files}"

        # Store transformations
        self.transform2 = transform2
        self.transform_171 = transform_171
        self.transform_1600 = transform_1600
        self.transform_1700 = transform_1700
        self.transform_doppl = transform_doppl
        self.transform_cont = transform_cont

    def __len__(self):
        return len(self.filtered_files["dir2"])
    
    def filter_files(self, directory, train):
        if not os.path.isdir(directory):
            print(f"[WARNING] Directory {directory} does not exist. Skipping...")
            return []
        
        fits_files = [f for f in os.listdir(directory) if f.endswith(".fits")]
        filtered_files = [os.path.join(directory, f) for f in fits_files if extract_year(f) in (train_years if train else val_years)]
        return sorted(filtered_files)

    def extract_timestamp(self, filename):
        match = re.search(r"_(20\d{2})\.(\d{2})\.(\d{2})_(\d{2})-(\d{2})-(\d{2})_TAI", filename)
        if match:
            return datetime(*map(int, match.groups()))
        return None
    
    def arcsinh_transform(self, image, alpha=0.001):
        """
        Apply arcsinh transformation to reduce peaks.
        Alpha controls the compression strength.
        """
        image = torch.tensor(image)
        return torch.asinh(alpha * image), alpha

    def __getitem__(self, idx):
        def load_fits(filepath):
            """Loads a FITS file and replaces NaNs with min values."""
            with fits.open(filepath) as hdul:
                data = hdul[1].data
                return np.nan_to_num(data, nan=np.nanmin(data))

        # Load images from non-FLR directories
        data = {key: load_fits(self.filtered_files[key][idx]) for key in self.filtered_files}

        # Load images from FLR directories
        data_flr = {key: load_fits(self.flr_files[key][idx]) for key in self.flr_files}

        # Convert to tensors
        for key in data:
            if "doppl" in key:
                data[key] = torch.clamp(to_tensor(data[key]), min=-2000, max=2000)
            elif key not in ["dir_171", "dir_1600", "dir_1700"]:  # Rotate all except these
                data[key] = rotate(to_tensor(data[key]))
            else:
                data[key] = to_tensor(data[key]).float()

        for key in data_flr:
            if "doppl" in key:
                data_flr[key] = torch.clamp(to_tensor(data_flr[key]), min=-2000, max=2000)
            elif key not in ["dir_171_flr", "dir_1600_flr", "dir_1700_flr"]:  # Rotate all except these
                data_flr[key] = rotate(to_tensor(data_flr[key]))
            else:
                data_flr[key] = to_tensor(data_flr[key]).float()

        # Apply transformations if provided
        if self.transform2:
            data["dir2"] = self.arcsinh_transform(res(data["dir2"]), alpha=0.1)[0]
            # data["dir2"] = 2 * ((data["dir2"] - self.global_min)/(self.global_max - self.global_min)) - 1
            data["dir2"] = self.transform2(data["dir2"])
            
            data_flr['dir2_flr'] = self.arcsinh_transform(res(data_flr["dir2_flr"]), alpha=0.1)[0]
            # data_flr['dir2_flr'] = 2 * ((data_flr['dir2_flr'] - self.global_min)/(self.global_max - self.global_min)) - 1
            data_flr["dir2_flr"] = self.transform2(data_flr["dir2_flr"])

        if self.transform_171:
            data["dir_171"] = self.transform_171(data["dir_171"])
            data_flr["dir_171_flr"] = self.transform_171(data_flr["dir_171_flr"])

        if self.transform_1600:
            data["dir_1600"] = self.transform_1600(data["dir_1600"])
            data_flr["dir_1600_flr"] = self.transform_1600(data_flr["dir_1600_flr"])

        if self.transform_1700:
            data["dir_1700"] = self.transform_1700(data["dir_1700"])
            data_flr["dir_1700_flr"] = self.transform_1700(data_flr["dir_1700_flr"])

        data['dir_doppl'] = res(data['dir_doppl'])
        data_flr['dir_doppl_flr'] = res(data_flr['dir_doppl_flr'])
        # data['dir_doppl'] = (data['dir_doppl'])/2000
        # data_flr['dir_doppl_flr'] = (data_flr['dir_doppl_flr'])/2000

        if self.transform_doppl:
            data["dir_doppl"] = self.transform_doppl(data["dir_doppl"])
            data_flr["dir_doppl_flr"] = self.transform_doppl(data_flr["dir_doppl_flr"])
            
        if self.transform_cont:
            data["dir_cont"] = self.arcsinh_transform(res(data["dir_cont"]), alpha=1e-06)[0]
            # data["dir_cont"] = 2 * ((data["dir_cont"] - self.global_min_cont)/(self.global_max_cont - self.global_min_cont)) - 1
            data["dir_cont"] = self.transform_cont(data["dir_cont"])
            data_flr["dir_cont_flr"] = self.arcsinh_transform(res(data_flr["dir_cont_flr"]), alpha=1e-06)[0]
            # data_flr["dir_cont_flr"] = 2 * ((data_flr["dir_cont_flr"] - self.global_min_cont)/(self.global_max_cont - self.global_min_cont)) - 1
            data_flr["dir_cont_flr"] = self.transform_cont(data_flr["dir_cont_flr"])
            
        with fits.open(self.filtered_files['dir2'][idx]) as hdul:
            data2 = hdul[1].data
            data2 = np.nan_to_num(data2, nan=np.nanmin(data2))
            # data2 = data2 / 10**3
            header_1 = hdul[1].header
            
        with fits.open(self.flr_files['dir2_flr'][idx]) as hdul:
            data3 = hdul[1].data
            data3 = np.nan_to_num(data2, nan=np.nanmin(data2))
            # data2 = data2 / 10**3
            header_3 = hdul[1].header

        return data, data_flr, header_1['T_OBS'], header_3['T_OBS'], header_1['CDELT1'], header_1['CDELT2'], header_1['CRPIX1'], header_1['CRPIX2'], header_1['RSUN_OBS'], self.filtered_files['dir2'][idx], self.flr_files['dir2_flr'][idx]
    
#Channels that correspond to HMI Magnetograms 
HMI_WL = ['Bx','By','Bz']
#A colormap for visualizing HMI
HMI_CM = LinearSegmentedColormap.from_list("bwrblack", ["#0000ff","#000000","#ff0000"])

def channel_to_map(name):
    """Given channel name, return colormap"""
    return HMI_CM if name in HMI_WL else cm.cmlist.get('sdoaia%d' % int(name))

def get_clip(X, name):
    """Given an image and the channel name, get the right clip"""
    return get_signed_pct_clip(X) if name in HMI_WL else get_pct_clip(X)

def get_pct_clip(X):
    """Return the 99.99th percentile"""
    return (0,np.quantile(X.ravel(),0.999))

def get_signed_pct_clip(X):
    """Return the 99.99th percentile by magnitude, but symmetrize it so 0 is in the middle"""
    v = np.quantile(np.abs(X.ravel()),0.999)
    return (-v,v)

def vis(X, cm, clip=None):
    """Given image, colormap, and a clipping, visualize results"""
    Xc = X
    if clip:
        Xc = np.clip((X-clip[0])/(clip[1]-clip[0]),0,1)
    Xcv = cm(Xc)
    return (Xcv[:,:,:3]*255).astype(np.uint8)

def show_grid(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    nrows=int(len(imgs)/4)
    ncols=4
    if nrows <= 0:
        nrows = 1
        ncols = len(imgs)
    fix, axs = plt.subplots(figsize=(24,24), ncols=ncols, nrows=nrows, squeeze=False)
    row = 0
    for i, img in enumerate(imgs):
        col = i % 4
        if i != 0 and i % 4 == 0:
            row = row + 1
        axs[row, col].imshow(img)
        axs[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.pause(10)
    plt.show()
    
def tensor_to_img(img, wave):
    V = []
    imgs = []
    for i in range(img.shape[0]):
        imgs.append(img[i])
    for images in imgs:
        images = images.permute(1, 2, 0)
        images = np.squeeze(images.cpu().numpy())
        v = vis(images, channel_to_map(wave))
        v = Image.fromarray(v)
        V.append(v)
    return show_grid(V)

CHANNEL_PREPROCESS = {
    "94A": {"min": 0.1, "max": 800, "scaling": "log10"},
    "131A": {"min": 0.7, "max": 1900, "scaling": "log10"},
    "171A": {"min": 5, "max": 3500, "scaling": "log10"},
    "193A": {"min": 20, "max": 5500, "scaling": "log10"},
    "211A": {"min": 7, "max": 3500, "scaling": "log10"},
    "304A": {"min": 0.1, "max": 3500, "scaling": "log10"},
    "335A": {"min": 0.4, "max": 1000, "scaling": "log10"},
    "1600A": {"min": 10, "max": 800, "scaling": "log10"},
    "1700A": {"min": 220, "max": 5000, "scaling": "log10"},
    "4500A": {"min": 4000, "max": 20000, "scaling": "log10"},
    "continuum": {"min": 0, "max": 65535, "scaling": None},
    "magnetogram": {"min": -250, "max": 250, "scaling": None},
    "Bx": {"min": -250, "max": 250, "scaling": None},
    "By": {"min": -250, "max": 250, "scaling": None},
    "Bz": {"min": -250, "max": 250, "scaling": None},
}


def get_default_transforms(target_size=256, channel="171", mask_limb=False, radius_scale_factor=1.0):
    """Returns a Transform which resizes 2D samples (1xHxW) to a target_size (1 x target_size x target_size)
    and then converts them to a pytorch tensor.

    Apply the normalization necessary for the SDO ML Dataset. Depending on the channel, it:
      - masks the limb with 0s
      - clips the "pixels" data in the predefined range (see above)
      - applies a log10() on the data
      - normalizes the data to the [0, 1] range
      - normalizes the data around 0 (standard scaling)

    Args:
        target_size (int, optional): [New spatial dimension of the input data]. Defaults to 256.
        channel (str, optional): [The SDO channel]. Defaults to 171.
        mask_limb (bool, optional): [Whether to mask the limb]. Defaults to False.
        radius_scale_factor (float, optional): [Allows to scale the radius that is used for masking the limb]. Defaults to 1.0.
    Returns:
        [Transform]
    """

    transforms = []

    # also refer to
    # https://pytorch.org/vision/stable/transforms.html
    # https://github.com/i4Ds/SDOBenchmark/blob/master/dataset/data/load.py#L363
    # and https://gitlab.com/jdonzallaz/solarnet-thesis/-/blob/master/solarnet/data/transforms.py
    preprocess_config = CHANNEL_PREPROCESS[channel]

    if preprocess_config["scaling"] == "log10":
        # TODO does it make sense to use vflip(x) in order to align the solar North as in JHelioviewer?
        # otherwise this has to be done during inference
        def lambda_transform(x): return torch.log10(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.log10(preprocess_config["min"])
        std = math.log10(preprocess_config["max"]) - \
            math.log10(preprocess_config["min"])
    elif preprocess_config["scaling"] == "sqrt":
        def lambda_transform(x): return torch.sqrt(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.sqrt(preprocess_config["min"])
        std = math.sqrt(preprocess_config["max"]) - \
            math.sqrt(preprocess_config["min"])
    else:
        def lambda_transform(x): return torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        )
        mean = preprocess_config["min"]
        std = preprocess_config["max"] - preprocess_config["min"]

    def limb_mask_transform(x):
        h, w = x.shape[1], x.shape[2]  # C x H x W

        # fixed disk size of Rs of 976 arcsec, pixel size in the scaled image (512x512) is ~4.8 arcsec
        original_resolution = 4096
        scaled_resolution = h
        pixel_size_original = 0.6
        radius_arcsec = 976.0
        radius = (radius_arcsec / pixel_size_original) / \
            original_resolution * scaled_resolution

        mask = create_circular_mask(
            h, w, radius=radius*radius_scale_factor)
        mask = torch.as_tensor(mask, device=x.device)
        return torch.where(mask, x, torch.tensor(0.0))

    if mask_limb:
        def mask_lambda_func(x):
            return limb_mask_transform(x)
        transforms.append(mask_lambda_func)
        # transforms.append(Lambda(lambda x: limb_mask_transform(x)))

    transforms.append(Resize((target_size, target_size)))
    # TODO find out if these transforms make sense
    def test_lambda_func(x):
        return lambda_transform(x)
    transforms.append(test_lambda_func)
    # transforms.append(Lambda(lambda x: lambda_transform(x)))
    transforms.append(Normalize(mean=[mean], std=[std]))
    # required to remove strange distribution of pixels (everything too bright)
    transforms.append(Normalize(mean=(0.5), std=(0.5)))

    return Compose(transforms)

def create_circular_mask(h, w, center=None, radius=None):
    # TODO investigate the use of a circular mask to prevent focussing to much on the limb
    # https://gitlab.com/jdonzallaz/solarnet-app/-/blob/master/src/prediction.py#L9

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))

    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def setup_logging(dir_name, run_name):
    """
    Setting up the folders for saving the model and the results

    """
    os.makedirs(f"models_{dir_name}", exist_ok=True)
    os.makedirs(f"results_{dir_name}", exist_ok=True)
    os.makedirs(os.path.join(f"models_{dir_name}", run_name), exist_ok=True)
    os.makedirs(os.path.join(f"results_{dir_name}", run_name), exist_ok=True)