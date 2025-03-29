# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 09:44:22 2025

@author: pio-r
"""

import argparse
import os
from copy import deepcopy
import json
from pathlib import Path
import time

import accelerate
import safetensors.torch as safetorch
import torch
import torch._dynamo
from torch import optim
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import k_diffusion as K
import pandas as pd
from dataset_functions import PairedFITSDataset, get_default_transforms, save_images
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from util import generate_samples

def main():
            
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=1,
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
    p.add_argument('--dir-name', type=str, default='256_6to_1_unetcond_01',
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
    p.add_argument('--use_wandb', type=str, default=False,
                   help='Use wandb')
    p.add_argument('--wandb-entity', type=str,
                   help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                   help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                   help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-save-model', action='store_true',
                   help='save model to wandb')
    
    args = p.parse_args(["--config", "./configs/config_256x256_sdohmi.json"])
    
    dir_path_res = f"results_{args.dir_name}"
    dir_path_mdl = f"model_{args.dir_name}"
    
    if not os.path.exists(dir_path_res):
        os.makedirs(dir_path_res)
        
    if not os.path.exists(dir_path_mdl):
        os.makedirs(dir_path_mdl)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass
    
    config = K.config.load_config(args.config)
    model_config = config['model']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']
    
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']
    
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps, mixed_precision=args.mixed_precision)
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
    use_wandb = args.use_wandb # accelerator.is_main_process and args.wandb_project
    if use_wandb:
        import wandb
        log_config = vars(args)
        log_config['config'] = config
        log_config['parameters'] = K.utils.n_params(inner_model)
        wandb.init(project="sdo_img2img", entity="francescopio", config=log_config, save_code=True)
    
    lr = opt_config['lr'] if args.lr is None else args.lr
    groups = inner_model.param_groups(lr)
    if opt_config['type'] == 'adamw':
        opt = optim.AdamW(groups,
                          lr=lr,
                          betas=tuple(opt_config['betas']),
                          eps=opt_config['eps'],
                          weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'adam8bit':
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(groups,
                                 lr=lr,
                                 betas=tuple(opt_config['betas']),
                                 eps=opt_config['eps'],
                                 weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'sgd':
        opt = optim.SGD(groups,
                        lr=lr,
                        momentum=opt_config.get('momentum', 0.),
                        nesterov=opt_config.get('nesterov', False),
                        weight_decay=opt_config.get('weight_decay', 0.))
    else:
        raise ValueError('Invalid optimizer type')
    
    if sched_config['type'] == 'inverse':
        sched = K.utils.InverseLR(opt,
                                  inv_gamma=sched_config['inv_gamma'],
                                  power=sched_config['power'],
                                  warmup=sched_config['warmup'])
    elif sched_config['type'] == 'exponential':
        sched = K.utils.ExponentialLR(opt,
                                      num_steps=sched_config['num_steps'],
                                      decay=sched_config['decay'],
                                      warmup=sched_config['warmup'])
    elif sched_config['type'] == 'constant':
        sched = K.utils.ConstantLRWithWarmup(opt, warmup=sched_config['warmup'])
    else:
        raise ValueError('Invalid schedule type')
    
    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                  max_value=ema_sched_config['max_value'])
    ema_stats = {}
    
    
    # Create Dataloader
    
    # Input:
    # dir2 = '/mnt/nas05/astrodata01/sdo_hmi/mag_24_filtered/'
    dir2 = '//10.35.146.35/astrodata01/sdo_hmi/mag_24_filtered/'
    # dir171 = '/mnt/nas05/astrodata01/sdo_aia/data_171_24_filtered/'
    dir171 = '//10.35.146.35/astrodata01/sdo_aia/data_171_24_filtered/'
    # dir1600 = '/mnt/nas05/astrodata01/sdo_aia/data_1600_24_filtered/'
    dir1600 = '//10.35.146.35/astrodata01/sdo_aia/data_1600_24_filtered/'
    # dir1700 = '/mnt/nas05/astrodata01/sdo_aia/data_1700_24_filtered/'
    dir1700 = '//10.35.146.35/astrodata01/sdo_aia/data_1700_24_filtered/'
    # dirdoppl = '/mnt/nas05/astrodata01/sdo_hmi/dopplergram_24_filtered/'
    dirdoppl = '//10.35.146.35/astrodata01/sdo_hmi/dopplergram_24_filtered/'
    # dircont = '/mnt/nas05/astrodata01/sdo_hmi/continuum_24_filtered/'
    dircont = '//10.35.146.35/astrodata01/sdo_hmi/continuum_24_filtered/'
    
    # Flr
    # dir2_flr = '/mnt/nas05/astrodata01/sdo_hmi/mag_filtered/'
    dir2_flr = '//10.35.146.35/astrodata01/sdo_hmi/mag_filtered/'
    # dir171_flr = '/mnt/nas05/astrodata01/sdo_aia/data_171_filtered/'
    dir171_flr = '//10.35.146.35/astrodata01/sdo_aia/data_171_filtered/'
    # dir1600_flr = '/mnt/nas05/astrodata01/sdo_aia/data_1600_filtered/'
    dir1600_flr = '//10.35.146.35/astrodata01/sdo_aia/data_1600_filtered/'
    # dir1700_flr = '/mnt/nas05/astrodata01/sdo_aia/data_1700_filtered/'
    dir1700_flr = '//10.35.146.35/astrodata01/sdo_aia/data_1700_filtered/'
    # dirdoppl_flr = '/mnt/nas05/astrodata01/sdo_hmi/doppl_filter_effects_corr/'
    dirdoppl_flr = '//10.35.146.35/astrodata01/sdo_hmi/doppl_filter_effects_corr/'
    # dircont_flr = '/mnt/nas05/astrodata01/sdo_hmi/continuum_filtered/'
    dircont_flr = '//10.35.146.35/astrodata01/sdo_hmi/continuum_filtered/'
    
    transform_171_transf = get_default_transforms(
        target_size=256, channel="171A", mask_limb=False, radius_scale_factor=1.0).transforms
    
    transform_171 = transforms.Compose([
        *transform_171_transf,
    ])
    
    transform_1600_transf = get_default_transforms(
        target_size=256, channel="1600A", mask_limb=False, radius_scale_factor=1.0).transforms
    
    transform_1600 = transforms.Compose([
        *transform_1600_transf,
    ])
    
    transform_1700_transf = get_default_transforms(
        target_size=256, channel="1700A", mask_limb=False, radius_scale_factor=1.0).transforms
    
    transform_1700 = transforms.Compose([
        *transform_1700_transf,
    ])
    
    transform_doppl = transforms.Compose([
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomHorizontalFlip(p=1.0),  # This line adds a 90-degree rotation to the right
        transforms.Normalize(mean=(-2000), std=(4000)),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ])
    
    path_df_mag = r"\\10.35.146.35\data01\francesco\sdo_img2img/mag_24_jsoc_stats_arcsinh.csv"
    # path_df_mag = r"/mnt/nas05/data01/francesco/sdo_img2img/mag_24_jsoc_stats_arcsinh.csv"
    path_df_dpl = r"\\10.35.146.35\data01\francesco\sdo_img2img/continuum_24_jsoc_stats_arcsinh_v2.csv"
    # path_df_dpl = r"/mnt/nas05/data01/francesco/sdo_img2img/continuum_24_jsoc_stats_arcsinh_v2.csv"
    
    # df_mag = pd.read_csv(r"/mnt/nas05/data01/francesco/sdo_img2img/mag_24_jsoc_stats_arcsinh.csv")
    df_mag = pd.read_csv(r"\\10.35.146.35\data01\francesco\sdo_img2img/mag_24_jsoc_stats_arcsinh.csv")
    
    # df_cnt = pd.read_csv(r"/mnt/nas05/data01/francesco/sdo_img2img/continuum_24_jsoc_stats_arcsinh_v2.csv")
    df_cnt = pd.read_csv(r"\\10.35.146.35\data01\francesco\sdo_img2img/continuum_24_jsoc_stats_arcsinh_v2.csv")
    
    global_min = df_mag["Min"].min()
    global_max = df_mag["Max"].max()
    
    global_min_cont = df_cnt["Min"].min()
    global_max_cont = df_cnt["Max"].max()
    
    transform_2 = transforms.Compose([
        transforms.Normalize(mean=[global_min], std=[global_max - global_min]),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ])
    
    transform_cont = transforms.Compose([
        transforms.Normalize(mean=[global_min_cont], std=[global_max_cont - global_min_cont]),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ])
    
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
    
    
    train_dl = DataLoader(train_dataset, args.batch_size,) # shuffle=True, drop_last=False,
                                #num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    
    val_dl = DataLoader(val_dataset, args.batch_size,) # shuffle=True, drop_last=False,
                                #num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    
    print('Train loader and Valid loader are up!')
    
    inner_model, inner_model_ema, opt, train_dl = accelerator.prepare(inner_model, inner_model_ema, opt, train_dl)
    
    with torch.no_grad(), K.models.flops.flop_counter() as fc:
        x = torch.zeros([1, model_config['input_channels'], size[0], size[1]], device=device)
        sigma = torch.ones([1], device=device)
        extra_args = {}
        if getattr(unwrap(inner_model), "num_classes", 0):
            extra_args['class_cond'] = torch.zeros([1], dtype=torch.long, device=device)
        inner_model(x, sigma, **extra_args)
        if accelerator.is_main_process:
            print(f"Forward pass GFLOPs: {fc.flops / 1_000_000_000:,.3f}", flush=True)
    
    use_wandb = args.use_wandb
    
    if use_wandb:
        wandb.watch(inner_model)
    if accelerator.num_processes == 1:
        args.gns = False
    if args.gns:
        gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
        gns_stats = K.gns.GradientNoiseScale()
    else:
        gns_stats = None
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sample_density = K.config.make_sample_density(model_config)
    
    # Define the model 
    model = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)
    
    state_path = Path(f'{args.name}_state_{args.dir_name}.json')
    
    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        if not args.resume:
            state = json.load(open(state_path))
            ckpt_path = state['latest_checkpoint']
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        unwrap(model.inner_model).load_state_dict(ckpt['model'])
        unwrap(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        ema_stats = ckpt.get('ema_stats', ema_stats)
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        if args.gns and ckpt.get('gns_stats', None) is not None:
            gns_stats.load_state_dict(ckpt['gns_stats'])
        demo_gen.set_state(ckpt['demo_gen'])
        elapsed = ckpt.get('elapsed', 0.0)
    
        del ckpt
    else:
        epoch = 0
        step = 0
    
    if args.reset_ema:
        unwrap(model.inner_model).load_state_dict(unwrap(model_ema.inner_model).state_dict())
        ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                      max_value=ema_sched_config['max_value'])
        ema_stats = {}
    
    if args.resume_inference:
        if accelerator.is_main_process:
            print(f'Loading {args.resume_inference}...')
        ckpt = safetorch.load_file(args.resume_inference)
        unwrap(model.inner_model).load_state_dict(ckpt)
        unwrap(model_ema.inner_model).load_state_dict(ckpt)
        del ckpt
    
    def save():
        accelerator.wait_for_everyone()
        filename = f'/mnt/nas05/data01/francesco/sdo_img2img/k_diffusion/{dir_path_mdl}/{args.name}_{step:08}.pth'
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        inner_model = unwrap(model.inner_model)
        inner_model_ema = unwrap(model_ema.inner_model)
        obj = {
            'config': config,
            'model': inner_model.state_dict(),
            'model_ema': inner_model_ema.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step,
            'gns_stats': gns_stats.state_dict() if gns_stats is not None else None,
            'ema_stats': ema_stats,
            'demo_gen': demo_gen.get_state(),
            'elapsed': elapsed,
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': filename}
            json.dump(state_obj, open(state_path, 'w'))
        if args.wandb_save_model and use_wandb:
            wandb.save(filename)
    
    losses_since_last_print = []
    model = model.to(device)
    try:
        while True:
            # Training Loop
            epoch_train_loss = 0  # Track total training loss
            num_train_batches = len(train_dl)  # Number of batches
            model.train()
            for batch in tqdm(train_dl, smoothing=0.1, disable=not accelerator.is_main_process):
                if device.type == 'cuda':
                    start_timer = torch.cuda.Event(enable_timing=True)
                    end_timer = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_timer.record()
                else:
                    start_timer = time.time()
    
                with accelerator.accumulate(model):
                    inpt = torch.cat(list(batch[0].values()), dim=1).float().to(device)
                    trgt = batch[1]['dir2_flr'].float().to(device)
                    extra_args = {}
                    noise = torch.randn_like(trgt).to(device)
                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([trgt.shape[0]], device=device)
                    with K.models.checkpointing(args.checkpointing):
                        losses = model.loss(trgt, inpt, noise, sigma, mapping_cond=None, **extra_args)
                    loss = accelerator.gather(losses).mean().item()
                    losses_since_last_print.append(loss)
                    epoch_train_loss += loss  # Accumulate loss
                    accelerator.backward(losses.mean())
                    
                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, inpt.shape[0], inpt.shape[0] * accelerator.num_processes)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
    
                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
                    if accelerator.sync_gradients:
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()
    
                if device.type == 'cuda':
                    end_timer.record()
                    torch.cuda.synchronize()
                    elapsed += start_timer.elapsed_time(end_timer) / 1000
                else:
                    elapsed += time.time() - start_timer
    
                if step % 25 == 0:
                    loss_disp = sum(losses_since_last_print) / len(losses_since_last_print)
                    losses_since_last_print.clear()
                    avg_loss = ema_stats['loss']
                    if accelerator.is_main_process:
                        if args.gns:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}, gns: {gns_stats.get_gns():g}')
                        else:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}')
    
                step += 1

                if step == args.end_step:
                    if accelerator.is_main_process:
                        tqdm.write('Done!')
                    # return
            
            epoch_train_loss /= num_train_batches 
    
            # **Validation Loop (After Training, Before wandb Logging)**
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dl, desc="Validation", disable=not accelerator.is_main_process):
                    inpt = torch.cat(list(batch[0].values()), dim=1).float().to(device)
                    trgt = batch[1]['dir2_flr'].float().to(device)
                    noise = torch.randn_like(trgt).to(device)
            
                    sigma = sample_density([trgt.shape[0]], device=device).to(trgt.device)
                    extra_args = {}  # Ensure extra_args is defined
            
                    with K.models.checkpointing(args.checkpointing):
                        losses = model.loss_palette(inpt, noise, trgt, sigma, mapping_cond=None, **extra_args)
                    
                    val_loss += accelerator.gather(losses).mean().item()
            
            val_loss /= len(val_dl)
    
            # Print validation loss
            if accelerator.is_main_process:
                tqdm.write(f"Epoch {epoch}, Train Loss: {epoch_train_loss:.6f}, Validation Loss: {val_loss:.6f}")
    
            # Sampling
            
            if epoch % 5 == 0:
                
                # Test sampling 
                samples = generate_samples(model_ema, 1, device, inpt_cond=inpt[0], sampler="dpmpp_2m", sigma_min=sigma_min, sigma_max=sigma_max)
                
                img_rec_gen = (samples.clamp(-1, 1) + 1) / 2 # to be in [-1, 1], the plus 1 and the division by 2 is to bring back values to [0, 1]
                img_rec_gen = (img_rec_gen * 255).type(torch.uint8) # to bring in valid pixel range
                
                save_images(img_rec_gen[0][0].reshape(1, 1, 256, 256), os.path.join(f"results_{args.dir_name}", f"{epoch}_ema_cond.png"))
                true_img = inpt[0][0].reshape(1, 256, 256).permute(1, 2, 0).cpu().numpy()
                ema_samp = samples[0][0].detach().cpu().numpy()
                gt_peak = trgt[0].reshape(1, 256, 256).permute(1, 2, 0).cpu().numpy()
                
                # Create a figure with two subplots
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                
                # Plot the original image in the first subplot
                ax1.imshow(true_img, origin='lower', cmap='hmimag')
                ax1.set_title('24 before flare')
                # Plot the EMA sampled image in the second subplot
                ax2.imshow(gt_peak, origin='lower', cmap='hmimag')
                ax2.set_title('True flaring image')
                ax3.imshow(ema_samp, origin='lower', cmap='hmimag')
                ax3.set_title('Predicted flaring image')
                # Adjust the spacing between subplots
                plt.tight_layout()
                # Show the plot
                plt.show()
                
            # **wandb Logging (Now Includes Validation Loss)**
            if use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'loss': epoch_train_loss,
                    'val_loss': val_loss,  # ✅ Added validation loss
                    'lr': sched.get_last_lr()[0],
                    'ema_decay': ema_decay,
                    'Sampled images': wandb.Image(plt)
                }
                if args.gns:
                    log_dict['gradient_noise_scale'] = gns_stats.get_gns()
                wandb.log(log_dict, step=step)
                
                plt.close()
            save()
            epoch += 1  # Move to the next epoch
    
    except KeyboardInterrupt:
        pass
    
if __name__ == '__main__':
    main()
