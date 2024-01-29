# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."

Multi-view panic3d trial v1:
    

Code adapted from
"Alias-Free Generative Adversarial Networks"."""


import _util.util_v1 as uutil
import _util.twodee_v1 as u2d

import _train.eg3dc.util.eg3dc_v0 as ueg3d


import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop
# from training.training_loop import loops as training_loops
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    # exec(f'from training import {c.training_loop_version} as training_loop')
    # training_loop.training_loop(rank=rank, **c)
    getattr(training_loop, c.training_loop_version)(rank=rank, **c)
    # training_loops[c.training_loop_version](rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    # c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    # print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data_class, data_subset):
    try:
        # dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_kwargs = dnnlib.EasyDict(
            class_name=data_class,
            subset=data_subset,
        )
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# experiment name
@click.option('--name', type=str, required=True)
@click.option('--training_loop_version', type=str, required=True)
@click.option('--loss_module', type=str, required=True)
@click.option('--generator_module', type=str, required=True)
@click.option('--generator_backbone', type=str, required=True)
@click.option('--encoder_module', type=str, required=True)
@click.option('--discriminator_module', type=str, required=True)
@click.option('--cond_mode', type=str, required=True)
@click.option('--multi_view_cond_mode', type=str, default=None)
@click.option('--encoder_config',type=str,  metavar="FILE", help='path to config file', default='/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_train/swin_transformer/configs/swinv2.yaml')

@click.option('--lambda_gcond_lpips', type=float, default=10.0, required=True)
@click.option('--lambda_gcond_l1', type=float, default=1.0, required=True)
@click.option('--lambda_gcond_alpha_l2', type=float, default=0.0, required=False)
@click.option('--lambda_gcond_depth_l2', type=float, default=0.0, required=False)

@click.option('--lambda_gcond_sides_lpips', type=float, default=0.0, required=False)
@click.option('--lambda_gcond_sides_l1', type=float, default=0.0, required=False)
@click.option('--lambda_gcond_sides_alpha_l2', type=float, default=0.0, required=False)
@click.option('--lambda_gcond_sides_depth_l2', type=float, default=0.0, required=False)

@click.option('--lambda_gcond_back_lpips', type=float, default=0.0, required=False)
@click.option('--lambda_gcond_back_l1', type=float, default=0.0, required=False)
@click.option('--lambda_gcond_back_alpha_l2', type=float, default=0.0, required=False)
@click.option('--lambda_gcond_back_depth_l2', type=float, default=0.0, required=False)

@click.option('--lambda_gcond_rand_lpips', type=float, default=0.0, required=False)
@click.option('--lambda_gcond_rand_l1', type=float, default=0.0, required=False)
@click.option('--lambda_gcond_rand_alpha_l2', type=float, default=0.0, required=False)
@click.option('--lambda_gcond_rand_depth_l2', type=float, default=0.0, required=False)
# @click.option('--lambda_adv_g', type=float, default=1.0)
# @click.option('--lambda_adv_d', type=float, default=1.0)

@click.option('--lossmask_mode_adv', type=str, default='none', required=False)
@click.option('--lossmask_mode_recon', type=str, default='none', required=False)
@click.option('--lambda_recon_lpips', type=float, default=0.0, required=False)
@click.option('--lambda_recon_l1', type=float, default=0.0, required=False)
@click.option('--lambda_recon_alpha_l2', type=float, default=0.0, required=False)
@click.option('--lambda_recon_depth_l2', type=float, default=0.0, required=False)
@click.option('--seg_resolution',    help='Resolution of masks for discriminator.', metavar='INT', type=click.IntRange(min=0), default=128, required=False, show_default=True)
@click.option('--seg_channels',    help='Channels of masks for discriminator.', metavar='INT', type=click.IntRange(min=1), default=1, required=False, show_default=True)


@click.option('--paste_params_mode', type=str, default='none', required=False)



# data subset
@click.option('--data_subset', type=str, required=True)

# model options
@click.option('--triplane_depth', metavar='INT', required=False, default=1,)
@click.option('--triplane_width', metavar='INT', required=False, default=32,)
@click.option('--backbone_resolution', metavar='INT', required=False, default=256,)
@click.option('--use_triplane', metavar='INT', required=False, default=1,)
@click.option('--tanh_rgb_output', metavar='INT', required=False, default=0,)
@click.option('--resume_discrim', metavar='[PATH|URL]', type=str, default='')


# Required.
# @click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
# @click.option('--cfg',          help='Base configuration',                                      type=str, required=True)
# @click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)
@click.option('--seg_gamma',    help='R1 regularization seg weight', metavar='FLOAT',           type=click.FloatRange(min=0), required=False)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='noaug', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)


# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase_g',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cbase_d',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax_g',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--cmax_d',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0)) 
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True) # 2*e-4
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
# @click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

# @click.option('--sr_module',    help='Superresolution module', metavar='STR',  type=str, required=True)
@click.option('--neural_rendering_resolution_initial', help='Resolution to render at', metavar='INT',  type=click.IntRange(min=1), default=64, required=False)
@click.option('--neural_rendering_resolution_final', help='Final resolution to render at, if blending', metavar='INT',  type=click.IntRange(min=1), required=False, default=64)
@click.option('--neural_rendering_resolution_fade_kimg', help='Kimg to blend resolution over', metavar='INT',  type=click.IntRange(min=0), required=False, default=1000, show_default=True)

@click.option('--blur_fade_kimg', help='Blur over how many', metavar='INT',  type=click.IntRange(min=0), required=False, default=200)
@click.option('--gen_pose_cond', help='If true, enable generator pose conditioning.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--c-scale', help='Scale factor for generator pose conditioning.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=1)
@click.option('--c-noise', help='Add noise for generator pose conditioning.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)
@click.option('--gpc_reg_prob', help='Strength of swapping regularization. None means no generator pose conditioning, i.e. condition with zeros.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0.5)
@click.option('--gpc_reg_fade_kimg', help='Length of swapping prob fade', metavar='INT',  type=click.IntRange(min=0), required=False, default=1000)
@click.option('--disc_c_noise', help='Strength of discriminator pose conditioning regularization, in standard deviations.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)
@click.option('--sr_noise_mode', help='Type of noise for superresolution', metavar='STR',  type=click.Choice(['random', 'none']), required=False, default='none')
@click.option('--resume_blur', help='Enable to blur even on resume', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--sr_num_fp16_res',    help='Number of fp16 layers in superresolution', metavar='INT', type=click.IntRange(min=0), default=4, required=False, show_default=True)
@click.option('--g_num_fp16_res',    help='Number of fp16 layers in generator', metavar='INT', type=click.IntRange(min=0), default=0, required=False, show_default=True)
@click.option('--d_num_fp16_res',    help='Number of fp16 layers in discriminator', metavar='INT', type=click.IntRange(min=0), default=4, required=False, show_default=True)
@click.option('--sr_first_cutoff',    help='First cutoff for AF superresolution', metavar='INT', type=click.IntRange(min=2), default=2, required=False, show_default=True)
@click.option('--sr_first_stopband',    help='First cutoff for AF superresolution', metavar='FLOAT', type=click.FloatRange(min=2), default=2**2.1, required=False, show_default=True)
@click.option('--style_mixing_prob',    help='Style-mixing regularization probability for training.', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0, required=False, show_default=True)
@click.option('--sr-module',    help='Superresolution module override', metavar='STR',  type=str, required=False, default=None)
@click.option('--density_reg',    help='Density regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.25, required=False, show_default=True)
@click.option('--density_reg_every',    help='lazy density reg', metavar='int', type=click.FloatRange(min=1), default=4, required=False, show_default=True)
@click.option('--density_reg_p_dist',    help='density regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.004, required=False, show_default=True)
@click.option('--reg_type', help='Type of regularization', metavar='STR',  type=click.Choice(['l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed', 'total-variation']), required=False, default='l1')
@click.option('--decoder_lr_mul',    help='decoder learning rate multiplier.', metavar='FLOAT', type=click.FloatRange(min=0), default=1, required=False, show_default=True)
@click.option('--sr_channels_hidden', metavar='INT', type=click.IntRange(min=1), default=256, show_default=True)

def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    opts.outdir = f'./_train/eg3dc/runs/{opts.name}'
    opts.cfg = opts.name.split('_')[0]
    opts.data_class = f'_train.eg3dc.datasets.{opts.name.split("_")[0]}.DatasetWrapper'
    # print(kwargs['name'])
    # print(kwargs['cond_mode'])
    # print(kwargs['wtf'])
    # print(kwargs['lambda_Gwtf'])
    # print(kwargs['lambda_Gcond_lpips'])

    # resume hack (existing always overrules)
    def _get_resume(name, default):
        if default == "None":
            return None
        rdn = './_train/eg3dc/runs'
        mfn = 'metric-fid50k_full.jsonl'
        # fig = plt.figure(figsize=(10,8))
        # for dn in runs:
        dn = name
        # if rdn[0]=='_': continue
        # if not os.path.isdir(f'{rdn}/{dn}'): continue
        assert os.path.isdir(f'{rdn}/{dn}')
        cnt = 0
        data = []
        for ndn in sorted(os.listdir(f'{rdn}/{dn}')):
            if not ndn.isnumeric(): continue
            if not mfn in os.listdir(f'{rdn}/{dn}/{ndn}'): continue
            app = [
                (
                    cnt+int(j['snapshot_pkl'].split('-')[-1][:-len('.pkl')]),
                    j['results']['fid50k_full'],
                    j['snapshot_pkl'],
                    f'{rdn}/{dn}/{ndn}/{j["snapshot_pkl"]}',
                )
                for j in [
                    json.loads(line)
                    for line in uutil.read(f'{rdn}/{dn}/{ndn}/{mfn}').split('\n')
                    if line
                ]
            ]
            if len(app)==0: continue
            # plt.scatter(app[0][0], app[0][1], c='k', s=10)
            data.extend(app)
            cnt = data[-1][0]
        # use last available pkl
        for d in reversed(data):
            if os.path.isfile(d[-1]):
                return d[-1]
        return default
    opts.resume = _get_resume(opts.name, opts.resume)

    c = dnnlib.EasyDict() # Main config dict.
    if opts.multi_view_cond_mode == 'None':
        c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, cond_mode=opts.cond_mode, mapping_kwargs=dnnlib.EasyDict())
    else:
        c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, cond_mode=opts.cond_mode, mapping_kwargs=dnnlib.EasyDict(),multi_view_cond_mode=opts.multi_view_cond_mode)
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), cond_mode=opts.cond_mode, mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name=opts.loss_module)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data_class=opts.data_class, data_subset=opts.data_subset)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    # Hyperparameters & settings.
    c.training_loop_version = opts.training_loop_version
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = opts.cbase_g
    c.G_kwargs.channel_max = opts.cmax_g
    c.G_kwargs.mapping_kwargs.num_layers = opts.map_depth
    
    c.D_kwargs.channel_base = opts.cbase_d
    c.D_kwargs.channel_max = opts.cmax_d
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    if c.loss_kwargs.class_name=="training.loss.StyleGAN2LossMultiMaskOrthoCond":
        c.loss_kwargs.r1_gamma_seg = opts.seg_gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    c.G_kwargs.sr_channels_hidden = opts.sr_channels_hidden
    c.G_kwargs.triplane_width = opts.triplane_width
    c.G_kwargs.backbone = opts.generator_backbone
    c.G_kwargs.backbone_resolution = opts.backbone_resolution
    
    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    c.G_kwargs.class_name = opts.generator_module # 'training.triplane.TriPlaneGenerator' # 
    
    c.D_kwargs.class_name = opts.discriminator_module # 'training.dual_discriminator.DualDiscriminator' # 
    if c.D_kwargs.class_name == 'training.dual_discriminator.MaskDualDiscriminator':
        c.D_kwargs.seg_resolution = opts.seg_resolution
        c.D_kwargs.seg_channels = opts.seg_channels
    c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    c.loss_kwargs.filter_mode = 'antialiased' # Filter mode for raw images ['antialiased', 'none', float [0-1]]
    c.D_kwargs.disc_c_noise = opts.disc_c_noise # Regularization for discriminator pose conditioning

    if c.training_set_kwargs.resolution == 512:
        sr_module = 'training.superresolution.SuperresolutionHybrid8XDC'
    elif c.training_set_kwargs.resolution == 256:
        sr_module = 'training.superresolution.SuperresolutionHybrid4X'
    elif c.training_set_kwargs.resolution == 128:
        sr_module = 'training.superresolution.SuperresolutionHybrid2X'
    else:
        assert False, f"Unsupported resolution {c.training_set_kwargs.resolution}; make a new superresolution module"
    
    if opts.sr_module != None:
        sr_module = opts.sr_module
    if opts.encoder_module!="None":
        encoder_options = {
            'encoder':opts.encoder_module,
            'encoder_config': opts.encoder_config,
            'batch_gpu': c.batch_gpu,
        }
    else:
        encoder_options = {}
    rendering_options = {
        'image_resolution': c.training_set_kwargs.resolution,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'superresolution_module': sr_module,
        'c_gen_conditioning_zero': not opts.gen_pose_cond, # if true, fill generator pose conditioning label with dummy zero vector
        'gpc_reg_prob': opts.gpc_reg_prob if opts.gen_pose_cond else None,
        'c_scale': opts.c_scale, # mutliplier for generator pose conditioning label
        'superresolution_noise_mode': opts.sr_noise_mode, # [random or none], whether to inject pixel noise into super-resolution layers
        'density_reg': opts.density_reg, # strength of density regularization
        'density_reg_p_dist': opts.density_reg_p_dist, # distance at which to sample perturbed points for density regularization
        'reg_type': opts.reg_type, # for experimenting with variations on density regularization
        'decoder_lr_mul': opts.decoder_lr_mul, # learning rate multiplier for decoder
        'sr_antialias': True,

        'white_back': True,
        'triplane_depth': opts.triplane_depth,
        'use_triplane': opts.use_triplane,
        'tanh_rgb_output': opts.tanh_rgb_output,
    }

    if opts.cfg == 'ecrutileE':
        rendering_options.update({
            'box_warp': 0.7, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'ray_start': 0.5, # near point along each ray to start taking samples.
            'ray_end': 1.5, # far point along each ray to stop taking samples. 
            'distance': 1.0,
            
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'avg_camera_radius': 1.0, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0], # used only in the visualizer to control center of camera rotation.
        })
    # if opts.cfg == 'ffhq':
    #     rendering_options.update({
    #         'depth_resolution': 48, # number of uniform samples to take per ray.
    #         'depth_resolution_importance': 48, # number of importance samples to take per ray.
    #         'ray_start': 2.25, # near point along each ray to start taking samples.
    #         'ray_end': 3.3, # far point along each ray to stop taking samples. 
    #         'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
    #         'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
    #         'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
    #     })
    # elif opts.cfg == 'afhq':
    #     rendering_options.update({
    #         'depth_resolution': 48,
    #         'depth_resolution_importance': 48,
    #         'ray_start': 2.25,
    #         'ray_end': 3.3,
    #         'box_warp': 1,
    #         'avg_camera_radius': 2.7,
    #         'avg_camera_pivot': [0, 0, -0.06],
    #     })
    # elif opts.cfg == 'shapenet':
    #     rendering_options.update({
    #         'depth_resolution': 64,
    #         'depth_resolution_importance': 64,
    #         'ray_start': 0.1,
    #         'ray_end': 2.6,
    #         'box_warp': 1.6,
    #         'white_back': True,
    #         'avg_camera_radius': 1.7,
    #         'avg_camera_pivot': [0, 0, 0],
    #     })
    elif opts.cfg == 'multi':
        rendering_options.update({
            'box_warp': 0.7, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'ray_start': 0.5, # near point along each ray to start taking samples.
            'ray_end': 1.5, # far point along each ray to stop taking samples. 
            'distance': 1.0,
            
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'avg_camera_radius': 1.0, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0], # used only in the visualizer to control center of camera rotation.
        })
    elif opts.cfg == 'human':
        rendering_options.update({
            'box_warp': 2.1,# 0.7, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'ray_start': 2.5, # near point along each ray to start taking samples.
            'ray_end': 4.5, # far point along each ray to stop taking samples. 
            'distance': 3.5, # [1,-1]
            
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'avg_camera_radius': 1.0, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0], # used only in the visualizer to control center of camera rotation.
        })
    else:
        assert False, "Need to specify config"



    if opts.density_reg > 0:
        c.G_reg_interval = opts.density_reg_every # phase.interval
    c.G_kwargs.rendering_kwargs = rendering_options
    c.G_kwargs.encoder_kwargs = encoder_options
    c.G_kwargs.num_fp16_res = 0
    c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_fade_kimg = c.batch_size * opts.blur_fade_kimg / 32 # Fade out the blur during the first N kimg.

    c.loss_kwargs.gpc_reg_prob = opts.gpc_reg_prob if opts.gen_pose_cond else None
    c.loss_kwargs.gpc_reg_fade_kimg = opts.gpc_reg_fade_kimg
    c.loss_kwargs.dual_discrimination = True
    c.loss_kwargs.neural_rendering_resolution_initial = opts.neural_rendering_resolution_initial
    c.loss_kwargs.neural_rendering_resolution_final = opts.neural_rendering_resolution_final
    c.loss_kwargs.neural_rendering_resolution_fade_kimg = opts.neural_rendering_resolution_fade_kimg
    c.G_kwargs.sr_num_fp16_res = opts.sr_num_fp16_res

    c.G_kwargs.sr_kwargs = dnnlib.EasyDict(channel_base=opts.cbase_g, channel_max=opts.cmax_g, fused_modconv_default='inference_only')

    c.loss_kwargs.style_mixing_prob = opts.style_mixing_prob

    c.loss_kwargs.lambda_Gcond_lpips = opts.lambda_gcond_lpips
    c.loss_kwargs.lambda_Gcond_l1 = opts.lambda_gcond_l1
    c.loss_kwargs.lambda_Gcond_alpha_l2 = opts.lambda_gcond_alpha_l2
    c.loss_kwargs.lambda_Gcond_depth_l2 = opts.lambda_gcond_depth_l2

    c.loss_kwargs.lambda_Gcond_sides_lpips = opts.lambda_gcond_sides_lpips
    c.loss_kwargs.lambda_Gcond_sides_l1 = opts.lambda_gcond_sides_l1
    c.loss_kwargs.lambda_Gcond_sides_alpha_l2 = opts.lambda_gcond_sides_alpha_l2
    c.loss_kwargs.lambda_Gcond_sides_depth_l2 = opts.lambda_gcond_sides_depth_l2

    c.loss_kwargs.lambda_Gcond_back_lpips = opts.lambda_gcond_back_lpips
    c.loss_kwargs.lambda_Gcond_back_l1 = opts.lambda_gcond_back_l1
    c.loss_kwargs.lambda_Gcond_back_alpha_l2 = opts.lambda_gcond_back_alpha_l2
    c.loss_kwargs.lambda_Gcond_back_depth_l2 = opts.lambda_gcond_back_depth_l2

    c.loss_kwargs.lambda_Gcond_rand_lpips = opts.lambda_gcond_rand_lpips
    c.loss_kwargs.lambda_Gcond_rand_l1 = opts.lambda_gcond_rand_l1
    c.loss_kwargs.lambda_Gcond_rand_alpha_l2 = opts.lambda_gcond_rand_alpha_l2
    c.loss_kwargs.lambda_Gcond_rand_depth_l2 = opts.lambda_gcond_rand_depth_l2
    # c.loss_kwargs.lambda_adv_g = opts.lambda_adv_g
    # c.loss_kwargs.lambda_adv_d = opts.lambda_adv_d

    c.loss_kwargs.lossmask_mode_adv = opts.lossmask_mode_adv
    c.loss_kwargs.lossmask_mode_recon = opts.lossmask_mode_recon
    c.loss_kwargs.lambda_recon_lpips = opts.lambda_recon_lpips
    c.loss_kwargs.lambda_recon_l1 = opts.lambda_recon_l1
    c.loss_kwargs.lambda_recon_alpha_l2 = opts.lambda_recon_alpha_l2
    c.loss_kwargs.lambda_recon_depth_l2 = opts.lambda_recon_depth_l2

    c.loss_kwargs.paste_params_mode = opts.paste_params_mode

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        if not opts.resume_blur:
            c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
            c.loss_kwargs.gpc_reg_fade_kimg = 0 # Disable swapping rampup

    if opts.resume_discrim:
        c.resume_discrim_pkl = opts.resume_discrim
    else:
        c.resume_discrim_pkl = None

    # Performance-related toggles.
    # if opts.fp32:
    #     c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
    #     c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    c.G_kwargs.num_fp16_res = opts.g_num_fp16_res
    c.G_kwargs.conv_clamp = 256 if opts.g_num_fp16_res > 0 else None
    c.D_kwargs.num_fp16_res = opts.d_num_fp16_res
    c.D_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None

    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
