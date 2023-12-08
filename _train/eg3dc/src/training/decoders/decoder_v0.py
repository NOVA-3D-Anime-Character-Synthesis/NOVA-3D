
import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer, MultiViewImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib
from camera_utils import LookAtPoseSampler
from torch import nn
import kornia

import numpy as np
import _util.util_v1 as uutil
import _util.pytorch_v1 as utorch
import _util.twodee_v1 as u2d

import _databacks.lustrous_renders_v1 as dklustr

from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        self.force_sigmoid = False

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions, force_sigmoid=None): # B, point_num, 3
        """
            sampled_features : B, point_num, 3
        """
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features
        force_sigmoid = force_sigmoid or self.force_sigmoid

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        if force_sigmoid:
            rgb = torch.sigmoid(x[...,1:])
        else:
            rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma} # color和density

    def set_force_sigmoid(self, state):
        self.force_sigmoid = state
        return self.force_sigmoid


class MultiViewOSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        self.force_sigmoid = False

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features*2, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, front_sampled_features,back_sampled_features, ray_directions, force_sigmoid=None): # B, point_num, 3
        """
            sampled_features : B, point_num, 3
            ray
        """
        # Aggregate features
        front_sampled_features = front_sampled_features.mean(1) # 
        back_sampled_features = back_sampled_features.mean(1)
        x = torch.cat([front_sampled_features,back_sampled_features],-1)
        force_sigmoid = force_sigmoid or self.force_sigmoid

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        if force_sigmoid:
            rgb = torch.sigmoid(x[...,1:])
        else:
            rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma} # color和density

    def set_force_sigmoid(self, state):
        self.force_sigmoid = state
        return self.force_sigmoid

######## pasting utils ########

def sample_orthofront(front_rgb, view_xyz, bw):
    frgb,vxyz = front_rgb, view_xyz
    vij = 1 - (vxyz[:,[1,0]]+bw/2)/bw
    ans = nn.functional.grid_sample(
        frgb.permute(0,1,3,2),
        vij.permute(0,2,3,1)*2-1,
        padding_mode='border',
        mode='bilinear',
    )
    return ans
def get_front_occlusion(G, x, out, offset=0.01):
    ro = out['image_xyz'] #.detach().clone()
    ro = ro * torch.tensor([-1,1,-1], device=ro.device)[None,:,None,None]
    ro[:,2,:,:] -= G.rendering_kwargs['ray_start'] - offset
    rd = torch.zeros_like(out['image_xyz'])
    rd[:,2,:,:] = 1
    xin = {**x}
    xin['paste_params'] = None
    xin['force_rays'] = {
        'ray_origins': ro,
        'ray_directions': rd,
    }
    ans = G.f(xin, return_more=True)['image_weights']
    return ans
def get_front_weights(G, x,):
    # ro = out['image_xyz'] #.detach().clone()
    # ro = ro * torch.tensor([-1,1,-1], device=ro.device)[None,:,None,None]
    # ro[:,2,:,:] -= G.rendering_kwargs['ray_start'] - offset
    # rd = torch.zeros_like(out['image_xyz'])
    # rd[:,2,:,:] = 1
    device = x['cond']['image_ortho_front'].device
    xin = {
        k: v for k,v in x.items()
        if k not in ['paste_params', 'camera_params', 'conditioning_params', 'force_rays']
    }
    # xin['paste_params'] = None
    # xin['force_rays'] = {
    #     'ray_origins': ro,
    #     'ray_directions': rd,
    # }
    xin['elevations'] = torch.zeros(1).to(device)
    xin['azimuths'] = torch.zeros(1).to(device)
    xin['fovs'] = -torch.ones(1).to(device)
    ans = G.f(xin, return_more=True)['image_weights']
    return ans
def get_xyz_discrepancy(xyz, rays):
    a = rays['ray_origins']
    n = rays['ray_directions']
    p = xyz * torch.tensor([-1,1,-1], device=xyz.device)[None,:,None,None]
    ans = ( (p-a) - ((p-a)*n).sum(dim=1,keepdims=True) * n ).norm(2, dim=1, keepdim=True)
    return ans

def paste_front(
        G, x, out, mode='default',
        thresh_weight=0.95,
        thresh_edges=0.02,
        thresh_occ=0.05, offset_occ=0.01,
        thresh_dxyz=0.01,
        front_weight_erosion=0,
        grad_sample=False,
        force_image=None,
        **kwargs,
        ):
    view_xyz = out['image_xyz']
    view_alpha = out['image_weights']
    front_rgb = x['cond']['image_ortho_front']
    
    # masks
    with torch.no_grad():
        # mask visible weights
        wmask = (nn.functional.interpolate(
            out['image_weights'],
            front_rgb.shape[-1],
            mode='bilinear',
        )>thresh_weight).float()
        
        # mask deep crevaces
        smask = kornia.filters.sobel(nn.functional.interpolate(
            out['image_xyz'],
            front_rgb.shape[-1],
            mode='bilinear',
        )).norm(2,dim=1,keepdim=True)
        smask = (smask<thresh_edges).float()
        
        # mask occlusion from front
        fmask = (get_front_occlusion(G, x, out, offset=offset_occ)<thresh_occ).float()
        fmask = nn.functional.interpolate(fmask, front_rgb.shape[-1], mode='bilinear')

        # mask xyz discrepancies
        dmask = get_xyz_discrepancy(out['image_xyz'], x['force_rays'])
        dmask = nn.functional.interpolate(dmask, front_rgb.shape[-1], mode='nearest')
        dmask = (dmask<thresh_dxyz).float()

        # mask edges from front
        if front_weight_erosion>=1:
            frontw = get_front_weights(G, x)
            e = front_weight_erosion
            fwmask = kornia.morphology.erosion(
                (frontw>0.5).float(),
                torch.ones(e,e).to(out['image_weights'].device),
            )
            fwmask = sample_orthofront(
                fwmask,
                nn.functional.interpolate(view_xyz, front_rgb.shape[-1], mode='bilinear'),
                G.rendering_kwargs['box_warp'],
            )
            fwmask = nn.functional.interpolate(fwmask, front_rgb.shape[-1], mode='nearest')
        else:
            frontw = None
            fwmask = torch.ones(*dmask.shape).to(dmask.device)

        # composite
        mask = wmask*smask*fmask*dmask*fwmask

    # generate paste
    if force_image is None:
        tocopy = front_rgb if not x['normalize_images'] else front_rgb*2-1
    else:
        tocopy = force_image.t()[None,].to(mask.device)
    with (uutil.contextlib.nullcontext() if grad_sample else torch.no_grad()):
        paste = sample_orthofront(
            tocopy,
            nn.functional.interpolate(view_xyz, front_rgb.shape[-1], mode='bilinear'),
            G.rendering_kwargs['box_warp'],
        )
    ans = torch.lerp(out['image'], paste, mask)
    return uutil.Dict({
        'image': ans,
        'paste': paste,
        'mask': mask,
        'mask_weights': wmask,
        'mask_edges': smask,
        'mask_occ': fmask,
        'mask_dxyz': dmask,
        'mask_frontweight': fwmask,
        'frontweight': frontw,
    })




