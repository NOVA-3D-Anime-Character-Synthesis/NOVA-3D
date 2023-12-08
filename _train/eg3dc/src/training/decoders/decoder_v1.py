
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

    
class DirectionAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(DirectionAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(3, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, direction):
        batch_size = x.shape[0]
        
        # Linear transformation of input to query, key, and value
        query = self.query(direction)
        key = self.key(x)
        value = self.value(x)
        
        # Reshape query, key, and value for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scores using dot product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to value
        attention_output = torch.matmul(attention_weights, value)
        
        # Reshape attention output and apply linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(attention_output)
        
        return output

def func_x(x):
    return x
def func_pos(x, p_fn, freq):
    return p_fn(x*freq)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs if kwargs!=None else {}
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(partial(func_x))
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(partial(func_pos, p_fn=p_fn, freq=freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs, num_freqs, include_input=True,log_sampling=True):
        self.kwargs['input_dims'] = 3
        self.kwargs['include_input'] = include_input
        self.kwargs['max_freq_log2'] = num_freqs - 1 
        self.kwargs['num_freqs'] = num_freqs
        self.kwargs['periodic_fns'] = [torch.sin,torch.cos]
        self.kwargs['log_sampling'] = log_sampling
        self.create_embedding_fn()
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class CameraAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(CameraAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear((90+25+25), d_model) # front,back camera parameter; query point coordinate and query point direction
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.fc = nn.Linear(d_model, d_model) 
        self.embeder = Embedder()

        self.front_camera = torch.tensor(np.array([  1.      ,   0.      ,   0.      ,   0.      ,   0.      ,
        -1.      ,   0.      ,   0.      ,   0.      ,   0.      ,
        -1.      ,   1.      ,   0.      ,   0.      ,   0.      ,
         1.      , -57.294327,   0.      ,   0.5     ,   0.      ,
       -57.294327,   0.5     ,   0.      ,   0.      ,   1.      ]),dtype=torch.float)
        self.back_camera = torch.tensor(np.array([-1.0000000e+00,  0.0000000e+00,  1.2246469e-16, -1.2246469e-16,
        0.0000000e+00, -1.0000000e+00,  0.0000000e+00,  0.0000000e+00,
        1.2246469e-16,  0.0000000e+00,  1.0000000e+00, -1.0000000e+00,
        0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00,
       -5.7294327e+01,  0.0000000e+00,  5.0000000e-01,  0.0000000e+00,
       -5.7294327e+01,  5.0000000e-01,  0.0000000e+00,  0.0000000e+00,
        1.0000000e+00]),dtype=torch.float)

    def forward(self, x, coordinate,direction):
        batch_size = x.shape[0] # 4*442368 = 1769472
        front_camera = self.front_camera.repeat((coordinate.shape[0],coordinate.shape[1],1)).to(x.device)
        back_camera = self.back_camera.repeat((coordinate.shape[0],coordinate.shape[1],1)).to(x.device)
        coordinate = self.embeder.embed(coordinate, 10) # [63]
        direction = self.embeder.embed(direction, 4) # [27]

        camera = torch.cat([coordinate,direction,front_camera,back_camera],-1)

        # Linear transformation of input to query, key, and value
        query = self.query(camera) # torch.Size([4, 442368, 16])
        key = self.key(x)
        value = self.value(x)
        
        # Reshape query, key, and value for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scores using dot product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to value
        attention_output = torch.matmul(attention_weights, value)
        
        # Reshape attention output and apply linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(attention_output)
        
        return output

class MultiViewOSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        self.force_sigmoid = False
        self.attention = DirectionAttention(n_features, num_heads=1)
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, front_sampled_features,back_sampled_features, ray_directions, force_sigmoid=None): # B, point_num, 3
        """
            front_sampled_features:
            back_sampled_features:
            sampled_features : B, point_num, 3
            ray_directions
        """
        # Aggregate features
        front_sampled_features = front_sampled_features.mean(1) # 
        back_sampled_features = back_sampled_features.mean(1)
        
        N, M, C = front_sampled_features.shape
        front_sampled_features = front_sampled_features.view(N*M, C)
        front_sampled_features = self.attention(front_sampled_features,ray_directions) 
        front_sampled_features = front_sampled_features.view(N, M, -1)
        N, M, C = back_sampled_features.shape
        back_sampled_features = back_sampled_features.view(N*M, C)
        back_sampled_features = self.attention(back_sampled_features, ray_directions)
        back_sampled_features = back_sampled_features.view(N, M, -1)
        # cross attention：direction
        # attention = cross_attention(front_sampled_features, back_sampled_features, ray_directions)
        x = front_sampled_features + back_sampled_features
        
        # x = torch.cat([front_sampled_features,back_sampled_features],-1)
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




