# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math
import torch
import torch.nn as nn

from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils

import _util.util_v1 as uutil


def generate_planes(use_triplane=False):
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],[
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],[
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ] if not use_triplane else [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
    ], dtype=torch.float32) # 3,3,3

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3) # 转置矩阵
    projections = torch.bmm(coordinates, inv_planes)
    return projections #[..., :2]  # yichuns multiplane

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None, triplane_depth=1):
    # for q in ['plane_axes', 'plane_features', 'coordinates', 'mode', 'padding_mode', 'box_warp', 'triplane_depth']:
    #     uutil.pdump(eval(q), uutil.mkfile(f'/dev/shm/sample_from_planes/{q}.pkl'))
    if triplane_depth==1:  # normal eg3d
        assert padding_mode == 'zeros'
        N, n_planes, C, H, W = plane_features.shape
        _, M, _ = coordinates.shape
        plane_features = plane_features.view(N*n_planes, C, H, W)

        coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

        projected_coordinates = project_onto_planes(plane_axes, coordinates)[..., :2].unsqueeze(1) # 2
        output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        return output_features
    else:  # yichuns multiplane
        assert padding_mode == 'zeros'
        N, n_planes, CD, H, W = plane_features.shape # B,3,48,256,256
        _, M, _ = coordinates.shape
        C, D = CD // triplane_depth, triplane_depth # 3
        plane_features = plane_features.view(N*n_planes, C, D, H, W) # 3*B, 16, 3 ,256,256

        coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds, 将坐标映射到 box bound

        projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1).unsqueeze(2) # (N x n_planes) x 1 x 1 x M x 3
        output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 4, 3, 2, 1).reshape(N, n_planes, M, C) # 双线性采样
        return output_features

def sample_from_multiplanes(plane_axes, plane_features, coordinates, triplane_view, mode='bilinear', padding_mode='zeros', box_warp=None, triplane_depth=1):
    # for q in ['plane_axes', 'plane_features', 'coordinates', 'mode', 'padding_mode', 'box_warp', 'triplane_depth']:
    #     uutil.pdump(eval(q), uutil.mkfile(f'/dev/shm/sample_from_planes/{q}.pkl'))
    if triplane_depth==1:  # normal eg3d
        assert padding_mode == 'zeros'
        N, n_planes, C, H, W = plane_features.shape
        _, M, _ = coordinates.shape
        plane_features = plane_features.view(N*n_planes, C, H, W)

        coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds
        
        projected_coordinates = project_onto_planes(plane_axes, coordinates)[..., :2].unsqueeze(1)
        output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        return output_features
    else:  # yichuns multiplane
        assert padding_mode == 'zeros'
        N, n_planes, CD, H, W = plane_features.shape # B,3,48,256,256
        _, M, _ = coordinates.shape
        C, D = CD // triplane_depth, triplane_depth # 16, 3
        plane_features = plane_features.view(N*n_planes, C, D, H, W) # 3*B, 16, 3 ,256,256
        
        coordinates = coordinates * triplane_view # 广播机制
        coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds, 将坐标映射到

        projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1).unsqueeze(2) # (N x n_planes) x 1 x 1 x M x 3
        output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 4, 3, 2, 1).reshape(N, n_planes, M, C) # 双线性采样
        return output_features


# def project_onto_planes(planes, coordinates):
#     """
#     Does a projection of a 3D point onto a batch of 2D planes,
#     returning 2D plane coordinates.

#     Takes plane axes of shape n_planes, 3, 3
#     # Takes coordinates of shape N, M, 3
#     # returns projections of shape N*n_planes, M, 2
#     """
#     N, M, C = coordinates.shape
#     n_planes, _, _ = planes.shape
#     coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
#     inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
#     projections = torch.bmm(coordinates, inv_planes)
#     return projections

# def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
#     assert padding_mode == 'zeros'
#     N, n_planes, C, H, W = plane_features.shape
#     _, M, _ = coordinates.shape
#     plane_features = plane_features.view(N*n_planes, C, H, W)

#     coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

#     projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
#     output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
#     return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

def triplane_crop_mask(xyz_unformatted, thresh, boxwarp, allow_bottom=True):
    bw,tc = boxwarp, thresh
    device = xyz_unformatted.device
    # xyz = 0.5 * (xyz_unformatted+1) * torch.tensor([-1,1,-1]).to(device)[None,None,:]
    xyz = (xyz_unformatted) * torch.tensor([-1,1,-1]).to(device)[None,None,:]
    ans = (xyz[:,:,[0,2]].abs() <= (bw/2-tc)).all(dim=-1,keepdim=True)
    if allow_bottom:
        ans = ans | (
            (xyz[:,:,1:2] <= -(bw/2-tc)) &
            (xyz[:,:,[0,2]].abs() <= (bw/2-tc)).all(dim=-1,keepdim=True)
        )
    return ~ans
def cull_clouds_mask(denities, thresh):
    denities = torch.nn.functional.softplus(denities - 1) # activation bias of -1 makes things initialize better
    alpha = 1 - torch.exp(-denities)
    return alpha < thresh


class ImportanceRenderer(torch.nn.Module):
    def __init__(self, use_triplane=False):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes(use_triplane=use_triplane)

    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options, triplane_crop=None, cull_clouds=None, binarize_clouds=None):
        """
            input:
                planes:[B,3,48,256,256]
                decoder:OSGDecoder  
                ray_origins:[B, 9216,3] 
            return:
                rgb_final
                depth_final
                weights.sum(2)
                xyz_final
        """
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3) # x[-0.35,0.35],y[-0.35,0.35],z[0.5,-0.5]
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)


        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        xyz_coarse = out['xyz']
        if triplane_crop:
            # print(xyz_fine.amin(dim=(0,1)))
            # print(xyz_fine.amax(dim=(0,1)))
            cropmask = triplane_crop_mask(xyz_coarse, triplane_crop, rendering_options['box_warp'])
            densities_coarse[cropmask] = -1e3
        if binarize_clouds:
            ccmask = cull_clouds_mask(densities_coarse, binarize_clouds)
            densities_coarse[ccmask] = -1e3
            densities_coarse[~ccmask] = 1e3
        elif cull_clouds:
            ccmask = cull_clouds_mask(densities_coarse, cull_clouds)
            densities_coarse[ccmask] = -1e3
        # if triplane_crop:
        #     cropmask = triplane_crop_mask(xyz_coarse, triplane_crop, rendering_options['box_warp'])
        #     densities_coarse[cropmask] = -1e3
        # print(out['rgb'].shape)
        # print(out['sigma'].shape)
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1]) # [4,9216,48,32]
        xyz_coarse = xyz_coarse.reshape(batch_size, num_rays, samples_per_ray, xyz_coarse.shape[-1]) # [4, 9216, 48, 3]
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance'] # 48
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            
            # shu added: have all directions point to center (for ambient-lit)
            if rendering_options.get('directionless', False):
                sample_directions = -sample_coordinates/sample_coordinates.norm(dim=-1, keepdim=True).clip(0.01)
            else:
                sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            xyz_fine = out['xyz']
            if triplane_crop: # None
                # print(xyz_fine.amin(dim=(0,1)))
                # print(xyz_fine.amax(dim=(0,1)))
                cropmask = triplane_crop_mask(xyz_fine, triplane_crop, rendering_options['box_warp'])
                densities_fine[cropmask] = -1e3
            if binarize_clouds: # None
                ccmask = cull_clouds_mask(densities_fine, binarize_clouds)
                densities_fine[ccmask] = -1e3
                densities_fine[~ccmask] = 1e3
            elif cull_clouds: # None
                ccmask = cull_clouds_mask(densities_fine, cull_clouds)
                densities_fine[ccmask] = -1e3
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
            xyz_fine = xyz_fine.reshape(batch_size, num_rays, N_importance, xyz_fine.shape[-1])

            all_depths, all_colors, all_densities, all_xyz = self.unify_samples(
                depths_coarse, colors_coarse, densities_coarse, xyz_coarse,
                depths_fine, colors_fine, densities_fine, xyz_fine,
            )

            # Aggregate
            # rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
            all_colors_ = torch.cat([all_colors, all_xyz], dim=-1)
            rgb_final_, depth_final, weights = self.ray_marcher(all_colors_, all_densities, all_depths, rendering_options)
            rgb_final = rgb_final_[...,:-3]
            xyz_final = rgb_final_[...,-3:]
        else:
            # rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)
            colors_coarse_ = torch.cat([colors_coarse, xyz_coarse], dim=-1)
            rgb_final_, depth_final, weights = self.ray_marcher(colors_coarse_, densities_coarse, depths_coarse, rendering_options)
            rgb_final = rgb_final_[...,:-3]
            xyz_final = rgb_final_[...,-3:]

        # print(rgb_final.shape)
        # print(depth_final.shape)
        # print(weights.shape)
        return rgb_final, depth_final, weights.sum(2), xyz_final

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        sampled_features = sample_from_planes(
            self.plane_axes, planes, sample_coordinates,
            padding_mode='zeros', box_warp=options['box_warp'],
            triplane_depth=1 if 'triplane_depth' not in options else options['triplane_depth'],
        )
        # print(sample_coordinates.shape)
        # print(sampled_features.shape)

        out = decoder(sampled_features, sample_directions)
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        # print(out['rgb'].shape)
        out['xyz'] = sample_coordinates#.permute(0,2,1)[...,None]
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, xyz1, depths2, colors2, densities2, xyz2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_xyz = torch.cat([xyz1, xyz2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_xyz = torch.gather(all_xyz, -2, indices.expand(-1, -1, -1, all_xyz.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities, all_xyz

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples
    
class MultiViewImportanceRenderer(torch.nn.Module):
    def __init__(self, use_triplane=False, triplane_width=32):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes(use_triplane=use_triplane)
        self.weight = torch.nn.Parameter(torch.zeros(triplane_width*3))
        self.softmoid_layer = torch.nn.Sigmoid()
    def forward(self, front_planes, back_planes, decoder, ray_origins, ray_directions, rendering_options, triplane_crop=None, cull_clouds=None, binarize_clouds=None):
        self.plane_axes = self.plane_axes.to(ray_origins.device)
        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto': # 没有方向的情况下进行的
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)


        out = self.run_model(front_planes,back_planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        xyz_coarse = out['xyz']
        if triplane_crop:
            # print(xyz_fine.amin(dim=(0,1)))
            # print(xyz_fine.amax(dim=(0,1)))
            cropmask = triplane_crop_mask(xyz_coarse, triplane_crop, rendering_options['box_warp'])
            densities_coarse[cropmask] = -1e3
        if binarize_clouds:
            ccmask = cull_clouds_mask(densities_coarse, binarize_clouds)
            densities_coarse[ccmask] = -1e3
            densities_coarse[~ccmask] = 1e3
        elif cull_clouds:
            ccmask = cull_clouds_mask(densities_coarse, cull_clouds)
            densities_coarse[ccmask] = -1e3
        # if triplane_crop:
        #     cropmask = triplane_crop_mask(xyz_coarse, triplane_crop, rendering_options['box_warp'])
        #     densities_coarse[cropmask] = -1e3
        # print(out['rgb'].shape)
        # print(out['sigma'].shape)
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        xyz_coarse = xyz_coarse.reshape(batch_size, num_rays, samples_per_ray, xyz_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            
            # shu added: have all directions point to center (for ambient-lit)
            if rendering_options.get('directionless', False):
                sample_directions = -sample_coordinates/sample_coordinates.norm(dim=-1, keepdim=True).clip(0.01)
            else:
                sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)

            out = self.run_model(front_planes,back_planes, decoder, sample_coordinates, sample_directions, rendering_options) # 镜像
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            xyz_fine = out['xyz']
            if triplane_crop:
                # print(xyz_fine.amin(dim=(0,1)))
                # print(xyz_fine.amax(dim=(0,1)))
                cropmask = triplane_crop_mask(xyz_fine, triplane_crop, rendering_options['box_warp'])
                densities_fine[cropmask] = -1e3
            if binarize_clouds:
                ccmask = cull_clouds_mask(densities_fine, binarize_clouds)
                densities_fine[ccmask] = -1e3
                densities_fine[~ccmask] = 1e3
            elif cull_clouds:
                ccmask = cull_clouds_mask(densities_fine, cull_clouds)
                densities_fine[ccmask] = -1e3
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
            xyz_fine = xyz_fine.reshape(batch_size, num_rays, N_importance, xyz_fine.shape[-1])

            all_depths, all_colors, all_densities, all_xyz = self.unify_samples(
                depths_coarse, colors_coarse, densities_coarse, xyz_coarse,
                depths_fine, colors_fine, densities_fine, xyz_fine,
            )

            # Aggregate
            # rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
            all_colors_ = torch.cat([all_colors, all_xyz], dim=-1)
            rgb_final_, depth_final, weights = self.ray_marcher(all_colors_, all_densities, all_depths, rendering_options)
            rgb_final = rgb_final_[...,:-3]
            xyz_final = rgb_final_[...,-3:]
        else:
            # rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)
            colors_coarse_ = torch.cat([colors_coarse, xyz_coarse], dim=-1)
            rgb_final_, depth_final, weights = self.ray_marcher(colors_coarse_, densities_coarse, depths_coarse, rendering_options)
            rgb_final = rgb_final_[...,:-3]
            xyz_final = rgb_final_[...,-3:]

        # print(rgb_final.shape)
        # print(depth_final.shape)
        # print(weights.shape)
        return rgb_final, depth_final, weights.sum(2), xyz_final

    def run_model(self, front_planes, back_planes, decoder, sample_coordinates, sample_directions, options):
        # softmask = self.softmoid_layer(self.weight) # [16]
        # softmask = softmask.view(1,1,front_planes.shape[2],1,1)
        # planes = front_planes * softmask + back_planes * (1-softmask) 
        # front_sampled_features = sample_from_planes(
        #     self.plane_axes, front_planes, sample_coordinates,
        #     padding_mode='zeros', box_warp=options['box_warp'],
        #     triplane_depth=1 if 'triplane_depth' not in options else options['triplane_depth'],
        # )

        # back_sampled_features = sample_from_planes(
        #     self.plane_axes, back_planes, sample_coordinates,
        #     padding_mode='zeros', box_warp=options['box_warp'],
        #     triplane_depth=1 if 'triplane_depth' not in options else options['triplane_depth'],
        # )
        front_view = torch.ones(3).to(front_planes.device)
        back_view = torch.ones(3).to(front_planes.device)
        # print("triplane.shape:",front_planes.shape) # [4, 3, 48, 256, 256]
        back_view[0],back_view[1] = -1,-1
        front_sampled_features = sample_from_multiplanes(
            self.plane_axes, front_planes, sample_coordinates,triplane_view=front_view,
            padding_mode='zeros', box_warp=options['box_warp'],
            triplane_depth=1 if 'triplane_depth' not in options else options['triplane_depth'],
        )

        back_sampled_features = sample_from_multiplanes(
            self.plane_axes, back_planes, sample_coordinates,triplane_view=back_view,
            padding_mode='zeros', box_warp=options['box_warp'],
            triplane_depth=1 if 'triplane_depth' not in options else options['triplane_depth'],
        )
        
        out = decoder(front_sampled_features, back_sampled_features , sample_directions)
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        # print(out['rgb'].shape)
        out['xyz'] = sample_coordinates#.permute(0,2,1)[...,None] 采样点
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, xyz1, depths2, colors2, densities2, xyz2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_xyz = torch.cat([xyz1, xyz2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_xyz = torch.gather(all_xyz, -2, indices.expand(-1, -1, -1, all_xyz.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities, all_xyz

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples
