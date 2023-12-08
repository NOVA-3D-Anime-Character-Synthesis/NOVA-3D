import torch
import torch.nn as nn

# timm [timm](https://github.com/rwightman/pytorch-image-models) 
# 需要pip install grad-cam timm`
from timm.models.layers import DropPath, trunc_normal_

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, **kwargs):
        super().__init__(1, 3, **kwargs)   #  3为通道数，你要是改代码的时候，不仅要改95行中dim的大小，也别忘了这一行改一下！


class PoolAttn(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, dim=256, norm_layer=GroupNorm):
        super().__init__()
        self.patch_pool1 = nn.AdaptiveAvgPool2d((None, 4)) # [8,3,512,512]->[8,3,512,4]
        self.patch_pool2 = nn.AdaptiveAvgPool2d((4, None)) # [8,3,512,512]->[8,3,4,512]

        self.embdim_pool1 = nn.AdaptiveAvgPool2d((None, 4))
        self.embdim_pool2 = nn.AdaptiveAvgPool2d((4, None))

        # self.act = act_layer()
        self.norm = norm_layer()
        # self.proj = nn.Conv2d(dim,dim,1)
        self.proj0 = nn.Conv2d(int(dim), int(dim), 3, 1, 1, bias=True, groups=dim)
        self.proj1 = nn.Conv2d(int(dim), int(dim), 3, 1, 1, bias=True, groups=dim)
        self.proj2 = nn.Conv2d(int(dim), int(dim), 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        # 40~45：对channel平面上的长和宽进行pooling操作
        B, C, H, W = x.shape # [8,3,512,512]
        x_patch_attn1 = self.patch_pool1(x) # torch.Size([8, 3, 512, 4])
        x_patch_attn2 = self.patch_pool2(x) #torch.Size([8, 3, 4, 512])
        x_patch_attn = x_patch_attn1 @ x_patch_attn2 # [8,3,512,512]
        x_patch_attn = self.proj0(x_patch_attn)

        # 47~51：沿着一维channel数量进行attention，但是channel一维标量无法进行attention，所以需要把一维标量变换出二维对象。
        x1 = x.contiguous().view(B, C, H * W).transpose(1, 2).contiguous().view(B, H * W, 3, -1) # torch.Size([8, 512*512, 3, 1])
        x_embdim_attn1 = self.embdim_pool1(x1)  # torch.Size([8, 512*512, 3, 4])
        x_embdim_attn2 = self.embdim_pool2(x1)  # torch.Size([8, 512*512, 4, 1])
        x_embdim_attn = x_embdim_attn1 @ x_embdim_attn2

        x_embdim_attn = x_embdim_attn.contiguous().view(B, H * W, C).transpose(1, 2).contiguous().view(B, C, H, W)
        x_embdim_attn = self.proj1(x_embdim_attn)

        x_out = self.norm(x_patch_attn + x_embdim_attn)
        x_out = self.proj2(x_out)
        return x_out


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x





class GOAEncoder(nn.Module):
    def __init__(self, encoder_config_path, batch_size=4, dim=3, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()
        self.batch_size = batch_size
        self.norm1 = norm_layer()
        # self.token_mixer = Pooling(pool_size=pool_size)  这个是来作消融实验时候用的
        self.token_mixer = PoolAttn(dim=dim, norm_layer=norm_layer)
        self.norm2 = norm_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        #  下面的操作就是将原来代码特征变为【8，512，8，8】
        self.patch_pool = nn.AdaptiveAvgPool2d((None, 8))
        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, stride=1, padding=0)
    def forward(self, x):

        x = x + self.drop_path(      # （权重）矩阵 * 模块输出的特征 = 目的是学习更多的信息。
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
        
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
            * self.mlp(self.norm2(x)))

            # [8,3,512,512]->[8,3,512,512] 
            
        x = self.conv_layer(x) # [8,8,512,512]
        x = self.patch_pool(x).view(-1, 512, 8, 8) # [8,8,512,8] [self.batch_size,512,512
        return x

