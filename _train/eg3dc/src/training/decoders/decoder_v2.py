# import sys
# sys.path.append('/HOME/zhouxinyi/panic3d_new')
# sys.path.append('/HOME/zhouxinyi/panic3d_new/_train/eg3dc/src')

import torch
import torch.nn as nn
import numpy as np
from viz.attention_widget import attention_draw
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

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t()) # out=x @ w.T + bias, w和b都是可学习参数
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation) # 4, 512
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

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
        # attention_draw(attention_weights=attention_weights)
        # Apply attention weights to value
        attention_output = torch.matmul(attention_weights, value)
        
        # Reshape attention output and apply linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(attention_output)
        
        return output
    

class CrossAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(CrossAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(3, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model // 2) # 转换回原来的维度
        # self.fc = nn.Linear(d_model, d_model) # Alternative

    def forward(self, x1, x2, direction):
        # x1: [4*9216*48, 16]
        # direction: [4*9216*48, 3]
        batch_size = x1.shape[0] # 4*9216*48
        x = torch.cat((x1,x2), dim=-1) # x1, x2同时作为K, V [4*9216*48, 32]
        
        # Linear transformation of input to query, key, and value
        query = self.query(direction) # [4*9216*48, 32]
        key = self.key(x) # [4*9216*48, 32]
        value = self.value(x) # [4*9216*48, 32]
        
        # Reshape query, key, and value for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Compute scores using dot product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1) # []
        # 三维热图可视化
        # attention_draw(attention_weights=attention_weights)
        
        # Apply attention weights to value
        attention_output = torch.matmul(attention_weights, value) #  [,8,8] [B*9216*48,1,1,32],前16为front，后16为back
        
        # Reshape attention output and apply linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(attention_output) 
        # Alternative 把前后两段加起来，维度变回16，再经过一个全连接层
        # output = attention_output[:,:x1.shape[1]] + attention_output[:,x1.shape[1]:]
        # output = self.fc(output) 
        
        return output


class MultiViewOSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        self.force_sigmoid = False
        # self.attention = DirectionAttention(n_features, num_heads=1)
        self.attention = CrossAttention(n_features * 2, num_heads=8) # 把两个方向的features拼接起来，维度*2
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, front_sampled_features,back_sampled_features, ray_directions, force_sigmoid=None): # B, point_num, 3
        """
            front_sampled_features : N, n_planes, M, C   N=4*9216 n_planes=3 M=48 C=16
            back_sampled_features : N, n_planes, M, C
            ray_directions : N*n_planes*M, 3 
        """
        # Aggregate features of 2 triplanes respectively
        # 求均值
        front_sampled_features = front_sampled_features.mean(1)
        back_sampled_features = back_sampled_features.mean(1)
        
        N, M, C = front_sampled_features.shape # [4*9216, 48, 16]
        front_sampled_features = front_sampled_features.view(N*M, C) # [4*9216*48, 16]
        # front_sampled_features = self.attention(front_sampled_features,ray_directions) 
        # front_sampled_features = front_sampled_features.view(N, M, -1)
        # N, M, C = back_sampled_features.shape
        back_sampled_features = back_sampled_features.view(N*M, C) # [4*9216*48, 16]
        # back_sampled_features = self.attention(back_sampled_features, ray_directions)
        # back_sampled_features = back_sampled_features.view(N, M, -1)
        # cross attention：direction
        x = self.attention(front_sampled_features, back_sampled_features, ray_directions)
        # x = front_sampled_features + back_sampled_features
        
        # x = torch.cat([front_sampled_features,back_sampled_features],-1)
        force_sigmoid = force_sigmoid or self.force_sigmoid

        # N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x) #  N*M, 33
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


# # for debug
# if __name__ == "__main__":
#     triplane_width=16
#     decoder = MultiViewOSGDecoder(
#             triplane_width,
#             {
#                 'decoder_lr_mul': 1.0,
#                 'decoder_output_dim': 32,
#             },
#     )

#     B = 4
#     n_ray = 9216
#     N= B*n_ray
#     n_planes = 3
#     M = 48 
#     C = 16
#     front_sampled_features = torch.zeros([N, n_planes, M, C])
#     back_sampled_features = torch.zeros([N, n_planes, M, C])
#     ray_directions = torch.zeros([N, M, 3]).view(N*M, -1)
#     dict = decoder(front_sampled_features,back_sampled_features, ray_directions)
#     rgb = dict['rgb']
#     sigma = dict['sigma']
#     print("rgb: ", rgb.shape) # [4*9216,48,32]
#     print("sigma: ", sigma.shape) # [4*9216,48,1]