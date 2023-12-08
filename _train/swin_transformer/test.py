
from _train.swin_transformer.helpers.swin_encoder import GOAEncoder
import torch

B = 4
C = 3
H = 256
W = 256
x = torch.randn((B,C,H,W))

config_path = "/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_train/swin_transformer/configs/swinv2.yaml"
print(x.shape)

encoder  = GOAEncoder(config_path)
ws,ws_base = encoder(x)
print(ws.shape)