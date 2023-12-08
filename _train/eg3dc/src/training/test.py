### module test
import sys
sys.path.append("/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_train/eg3dc/src/")
from torch_utils.ops import upfirdn2d
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

backbone = "training.generator.Generator_v0"
backbone = dnnlib.util.construct_class_by_name(
                class_name=backbone,
                z_dim=512, c_dim=25, w_dim=512, img_resolution=256,
                img_channels=16*3,            
                cond_mode=None,
                multi_view_cond_mode=None, 
                mapping_kwargs=None,  
                **kwargs                                               
            )
# import torch
# import torch.nn as nn

# class MyModel(nn.Module):
#     def __init__(self,resample_filter=[1,3,3,1]):
#         super(MyModel, self).__init__()

#         # 定义一个持久性缓冲区
#         self.register_buffer('my_buffer', torch.randn(3, 3))
#         self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
# model = MyModel()

# # 打印模型
# print(model)

# # 持久性缓冲区在模型状态字典中
# print("Model State Dict:")
# print(model.state_dict())



### dataset test
# data_class = "_train.eg3dc.datasets.multi.DatasetWrapper"
# data_subset = "rutileEB"
# def init_dataset_kwargs(data_class, data_subset):
#     try:
#         # dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
#         dataset_kwargs = dnnlib.EasyDict(
#             class_name=data_class,
#             subset=data_subset,
#         )
#         dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
#         dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
#         dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
#         dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
#         return dataset_kwargs, dataset_obj.name
#     except IOError as err:
#         raise click.ClickException(f'--data: {err}')
    
# training_set_kwargs, dataset_name = init_dataset_kwargs(data_class=data_class, data_subset=data_subset)
# print(training_set_kwargs)
# training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
# training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
# training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))