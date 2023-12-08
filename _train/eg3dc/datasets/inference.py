
from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
from _databacks import lustrous_renders_v1 as dklustr

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        args = args or Dict()
        self.seed = 0
        self.dk = dklustr.DatabackendMinna()
        self.front_bns = [
            f'human_rutileE/ortho/{bn[-1]}/{bn}/front'
            for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_test.csv')
        ]
        self.back_bns = [
            f'human_rutileE/ortho/{bn[-1]}/{bn}/back'
            for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_test.csv')
        ]
        self.inference_opts = {
            'triplane_crop': 0.1,
            'cull_clouds': 0.5,
            # 'binarize_clouds': 0.4,
            'paste_params': {
                'mode': 'default',
                'thresh_weight': 0.95,
                'thresh_edges': 0.02,
                'thresh_occ': 0.05, 'offset_occ': 0.01,
                'thresh_dxyz': 0.000005,
            },
        }
        from _train.danbooru_tagger.helpers.katepca import ResnetFeatureExtractorPCA
        self.resnet = ResnetFeatureExtractorPCA(
            './_data/lustrous/preprocessed/minna_resnet_feats_ortho/pca.pkl', 512,
        ).eval().to(device)

    def __getitem__(self, idx):
        front_x = dk[self.front_bns[idx]]
        back_x = dk[self.back_bns[idx]]
        with torch.no_grad():
            front_resnet_feature = self.resnet(front_x.image)
            back_resnet_feature = self.resnet(back_x.image)
        data_item = {}
        data_item['geometry'] = {
            'cond': {
                'image_ortho_front': front_x.image.bg('w').convert('RGB').t()[None].to(device),
                'image_ortho_back': back_x.image.bg('w').convert('RGB').t()[None].to(device),
                'resnet_chonk': front_resnet_feature[None,0],
                'resnet_chonk_front':front_resnet_feature[None,0],
                'resnet_chonk_back':back_resnet_feature[None,0],
            },
            'seeds': [self.seed,],
            **self.inference_opts,
        }
        data_item['image'] = {}
        
        return data_item
    
    def __len__(self):
        return len(self.front_bns)