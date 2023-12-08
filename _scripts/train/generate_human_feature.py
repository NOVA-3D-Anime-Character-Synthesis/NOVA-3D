import pickle
from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
# from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
# from _util.threedee_v0 import * ; import _util.threedee_v0 as u3d
# from _util.video_v1 import * ; import _util.video_v1 as uvid

# import _train.eg3dc.util.eg3dc_v0 as ueg3d
# import _util.serving_v1 as userving
# from _util import sketchers_v2 as usketch
# from _util import eg3d_metrics3d as egm

device = torch.device('cuda')

# inferquery = 'ecrutileE_eclustrousC_n120-00000-000200'


# panic3d train debug, generate resnet_chonk
# load dataset
from _databacks import lustrous_renders_v1 as dklustr
dk = dklustr.DatabackendMinna()
front_bns = [
    f'human_rutileE/ortho/{bn[-1]}/{bn}/front'
    for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_all.csv')
]
back_bns = [
    f'human_rutileE/ortho/{bn[-1]}/{bn}/back'
    for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_all.csv')
]
front_eds = [
    f'./_data/lustrous/renders/human_rutileE/ortho_katepca_chonk/{ed[-1]}/{ed}/front.pkl'
    for ed in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_all.csv')
]
back_eds = [
    f'./_data/lustrous/renders/human_rutileE/ortho_katepca_chonk/{ed[-1]}/{ed}/back.pkl'
    for ed in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_all.csv')
]
# print(bns[0])
# aligndata = pload('./_data/lustrous/renders/daredemoE/fandom_align_alignment.pkl')


# load reconstruction module (resnet extractor)
from _train.danbooru_tagger.helpers.katepca import ResnetFeatureExtractorPCA
resnet = ResnetFeatureExtractorPCA(
    './_data/lustrous/preprocessed/minna_resnet_feats_ortho/pca.pkl', 512,
).eval().to(device)

# load reconstruction module
# ckpt = ueg3d.load_eg3dc_model(inferquery, force_sigmoid=True)
# G = ckpt.G.eval().to(device)
# inference_opts = {
#     'triplane_crop': 0.1,
#     'cull_clouds': 0.5,
#     # 'binarize_clouds': 0.4,
#     'paste_params': {
#         'mode': 'default',
#         'thresh_weight': 0.95,
#         'thresh_edges': 0.02,
#         'thresh_occ': 0.05, 'offset_occ': 0.01,
#         'thresh_dxyz': 0.000005,
#     },
# }


# eval over samples
# bw = G.rendering_kwargs['box_warp']
# rk = G.rendering_kwargs
# r0,r1 = rk['ray_start'], rk['ray_end']
seed = 0
for bn,ed in tqdm(zip(front_bns,front_eds)):
    # preprocess
    x = dk[bn]
    with torch.no_grad():
        # print(x.image.shape)
        x['resnet_features'] = resnet(x.image).cpu()
    # get geometry (marching cubes)
    with torch.no_grad():
        directory = os.path.dirname(ed)
        os.makedirs(directory, exist_ok=True)
        with open(ed,'wb') as file:
            pickle.dump(x['resnet_features'][None,0], file)
    # break
for bn,ed in tqdm(zip(back_bns,back_eds)):
    # preprocess
    x = dk[bn]
    with torch.no_grad():
        # print(x.image.shape)
        x['resnet_features'] = resnet(x.image).cpu()
    # get geometry (marching cubes)
    with torch.no_grad():
        directory = os.path.dirname(ed)
        os.makedirs(directory, exist_ok=True)
        with open(ed,'wb') as file:
            pickle.dump(x['resnet_features'][None,0], file)
#     # break