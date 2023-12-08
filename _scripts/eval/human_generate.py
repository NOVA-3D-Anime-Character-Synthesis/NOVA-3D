



from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
from _util.threedee_v0 import * ; import _util.threedee_v0 as u3d
from _util.video_v1 import * ; import _util.video_v1 as uvid
# import _train.eg3dc.util.eg3dc_v0 as ueg3d
import _train.eg3dc.util.eg3dc_v1 as ueg3d
import _train.eg3dc.datasets.inference as eginf
import _util.serving_v1 as userving
from _util import sketchers_v2 as usketch
from _util import eg3d_metrics3d as egm
import os


os.environ['CUDA_VISIBLE_DEVICES']='1'
device = torch.device('cuda')

ap = uutil.argparse.ArgumentParser()
ap.add_argument('--generator_module')
ap.add_argument('--dataset')
ap.add_argument('--inferquery')
# ap.add_argument('--batch_size', type=int, default=1)
args = ap.parse_args()
generator_module = args.generator_module
inferquery = args.inferquery # 'human_ecrutileE_eclustrousC_n120-00000-000040'
dataset = args.dataset
assert dataset=="human" or dataset=="ecrutileE" or dataset=="multi"
edn = f'./temp/eval/{inferquery}'

# load dataset
from _databacks import lustrous_renders_v1 as dklustr
dk = dklustr.DatabackendMinna()
if dataset == "human":
    front_bns = [
        f'human_rutileE/ortho/{bn[-1]}/{bn}/front'
        for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_test.csv')
        # for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_test_partial.csv')
    ]
    back_bns = [
        f'human_rutileE/ortho/{bn[-1]}/{bn}/back'
        for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_test.csv')
        # for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_test_partial.csv')
    ]
else:
    front_bns = [
    f'rutileE/ortho/{bn[-1]}/{bn}/front'
    for bn in uutil.read_bns('./_data/lustrous/subsets/rutileEB_test.csv')
    ]
    back_bns = [
        f'rutileE/ortho/{bn[-1]}/{bn}/back'
        for bn in uutil.read_bns('./_data/lustrous/subsets/rutileEB_test.csv')
    ]
# load reconstruction module
ckpt = ueg3d.load_eg3dc_model(inferquery, generator_module, force_sigmoid=True)
G = ckpt.G.eval().to(device)
inference_opts = {
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

# load reconstruction module (resnet extractor)
from _train.danbooru_tagger.helpers.katepca import ResnetFeatureExtractorPCA
resnet = ResnetFeatureExtractorPCA(
    './_data/lustrous/preprocessed/minna_resnet_feats_ortho/pca.pkl', 512,
).eval().to(device)
# eval over samples
size = 512
bw = G.rendering_kwargs['box_warp']
rk = G.rendering_kwargs
r0,r1 = rk['ray_start'], rk['ray_end']
if dataset == 'human':
    distances = 3.5
else:
    distances = 1.0
seed = 0
# i = 1

for front_bn,back_bns in tqdm(zip(front_bns, back_bns)):
    # preprocess
    front_x = dk[front_bn]
    back_x = dk[back_bns]
    front_x.image = front_x.image.resize(size)
    back_x.image = back_x.image.resize(size)
    with torch.no_grad():
        front_resnet_feature = resnet(front_x.image)
        back_resnet_feature = resnet(back_x.image)

    # get geometry (marching cubes)
    with torch.no_grad():
        xin = {
            'cond': {
                'image_ortho_front': front_x.image.bg('w').convert('RGB').t()[None].to(device),
                'image_ortho_back': back_x.image.bg('w').convert('RGB').t()[None].to(device),
                'resnet_chonk': front_resnet_feature[None,0],
                'resnet_chonk_front':front_resnet_feature[None,0],
                'resnet_chonk_back':back_resnet_feature[None,0],
            },
            'distances':distances * torch.ones(1).to(device),
            'seeds': [seed,],
            **inference_opts,
        }
        vol = egm.get_eg3d_volume(G, xin)
        mc = egm.marching_cubes(
            vol['densities'].cpu().numpy()[0,0],
            vol['rgbs'].cpu().numpy()[0,:3],
            G.rendering_kwargs['box_warp'],
            level=0.5,
        )
    fn_march = f'{edn}/{front_bn.replace("ortho","marching_cubes")}.pkl'
    uutil.pdump(mc, mkfile(fn_march))

    # get images (various views)
    for cm,cam_view,elev,azim,fov in [
        ('camO', 'front', 0, 0, -1),
        ('camO', 'left', 0, 90, -1),
        ('camO', 'right', 0, -90, -1),
        ('camO', 'back', 0, 180, -1),
        *[
            ('camP', f'{v:04d}', *dklustr.cam60[v].to(device), 30)
            for v in dklustr.camsubs['spin12']
        ],
        *[
            ('camVideo', f'{v:04d}', *dklustr.cam360[v].to(device), 30)
            for v in dklustr.camsubs['spin360']
        ],
    ]:
        with torch.no_grad():
            xin = {
                'elevations': elev *torch.ones(1).to(device),
                'azimuths': azim *torch.ones(1).to(device),
                'fovs': fov *torch.ones(1).to(device),
                'distances': distances * torch.ones(1).to(device),
                'cond': {
                    'image_ortho_front': front_x.image.bg('w').convert('RGB').t()[None].to(device),
                    'image_ortho_back': back_x.image.bg('w').convert('RGB').t()[None].to(device),
                    'resnet_chonk': front_resnet_feature[None,0],
                    'resnet_chonk_front':front_resnet_feature[None,0],
                    'resnet_chonk_back':back_resnet_feature[None,0],
                },
                'seeds': [seed,],
                **inference_opts,
            }
            out = G.f(xin, return_more=True)

        if cm=='camO':
            fn_pred_rgb = f'{edn}/{front_bn.replace("ortho","ortho")}.png'
            fn_pred_xyza = f'{edn}/{front_bn.replace("ortho","ortho_xyza")}.png'
        elif cm=='camP':
            fn_pred_rgb = f'{edn}/{front_bn.replace("ortho","rgb60")}.png'
            fn_pred_xyza = f'{edn}/{front_bn.replace("ortho","xyza60")}.png'
        elif cm=='camVideo':
            fn_pred_rgb = f'{edn}/{front_bn.replace("ortho","rgb360")}.png'
            fn_pred_xyza = f'{edn}/{front_bn.replace("ortho","xyza360")}.png'
        else:
            assert 0
        fn_pred_rgb = fn_pred_rgb.replace('/front', f'/{cam_view}')
        fn_pred_xyza = fn_pred_xyza.replace('/front', f'/{cam_view}')

        xyza = torch.cat([
            (out['image_xyz']+bw/2)/bw,
            out['image_weights'],
        ], dim=1)
        I(out['image']).save(mkfile(fn_pred_rgb))
        I(xyza).save(mkfile(fn_pred_xyza))
        
        del xin, out, xyza
        # break
    # if i==64:
    #     break
    # i+=1

