



from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
from _util.threedee_v0 import * ; import _util.threedee_v0 as u3d
from _util.video_v1 import * ; import _util.video_v1 as uvid

import _train.eg3dc.util.eg3dc_v0 as ueg3d
import _util.serving_v1 as userving
from _util import sketchers_v2 as usketch
from _util import eg3d_metrics3d as egm

device = torch.device('cuda')


# ap = uutil.argparse.ArgumentParser()
# ap.add_argument('name')
# args = ap.parse_args()
# inferquery = args.name
inferquery = 'BACK_ecrutileE_eclustrousC_n120-00000-000200'
edn = f'./temp/eval/{inferquery}'

# load dataset
from _databacks import lustrous_renders_v1 as dklustr
dk = dklustr.DatabackendMinna()
front_bns = [
    f'rutileE/ortho/{bn[-1]}/{bn}/front'
    for bn in uutil.read_bns('./_data/lustrous/subsets/rutileEB_test.csv')
]
back_bns = [
    f'rutileE/ortho/{bn[-1]}/{bn}/back'
    for bn in uutil.read_bns('./_data/lustrous/subsets/rutileEB_test.csv')
]

# load reconstruction module
ckpt = ueg3d.load_eg3dc_model(inferquery, force_sigmoid=True)
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
bw = G.rendering_kwargs['box_warp']
rk = G.rendering_kwargs
r0,r1 = rk['ray_start'], rk['ray_end']
seed = 0
for back_bn in tqdm(back_bns):
    # preprocess
    back_x = dk[back_bn]
    with torch.no_grad():
        back_x['resnet_features'] = resnet(back_x.image)

    # get geometry (marching cubes)
    with torch.no_grad():
        xin = {
            'cond': {
                'image_ortho_front': back_x.image.bg('w').convert('RGB').t()[None].to(device),
                'resnet_chonk': back_x['resnet_features'][None,0],
            },
            'seeds': [seed,],
            **inference_opts,
        }
        vol = egm.get_eg3d_back_volume(G, xin)
        mc = egm.marching_cubes(
            vol['densities'].cpu().numpy()[0,0],
            vol['rgbs'].cpu().numpy()[0,:3],
            G.rendering_kwargs['box_warp'],
            level=0.5,
        )
    fn_march = f'{edn}/{back_bn.replace("ortho","marching_cubes")}.pkl'
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
    ]:
        with torch.no_grad():
            xin = {
                'elevations': elev *torch.ones(1).to(device),
                'azimuths': azim *torch.ones(1).to(device),
                'fovs': fov *torch.ones(1).to(device),
                'cond': {
                    'image_ortho_front': back_x.image.bg('w').convert('RGB').t()[None].to(device),
                    'resnet_chonk': back_x['resnet_features'][None,0],
                },
                'seeds': [seed,],
                **inference_opts,
            }
            out = G.f(xin, return_more=True)

        if cm=='camO':
            fn_pred_rgb = f'{edn}/{back_bn.replace("ortho","ortho")}.png'
            fn_pred_xyza = f'{edn}/{back_bn.replace("ortho","ortho_xyza")}.png'
        elif cm=='camP':
            fn_pred_rgb = f'{edn}/{back_bn.replace("ortho","rgb60")}.png'
            fn_pred_xyza = f'{edn}/{back_bn.replace("ortho","xyza60")}.png'
        else:
            assert 0
        fn_pred_rgb = fn_pred_rgb.replace('/back', f'/{cam_view}')
        fn_pred_xyza = fn_pred_xyza.replace('/back', f'/{cam_view}')

        xyza = torch.cat([
            (out['image_xyz']+bw/2)/bw,
            out['image_weights'],
        ], dim=1)
        I(out['image']).save(mkfile(fn_pred_rgb))
        I(xyza).save(mkfile(fn_pred_xyza))
        
        del xin, out, xyza
        # break

