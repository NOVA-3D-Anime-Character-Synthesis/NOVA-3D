



from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
from _util.threedee_v0 import * ; import _util.threedee_v0 as u3d

from _databacks import lustrous_gltf_v0_measurable as uvrm

from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as ssim
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device('cuda')

ap = uutil.argparse.ArgumentParser()
ap.add_argument('--inferquery')
ap.add_argument('--dataset')
args = ap.parse_args()
inferquery = args.inferquery
# inferquery = 'human_ecrutileE_eclustrousC_n120-00000-000040'
dataset = args.dataset
assert dataset=="human" or dataset=="ecrutileE" or dataset=="multi"
edn = f'./temp/eval/{inferquery}'


# load dataset
from _databacks import lustrous_renders_v1 as dklustr
dk = dklustr.DatabackendMinna()

bns = [
    f'human_rutileE/ortho/{bn[-1]}/{bn}/front'
    # for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_test_partial.csv')
    for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_test.csv')
]
# back_bns = [
#     f'human_rutileE/ortho/{bn[-1]}/{bn}/back'
#     for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_test.csv')
# ]
aligndata = pload('./_data/lustrous/renders/daredemoE/fandom_align_alignment.pkl')


# load metrics
import clip
clip_model,preprocess = clip.load("ViT-B/32", device=device)
def clipsim(a, b):
    with torch.no_grad():
        a = preprocess(I(a).pil()).unsqueeze(0).to(device)
        b = preprocess(I(b).pil()).unsqueeze(0).to(device)
        a = clip_model.encode_image(a)
        b = clip_model.encode_image(b)
        cs = (a*b).sum() / (a[0].norm() * b[0].norm())
    return cs
lpips = utorch.LPIPSLoss().to(device)
psnr = torchmetrics.PeakSignalNoiseRatio().to(device)

def ssim_metric(a, b):
    a = a.cpu().squeeze(0).permute(1, 2, 0).numpy() # 1,3,512,512
    b = b.cpu().squeeze(0).permute(1, 2, 0).numpy() # 1,3,512,512
    ssim_value = ssim(a, b,win_size=3,data_range=1,multichannel=True)
    return ssim_value

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg')
def calculate_fid(model, a, b):
        # pre-process images
        a = preprocess_input(a)
        b = preprocess_input(b)
        # calculate activations
        act1 = model.predict(a)
        act2 = model.predict(b)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = numpy.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(np.dot(sigma1, sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
                covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

mets = {
    'lpips': lpips,
    'psnr': psnr,
    'clip': clip,
    'ssim': ssim_metric,

}


# 3d helpers
def filter_mesh(v,f, roi, bw, size=512):
    (fcx,fcy),(fsx,fsy) = roi
    fcx,fcy,fsx,fsy = fcx/size,fcy/size,fsx/size,fsy/size
    cx,cy = (
        -bw/2 + (fcy*bw),
        bw/2 - (fcx*bw),
    )
    sx,sy = bw*fsy, bw*fsx
    vmask = (
        (cx < v[:,0]) & (v[:,0] < cx+sx)
        &
        (cy-sy < v[:,1]) & (v[:,1] < cy)
    )
    wv = vmask
    faces = f
    wf = np.isin(faces, np.where(wv)[0]).all(axis=1)
    faces = (np.cumsum(wv)-1)[faces[wf]]
    # return vmask
    # return v[wv], faces
    return {
        'verts': v[wv],
        'faces': faces,
    }
def get_point_mesh_distance(queries, v, f):
    query_mesh = queries
    n = len(queries)

    # find closest point on mesh
    dist2,fcl,vcl = igl.point_mesh_squared_distance(
        query_mesh, v, f,
    )
    dist = np.sqrt(dist2)#[...,None]
    return dist
def point_mesh_f1(p2s, s2p, thresh):
    pre = (p2s<=thresh).mean()
    rec = (s2p<=thresh).mean()
    return Dict({
        'precision': pre,
        'recall': rec,
        'threshold': thresh,
        'f1': (
            2*pre*rec / (pre+rec)
            if not pre==rec==0.0
            else 0.0
        )
    })

# eval over samples
if dataset == 'human':
    bw = 2.1 # 0.7
else:
    bw = 0.7
roi = bw
size = 512
n_sample = 10000
ans2d = defaultdict(lambda: defaultdict(list))
ans3d = defaultdict(list)
i = 1
for bn in tqdm(bns):

    # roi = aligndata[bn]['area_of_interest']
    # roi_horiz = ((roi[0][0], 0), (roi[1][0], 512))
    # roi_back = ((roi[0][0], 512-(roi[0][1]+roi[1][1])), (roi[1][0], roi[1][1]))


    ################ 2d metrics ################

    # front metrics
    gt_rgb = dk[bn].image.resize(size).convert('RGBA').bg('w').convert('RGB').t()[None].to(device)
    pred_rgb = I(f"{edn}/{bn}.png").resize(size).convert('RGBA').bg('w').convert('RGB').t()[None].to(device)
    for k,v in mets.items():
        ans2d['front'][k].append(v(pred_rgb, gt_rgb).item())
        
    # back metrics
    gt_rgb = dk[bn.replace('/front','/back')].image.resize(size).convert('RGBA').bg('w').convert('RGB').t()[None].to(device)
    pred_rgb = I(f"{edn}/{bn.replace('/front','/back')}.png").resize(size).convert('RGBA').bg('w').convert('RGB').t()[None].to(device)
    for k,v in mets.items():
        ans2d['back'][k].append(v(pred_rgb, gt_rgb).item())

    # left metrics
    gt_rgb = dk[bn.replace('/front','/left')].image.resize(size).convert('RGBA').bg('w').convert('RGB').t()[None].to(device)
    pred_rgb = I(f"{edn}/{bn.replace('/front','/left')}.png").resize(size).convert('RGBA').bg('w').convert('RGB').t()[None].to(device)
    for k,v in mets.items():
        ans2d['left'][k].append(v(pred_rgb, gt_rgb).item())

    # right metrics
    gt_rgb = dk[bn.replace('/front','/right')].image.resize(size).convert('RGBA').bg('w').convert('RGB').t()[None].to(device)
    pred_rgb = I(f"{edn}/{bn.replace('/front','/right')}.png").resize(size).convert('RGBA').bg('w').convert('RGB').t()[None].to(device)
    for k,v in mets.items():
        ans2d['right'][k].append(v(pred_rgb, gt_rgb).item())

    # # 360 metrics
    viewavg = defaultdict(list)
    for view in dklustr.camsubs['spin12']:
        view = f'/{view:04d}'
        gt_rgb = dk[bn.replace('/ortho','/rgb60').replace('/front',view)].image.resize(size).convert('RGBA').bg('w').convert('RGB').t()[None].to(device)
        pred_rgb = I(f"{edn}/{bn.replace('/ortho','/rgb60').replace('/front',view)}.png").resize(size).convert('RGBA').bg('w').convert('RGB').t()[None].to(device)
        for k,v in mets.items():
            viewavg[k].append(v(pred_rgb, gt_rgb).item())
    for k in mets:
        ans2d['360'][k].append(np.mean(viewavg[k]))


    ################ 3d metrics ################

    # # load pred mesh
    # fn_march = f'{edn}/{bn.replace("front","marching_cubes")}.pkl'
    # mc = pload(fn_march)
    # mc['verts'] = mc['verts'] * np.asarray([-1,1,1])[None]

    # # get pred mesh
    # mesh_pred = filter_mesh(mc['verts'], mc['faces'], roi, bw)
    # points_pred = uvrm.LustrousGLTFDecapitated.sample_points_near_surface(
    #     Dict({
    #         **mesh_pred,
    #         # 'verts': mc['verts'],
    #         # 'faces': mc['faces'],
    #         # 'boxwarp': bw,
    #     }),
    #     n_sample=n_sample,
    #     sigma=0.0,
    #     seed=bn,
    #     clip=False,
    # )

    # # # get gt mesh
    # _,dt,franch,idx,_ = bn.split('/')
    # gltf = uvrm.LustrousGLTF(f'./_data/lustrous/raw/vroid/{franch}/{idx}.vrm').remove_innards()
    # head = uvrm.LustrousGLTFDecapitated(gltf)
    # cv2our_world = np.asarray([
    #     [-1,0,0,0],
    #     [0,1,0,0],
    #     [0,0,-1,0],
    #     [0,0,0,1],
    # ])
    # cv2our_world_inv = np.linalg.inv(cv2our_world)
    # mesh_gt = filter_mesh(head.verts, head.faces, roi, bw)
    # points_gt = (
    #     cv2our_world_inv[:3,:3] @
    #     uvrm.LustrousGLTFDecapitated.sample_points_near_surface(
    #         Dict({
    #             **mesh_gt,
    #             # 'verts': mc['verts'],
    #             # 'faces': mc['faces'],
    #             # 'boxwarp': bw,
    #         }),
    #         n_sample, sigma=0, seed=bn, clip=False,
    #     ).T
    # ).T

    # # calc metrics
    # p2s = get_point_mesh_distance(
    #     points_pred,
    #     (cv2our_world_inv[:3,:3] @ mesh_gt['verts'].T).T,
    #     mesh_gt['faces'],
    # )
    # s2p = get_point_mesh_distance(
    #     points_gt,
    #     mesh_pred['verts'],
    #     mesh_pred['faces'],
    # )
    # ans3d['p2s'].append(p2s.mean())
    # ans3d['s2p'].append(s2p.mean())
    # ans3d['cd'].append((p2s.mean()+s2p.mean())/2)
    # for th in [0.005, 0.01, 0.05, 0.1, 0.5]:
    #     ans3d[f'f1_{int(th*1000):03d}'] = point_mesh_f1(p2s, s2p, th)['f1']


# display results
t = [
    ['subset', 'metric', 'value'],
    ['=::>'],
]
for s in reversed(sorted(ans2d)):
    t.extend([
        [(s,'l'), ('ssim','l'), (100*np.mean(ans2d[s]['ssim']), 'r:.3f'), ],
        [(s,'l'), ('lpips','l'), (100*np.mean(ans2d[s]['lpips']), 'r:.3f'), ],
        [(s,'l'), ('psnr','l'), (np.mean(ans2d[s]['psnr']), 'r:.3f'), ],
    ])
# t.extend([
#     ['geom::l', 'cd::l', (100*np.mean(ans3d['cd']), 'r:.3f')],
#     ['geom::l', 'f1@5::l', (100*np.mean(ans3d['f1_005']), 'r:.3f')],
#     ['geom::l', 'f1@10::l', (100*np.mean(ans3d['f1_010']), 'r:.3f')],
# ])
print(Table(t))










