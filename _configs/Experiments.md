


panic3d-anime-reconstruction

## download

![](./supplementary/schematic.png)

Download the `panic_data_models_merged.zip` from the project's [drive folder](https://drive.google.com/drive/folders/1Zpt9x_OlGALi-o-TdvBPzUPcvTc7zpuV?usp=share_link), and merge it with this repo's file structure.

There are several repos related to this work, roughly laid out according to the above schematic; please follow instructions in each to download

* A) [panic3d-anime-reconstruction](https://github.com/ShuhongChen/panic3d-anime-reconstruction) (this repo): reconstruction models
* B) [vtubers-dataset](https://github.com/ShuhongChen/vtubers-dataset): download 2D data
* C) [vroid-dataset](https://github.com/ShuhongChen/vroid-dataset): download 3D data
* D) [animerecon-benchmark](https://github.com/ShuhongChen/animerecon-benchmark): download 2D-3D paired evaluation dataset
* C+D) [vroid_renderer](https://github.com/ShuhongChen/vroid_renderer): convert and render 3D models

All the repos add to `./_data/lustrous` and share its file structure; we recommend making a folder on a large drive, then symlinking it as `./_data/lustrous` in each repo


## setup

Make a copy of `./_env/machine_config.bashrc.template` to `./_env/machine_config.bashrc`, and set `$PROJECT_DN` to the absolute path of this repository folder.  The other variables are optional.

This project requires docker with a GPU.  Run these lines from the project directory to pull the image and enter a container; note these are bash scripts inside the `./make` folder, not `make` commands.  Alternatively, you can build the docker image yourself.

    make/docker_pull
    make/shell_docker
    # OR
    make/docker_build
    make/shell_docker


## evaluation

Run this line to reproduce the best-result metrics from our paper.  There might be minor hardware variations and randomness in rendering; below are results from two different machines.

    python3 -m _scripts.eval.generate && python3 -m _scripts.eval.measure

    #      RTX 3080 Ti             GTX 1080 Ti
    #  subset metric  value    subset metric  value 
    # ======================  ======================
    #  front  clip   94.667    front  clip   94.659 
    #  front  lpips  19.373    front  lpips  19.367 
    #  front  psnr   16.914    front  psnr   16.910 
    #  back   clip   85.046    back   clip   85.117 
    #  back   lpips  30.017    back   lpips  30.012 
    #  back   psnr   15.508    back   psnr   15.509 
    #  360    clip   84.606    360    clip   84.629 
    #  360    lpips  25.252    360    lpips  25.250 
    #  360    psnr   15.977    360    psnr   15.976 
    #  geom   cd      1.329    geom   cd      1.328 
    #  geom   f1@5   37.725    geom   f1@5   38.051 
    #  geom   f1@10  65.498    geom   f1@10  65.813 


## training

The training script is at `./_train/eg3dc/runs/ecrutileE_eclustrousC_n120/ecrutileE_eclustrousC_n120.sh`, which contains the command and hyperparameters used for our best model

 119053

## gameday training
panic-3d train 
1. generate vroid rendering image feature:
export PROJECT_DN=$(pwd)
export MACHINE_NAME=gpuA100
python -m _scripts.train.data_clean --dataset rutileEB --phase all
python -m _scripts.train.generate_feature
python -m _scripts.train.generate_ortho_mask
nohup python -m _scripts.train.generate_rgb_mask &
nohup sh ./_train/eg3dc/runs/ecrutileE_eclustrousC_n120/multi_view_panic3d_train.sh &
[1] 6411

nohup sh ./_train/eg3dc/runs/multi_panic3d/multi_view_panic3d_mask_encoder_train.sh >> mv_panic_mask_encoderPat.out &
nohup sh ./_train/eg3dc/runs/multi_panic3d/multi_view_panic3d_mask_encoder_attention_train.sh >> mv_panic_mask_encoderPat_attention.out &

swin_transformer encoder:
python -m _train.swin_transformer.test


## nova-3d training
0. environment setup
mkdir /usr/local/dros/conda/envs/env_panohead
cd /HOME/HOME/common_env/
tar -zxvf panohead.tar.gz -C /usr/local/dros/conda/envs/env_panohead 
conda activate env_panohead
pip install timm
mkdir -p /root/.cache/torch/hub/checkpoints/
cp ./_data/alexnet-owt-7be5be79.pth /root/.cache/torch/hub/checkpoints/
cp ./_data/resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/
cp ./_data/deeplabv3_resnet101_coco-586e9e4e.pth /root/.cache/torch/hub/checkpoints/
mkdir -p ~/.keras/models/
cp ./_data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/
mkdir -p ~/.cache/clip/
cp ./_data/ViT-B-32.pt ~/.cache/clip/
conda activate env_panohead
export PROJECT_DN=$(pwd)
export MACHINE_NAME=gpuA100

1. nova-3d train前预处理：generate human-vroid rendering image feature:
python -m _scripts.train.data_clean --dataset human_rutileEB --phase all
all 11236 - 1013 = 10223
train 8952 - 815 = 8137l
val 1171 - 101 = 1070
test 1113 - 97 = 1016
nohup python -m _scripts.train.generate_human_feature >> generate_human_fature.out & 
python -m _scripts.train.generate_human_ortho_mask
nohup python -m _scripts.train.generate_human_rgb_mask >> generate_human_rgb_mask.out &

 2. train 
# baseline
## panic3d baseline 已完成fid48
    2023.11.14
    nohup sh ./_train/eg3dc/runs/human_multi_panic3d/human_ecrutileE_eclustrousC_n120_4gpu.sh >> human_panic3d_baseline.out &

## mv-panic3d 在跑
    2023.11.15
    nohup sh ./_train/eg3dc/runs/human_multi_panic3d/human_mv_panic3d.sh >> human_mv_panic3d_7319519094.out &
    nohup sh ./_train/eg3dc/runs/human_multi_panic3d/human_mv_panic3d_4gpu.sh >> human_mv_panic3d_4A100_2.out &
# 主实验
## mv-panic3d+mask+pat+attention 4gpu在跑
    2023.11.16
    nohup sh ./_train/eg3dc/runs/human_multi_panic3d/human_multi_view_panic3d_mask_encoder_attention_train.sh >> human_mv_panic3d_mask_pat_attention_735709060.out &
    nohup sh _train/eg3dc/runs/human_multi_panic3d/human_multi_view_panic3d_mask_encoder_attention_train_4gpu.sh  >> human_mv_panic3d_mask_pat_attention_4gpu2.out &
    nohup sh _train/eg3dc/runs/human_multi_panic3d/human_multi_view_panic3d_mask_encoder_attention_big_train_4gpu.sh  >> human_mv_panic3d_mask_pat_attention_big_4gpu_3.out &
    2024.1.23
    nohup sh ./_configs/nova_human/human_nova_1gpu.sh >> human_nova_1gpu.out &
    nohup sh ./_configs/nova_human/human_nova.sh >> human_nova.out &
# 消融实验
## mv-panic3d+mask+pat+Noattention 4gpu在跑
    2023.11.16
    nohup sh ./_train/eg3dc/runs/human_multi_panic3d/human_multi_view_panic3d_mask_encoder_Noattention_train_4gpu.sh >> human_mv_panic3d_mask_pat_cat_4A100.out &
## mv-panic3d+mask+NoPat+attention 4gpu高速缓存在跑
    nohup sh ./_train/eg3dc/runs/human_multi_panic3d/human_mv_panic3d_mask_Noencoder_attention_train_4gpu.sh >> human_mv_panic3d_mask_Nopat_atten_4A100_3.out &
## mv-panic3d+Nomask+Pat+attention 4gpu完成fid133
    nohup sh ./_train/eg3dc/runs/human_multi_panic3d/human_mv_panic3d_Nomask_encoder_attention_train_4gpu.sh >> human_mv_panic3d_Nomask_pat_atten_4A100_4.out &
# NOVA B
    nohup sh ./_train/eg3dc/runs/humanB_multi_panic3d/humanB_train_4gpu.sh >> humanB_NOVA_4gpu_3.out &
# NOVA-deepspeed
    ./_train/eg3dc/runs/human_multi_panic3d/human_nova_big_deepspeed.sh
3. test
python -m _scripts.eval.human_generate --generator_module training.triplane.TriPlaneGenerator_v1 --inferquery human human_ecrutileE_eclustrousC_n120-00000-000040 

python -m _scripts.eval.human_generate --generator_module training.triplane.TriPlaneGenerator_v1 --dataset human --inferquery human_multi_panic3d-00056-000040
# best, mvpanic3d_mask_encoder_attention
nohup python -m _scripts.eval.human_generate --generator_module training.triplane.TriPlaneGenerator_v1 --dataset human --inferquery human_multi_panic3d-00057-000880 &
fid66   nohup python -m _scripts.eval.human_generate --generator_module training.triplane.TriPlaneGenerator_v1 --dataset human --inferquery human_mv_panic3d_mask_encoder_attention-00001-000080 >> generate_best.out &
fid60   nohup python -m _scripts.eval.human_generate --generator_module training.triplane.TriPlaneGenerator_v1 --dataset human --inferquery human_mv_panic3d_mask_encoder_attention-00003-000040 >> best.out &


# baseline
nohup python -m _scripts.eval.human_generate --generator_module training.triplane.TriPlaneGenerator --dataset human --inferquery human_ecrutileE_eclustrousC_n120-00002-000240 >> generate_baseline.out &

# mv-panic3d
nohup python -m _scripts.eval.human_generate --generator_module training.triplane.TriPlaneGenerator --dataset human --inferquery human_mv_ecrutileE_eclustrousC_n120-00000-000480 &
nohup python -m _scripts.eval.human_generate --generator_module training.triplane.TriPlaneGenerator --dataset human --inferquery human_mv_ecrutileE_eclustrousC_n120-00003-000320 &

# 消融Noencoder
export CUDA_VISIBLE_DEVICES=1
nohup python -m _scripts.eval.human_generate --generator_module training.triplane.TriPlaneGenerator_v1 --dataset human --inferquery human_mv_panic3d_mask_Noencoder_attention-00001-000080 >> generate_ablation_Noencoder.out &
# 消融Noattention
nohup python -m _scripts.eval.human_generate --generator_module training.triplane.TriPlaneGenerator_v0 --dataset human --inferquery human_mv_panic3d_mask_encoder_concat-00001-000120 >> generate_ablation_Noattention.out &

human_measure:
python -m _scripts.eval.human_measure --inferquery human_multi_panic3d-00057-000880
python -m _scripts.eval.human_measure --inferquery human_ecrutileE_eclustrousC_n120-00002-000240
nohup python -m _scripts.eval.human_measure --inferquery human_mv_ecrutileE_eclustrousC_n120-00000-000480 >> measure_mv_panic3d.out & 
nohup python -m _scripts.eval.human_measure --inferquery human_mv_panic3d_mask_encoder_attention-00003-000040 >> measure_best.out &

nohup python -m _scripts.eval.human_measure --inferquery human_mv_panic3d_mask_Noencoder_attention-00001-000080 >> measure_ablation_noEncoder.out &

nohup python -m _scripts.eval.human_measure --inferquery human_mv_panic3d_mask_encoder_concat-00001-000120  >> measure_ablation_noAttention.out &

```
    
    opts.data_class = f'_train.eg3dc.datasets.{opts.name.split("_")[0]}.DatasetWrapper'
```

[Title](_train/eg3dc/runs/ecrutileE_eclustrousC_n120/00001/network-snapshot-001360.pkl)

## gameday test
python -m _scripts.eval.mv_generate
python -m _scripts.eval.front_generate
### NOVA Head
nohup python -m _scripts.eval.human_generate --inferquery multi_panic3d-00000-002040 --dataset multi --generator_module training.triplane.TriPlaneGenerator_v1 >> generate_nova_head.out &
nohup python -m _scripts.eval.human_measure --inferquery multi_panic3d-00000-002040  >> measure_nova_head.out &

## data migration
zip -r ./panic3d-anime-reconstruction_total.zip ./ &
nohup unzip -d ./multi_view_panic3d/ panic3d_data_total_v2.zip &
[1] 27209
2024.1.25
mkdir -p ./_data/lustrous/renders/
nohup cp -r ./_data/lustrous/renders/human_rutileE/ /root/workspace/_data/lustrous/renders/ > gpu2_1.out &
cp -r ./_data/lustrous/subsets/ /root/workspace/_data/lustrous/ 
## 挑图
首页图：


展示图
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/0/13958053289047560
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/0/275620893320860410
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/0/304713568393608940
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/0/508510521668523170
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/0/670119082639015540
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/0/976329964746524550
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/0/983740415920655850

数据集特点图：
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/0/13958053289047560
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/9/25702208362782739
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/9/42541341466447439
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/9/179010904680559549
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/9/382189100386668529
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_data/lustrous/renders/human_rutileE/ortho/9/632609457686638819

测试对比图：human_multi_panic3d-00057-000880 VS human_ecrutileE_eclustrousC_n120-00001-000280
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/temp/eval/human_multi_panic3d-00057-000880/human_rutileE/ortho/9/16629098228437899/front.png
/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/temp/eval/human_ecrutileE_eclustrousC_n120-00001-000280/human_rutileE/ortho/9/16629098228437899/front.png

artifacts对比图：
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/temp/eval/human_ecrutileE_eclustrousC_n120-00001-000280/human_rutileE/ortho/9/856945445812537029

鉴别器作用对比图：
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/temp/eval/human_mv_ecrutileE_eclustrousC_n120-00000-000480/human_rutileE/ortho/9/106617834206412049

best的背部与baseline的背部
/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/temp/eval/human_mv_ecrutileE_eclustrousC_n120-00000-000480/human_rutileE/ortho/9/145169368815004339



