#!/bin/bash

#SBATCH --job-name=ecrutileE_eclustrousC_n120

# source ./venv/bin/activate
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate eg3d

# source ./_env/project_config.bashrc
# source ./_env/machine_config.bashrc
# source ./_env/make_config.bashrc

# && CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m _train.eg3dc.trainers.train_eclustrousC \
# && CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m _train.eg3dc.trainers.train_eclustrousC \
cd $PROJECT_DN \
&& CUDA_VISIBLE_DEVICES=0,1 python3 -m _train.eg3dc.trainers.train_multi_view_panic3d_mask_encoder \
    --name=human_nova_big \
    --training_loop_version=training_loop_v2 \
    --loss_module=training.loss.StyleGAN2LossMultiMaskOrthoCond \
    --generator_module=training.triplane.TriPlaneGenerator_v1 \
    --generator_backbone=training.generator.Generator_v0 \
    --encoder_module=training.encoder.Encoder_v1 \
    --discriminator_module=training.dual_discriminator.MaskDualDiscriminator \
    --cond_mode=ortho_front.cond_img_norm_4.concatfront.crossavg_4.reschonk_add_512.inj_6b_4 \
    --multi_view_cond_mode=ortho_front.cond_img_norm_4.concatfront.crossavg_4.reschonk_add_512.inj_6b_4,ortho_back.cond_img_norm_4.concatfront.crossavg_4.reschonk_add_512.inj_6b_4 \
    --encoder_config=/HOME/HOME/gameday/gameday-3d-human-reconstruction-multi_view_panic3d-/_train/swin_transformer/configs/swinv2.yaml \
    --data_subset=human_rutileEB \
    \
    --gpus=2 \
    --batch=8 \
    --kimg=25000 \
    --snap=10 \
    \
    --resume_discrim=./_data/eg3d/networks/ffhq512-128.pkl \
    --resume=None \
    \
    --triplane_depth=3 \
    --triplane_width=16 \
    --sr_channels_hidden=16 \
    --backbone_resolution=512 \
    --cbase_g=32768 --cmax_g=512 \
    --cbase_d=32768 --cmax_d=512 \
    \
    --neural_rendering_resolution_initial=176 \
    --neural_rendering_resolution_final=176 \
    --neural_rendering_resolution_fade_kimg=1000 \
    \
    --gamma=5.0 \
    --seg_gamma=1.0 \
    --density_reg=0.5 \
    --dlr=0.0010 \
    --glr=0.0025 \
    --blur_fade_kimg=0 \
    --reg_type=l1 \
    \
    --lambda_gcond_lpips=20.0 \
    --lambda_gcond_l1=4.0 \
    --lambda_gcond_alpha_l2=1.0 \
    --lambda_gcond_depth_l2=1000.0 \
    \
    --lambda_gcond_sides_lpips=20.0 \
    --lambda_gcond_sides_l1=4.0 \
    --lambda_gcond_sides_alpha_l2=1.0 \
    --lambda_gcond_sides_depth_l2=1000.0 \
    \
    --lambda_gcond_back_lpips=20.0 \
    --lambda_gcond_back_l1=4.0 \
    --lambda_gcond_back_alpha_l2=1.0 \
    --lambda_gcond_back_depth_l2=1000.0 \


    # --metrics=fid100
    # --cond_mode=ortho_front.add_shuffle2 \
    # --use_triplane=1 \
    # --batch-gpu=4 \
    # --resume=./_train/runs/ecrutileE_eclustrousC_n120/00002/network-snapshot-000400.pkl \
    # --resume=./_train/runs/kongoC_n14/00000/network-snapshot-002160.pkl \
    # --resume=./networks/ffhq512-128.pkl \
    # --reg_type=l1 \   # // 'l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed', 'total-variation' 
# python3 ~/sleep.py

