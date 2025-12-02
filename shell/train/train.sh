#!/bin/bash
cd /home/user/Brownian-Bridge-Diffusion-Model/CT2MRI-main

date="251107"

config_name="BBDM_base_pure.yaml"
HW="160"
plane="axial"
gpu_ids="0"
batch=16
ddim_eta=0.0

prefix="ct2mr_aligned"

exp_name="${date}_${HW}_BBDM_${plane}_DDIM_${prefix}"

mkdir -p /home/user/Brownian-Bridge-Diffusion-Model/CT2MRI-main/results/ct2mr_160/

# resume_model="/home/user/Brownian-Bridge-Diffusion-Model/CT2MRI-main/results/ct2mr_$HW/$exp_name/checkpoint/last_model.pth"
# resume_optim="/home/user/Brownian-Bridge-Diffusion-Model/CT2MRI-main/results/ct2mr_$HW/$exp_name/checkpoint/last_optim_sche.pth"
python -u /home/user/Brownian-Bridge-Diffusion-Model/CT2MRI-main/main.py \
    --train \
    --exp_name $exp_name \
    --config /home/user/Brownian-Bridge-Diffusion-Model/CT2MRI-main/configs/$config_name \
    --HW $HW \
    --plane $plane \
    --batch $batch \
    --ddim_eta $ddim_eta \
    --sample_at_start \
    --save_top \
    --gpu_ids $gpu_ids
#    --dataset_type "ct2mr" \

    # --resume_model $resume_model \
    # --resume_optim $resume_optim \
   

