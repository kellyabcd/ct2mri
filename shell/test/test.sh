#!/bin/bash
# 进入项目根目录（main.py所在目录）
cd /home/user/Brownian-Bridge-Diffusion-Model/CT2MRI-main

# 定义日期和前缀变量
date="251107"
prefix="ct2mr_aligned"

config_name="BBDM_base_pure.yaml"
HW="160"
plane="axial"
ddim_eta=0.0

# gpu_ids="1"
gpu_ids="1,2"
export CUDA_VISIBLE_DEVICES=""  # 禁止程序访问任何GPU



# 生成实验名称
exp_name="${date}_${HW}_BBDM_${plane}_DDIM_${prefix}"

# 测试相关参数（修正模型路径为实际项目路径）
test_epoch="29"
# resume_model="/home/user/Brownian-Bridge-Diffusion-Model/CT2MRI-main/results/ct2mr_$HW/$exp_name/checkpoint/latest_model_$test_epoch.pth"
# resume_optim="/home/user/Brownian-Bridge-Diffusion-Model/CT2MRI-main/results/ct2mr_$HW/$exp_name/checkpoint/latest_optim_sche_$test_epoch.pth"
resume_model="/home/user/Brownian-Bridge-Diffusion-Model/CT2MRI-main/results/ct2mr_160/251107_160_BBDM_axial_DDIM_ct2mr_aligned/checkpoint/latest_model_29.pth"
resume_optim="/home/user/Brownian-Bridge-Diffusion-Model/CT2MRI-main/results/ct2mr_160/251107_160_BBDM_axial_DDIM_ct2mr_aligned/checkpoint/latest_optim_sche_29.pth"

sample_step=50
inference_type="normal"
ISTA_step_size=0.5
num_ISTA_step=1

# 运行main.py（使用当前目录的main.py，因已cd到项目根目录）
python main.py \
    --exp_name $exp_name \
    --config /home/user/Brownian-Bridge-Diffusion-Model/CT2MRI-main/configs/$config_name \
    --sample_to_eval \
    --gpu_ids $gpu_ids \
    --resume_model $resume_model \
    --resume_optim $resume_optim \
    --HW $HW \
    --plane $plane \
    --ddim_eta $ddim_eta \
    --sample_step $sample_step \
    --inference_type $inference_type \
    --ISTA_step_size $ISTA_step_size \
    --num_ISTA_step $num_ISTA_step \
    --dataset_path "/home/user/Brownian-Bridge-Diffusion-Model/Syna2023"