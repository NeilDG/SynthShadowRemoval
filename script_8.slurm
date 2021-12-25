#!/bin/bash
#SBATCH -J D_8
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=6
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_8.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=8

echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python iid_main.py \
--iteration=$ITERATION \
--server_config=1 \
--num_workers=6 \
--batch_size=512 \
--num_blocks=6 --net_config=1 --use_mask=0 \
--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --bce_weight=0.0 --use_bce=1 --min_epochs=200 \
--version_name="rgb2albedo_v6.00" --map_choice="albedo" \
--img_to_load=-1 --load_previous=0

srun python iid_main.py \
--iteration=$ITERATION \
--server_config=1 \
--num_workers=6 \
--batch_size=512 \
--num_blocks=6 --net_config=1 --use_mask=0 \
--l1_weight=10.0 --lpip_weight=1.0 --ssim_weight=0.0 --bce_weight=0.0 --use_bce=1 --min_epochs=200 \
--version_name="rgb2shading_v6.00" --map_choice="shading" \
--img_to_load=-1 --load_previous=0

conda deactivate