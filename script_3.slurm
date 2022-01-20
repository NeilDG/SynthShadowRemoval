#!/bin/bash
#SBATCH -J E_3
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=6
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_3.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=3
L1_WEIGHT=0.0
LPIP_WEIGHT=10.0
SSIM_WEIGHT=1.0
BCE=0

echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python shadowmap_main.py \
--iteration=$ITERATION \
--server_config=1 \
--num_workers=6 \
--batch_size=512 \
--num_blocks=6 --net_config=1 \
--l1_weight=$L1_WEIGHT --lpip_weight=$LPIP_WEIGHT --ssim_weight=$SSIM_WEIGHT --use_bce=$BCE --patch_size=64 \
--version_name="rgb2shadow_v8.01" --mode="azimuth" --min_epochs=100 --light_angle=0 \
--img_to_load=-1 --load_previous=0

srun python shadowmap_main.py \
--iteration=$ITERATION \
--server_config=1 \
--num_workers=6 \
--batch_size=512 \
--num_blocks=6 --net_config=1 \
--l1_weight=$L1_WEIGHT --lpip_weight=$LPIP_WEIGHT --ssim_weight=$SSIM_WEIGHT --use_bce=$BCE --patch_size=64 \
--version_name="rgb2shadow_v8.01" --mode="azimuth" --min_epochs=200 --light_angle=36 \
--img_to_load=-1 --load_previous=1

srun python shadowmap_main.py \
--iteration=$ITERATION \
--server_config=1 \
--num_workers=6 \
--batch_size=512 \
--num_blocks=6 --net_config=1 \
--l1_weight=$L1_WEIGHT --lpip_weight=$LPIP_WEIGHT --ssim_weight=$SSIM_WEIGHT --use_bce=$BCE --patch_size=64 \
--version_name="rgb2shadow_v8.01" --mode="azimuth" --min_epochs=300 --light_angle=72 \
--img_to_load=-1 --load_previous=1

srun python shadowmap_main.py \
--iteration=$ITERATION \
--server_config=1 \
--num_workers=6 \
--batch_size=512 \
--num_blocks=6 --net_config=1 \
--l1_weight=$L1_WEIGHT --lpip_weight=$LPIP_WEIGHT --ssim_weight=$SSIM_WEIGHT --use_bce=$BCE --patch_size=64 \
--version_name="rgb2shadow_v8.01" --mode="azimuth" --min_epochs=400 --light_angle=108 \
--img_to_load=-1 --load_previous=1

srun python shadowmap_main.py \
--iteration=$ITERATION \
--server_config=1 \
--num_workers=6 \
--batch_size=512 \
--num_blocks=6 --net_config=1 \
--l1_weight=$L1_WEIGHT --lpip_weight=$LPIP_WEIGHT --ssim_weight=$SSIM_WEIGHT --use_bce=$BCE --patch_size=64 \
--version_name="rgb2shadow_v8.01" --mode="azimuth" --min_epochs=500 --light_angle=144 \
--img_to_load=-1 --load_previous=1


conda deactivate