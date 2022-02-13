#!/bin/bash
#SBATCH -J S_2
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=6
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_2.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=6
L1_WEIGHT=10.0
LPIP_WEIGHT=1.0
SSIM_WEIGHT=0.0
BCE=1

echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python shading_main.py \
--iteration=$ITERATION \
--server_config=1 \
--num_workers=6 \
--batch_size=128 \
--net_config=1 --num_blocks=6 \
--l1_weight=$L1_WEIGHT --lpip_weight=$LPIP_WEIGHT --ssim_weight=$SSIM_WEIGHT --use_bce=$BCE --patch_size=128 \
--version_name="rgb2shading_v8.07" --min_epochs=100 \
--img_to_load=-1 --load_previous=0


conda deactivate