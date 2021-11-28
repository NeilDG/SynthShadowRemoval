#!/bin/bash
#SBATCH -J RM_3
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=6
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_4.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=1

echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python render_maps_main.py \
--iteration=$ITERATION \
--server_config=1 \
--num_workers=6 \
--batch_size=256 \
--num_blocks=3 --net_config=3 \
--l1_weight=0.0 --lpip_weight=10.0 --ssim_weight=0.0 --use_bce=0 --min_epochs=120 \
--version_name="rgb2albedo_v1.08" --map_choice="albedo" \
--img_to_load=-1 --load_previous=0

conda deactivate