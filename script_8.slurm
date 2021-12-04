#!/bin/bash
#SBATCH -J RM_8
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

srun python render_maps_main.py \
--iteration=$ITERATION \
--server_config=1 \
--num_workers=6 \
--batch_size=1024 \
--num_blocks=0 --net_config=2 \
--l1_weight=1.0 --lpip_weight=5.0 --ssim_weight=0.0 --bce_weight=10.0 --use_bce=1 --min_epochs=200 \
--version_name="rgb2normal_v1.08" --map_choice="normal" \
--img_to_load=-1 --load_previous=1

conda deactivate