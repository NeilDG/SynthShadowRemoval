#!/bin/bash
#SBATCH -J ZDB
#SBATCH --partition=debug
#SBATCH --qos=240c-1h_debug
#SBATCH --ntasks=1
#SBATCH --output=script_debug.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=$1
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda/3-2021.11
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python shadow_train_main-3.py \
--server_config=0 --img_to_load=200000 --train_mode="train_shadow" \
--plot_enabled=0 --save_per_iter=500 --network_version="rgb2ns_v61.26_places" --iteration=$ITERATION

conda deactivate

