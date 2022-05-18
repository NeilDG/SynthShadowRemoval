#!/bin/bash
#SBATCH -J R_DB
#SBATCH --partition=debug
#SBATCH --qos=240c-1h_debug
#SBATCH --ntasks=1
#SBATCH --output=script_debug.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=5
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda/3-2021.11
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python relighting_main.py \
--iteration=$ITERATION \
--server_config=1 \
--batch_size=320 \
--patch_size=64 \
--plot_enabled=0 \
--version_name="maps2rgb_rgb2maps_v4.08" --min_epochs=50 --net_config=1 --num_blocks=6 \
--img_to_load=-1 --load_previous=0


conda deactivate