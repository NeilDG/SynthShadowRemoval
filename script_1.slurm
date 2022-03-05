#!/bin/bash
#SBATCH -J S_1
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=6
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_1.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=5
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python relighting_main.py \
--iteration=$ITERATION \
--server_config=1 \
--num_workers=6 \
--batch_size=64 \
--net_config=1 --num_blocks=6 \
--patch_size=128 \
--version_name="maps2rgb_rgb2maps_v2.00" --mode=azimuth --min_epochs=100 \
--img_to_load=-1 --load_previous=0


conda deactivate