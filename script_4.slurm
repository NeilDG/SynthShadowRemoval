#!/bin/bash
#SBATCH -J CY_4
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
module load anaconda/3-2021.11
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python cyclegan_main.py \
--iteration=$ITERATION \
--server_config=1 \
--img_per_iter=8 \
--batch_size=4 \
--patch_size=32 \
--version_name="synth2rgb_v4.01" --min_epochs=15 --net_config=1 --num_blocks=9 \
--img_to_load=-1 --load_previous=0


conda deactivate