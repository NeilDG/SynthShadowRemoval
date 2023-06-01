#!/bin/bash
#SBATCH -J SM_3
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=6
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_3.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

NETWORK_VERSION=$1
ITERATION=$2
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda/3-2021.11
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python shadow_train_main.py \
--server_config=4 --img_to_load=-1 --train_mode="train_shadow_matte" \
--plot_enabled=0 --save_per_iter=500 --network_version=$NETWORK_VERSION --iteration=$ITERATION

conda deactivate