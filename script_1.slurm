#!/bin/bash
#SBATCH -J Z1
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=6
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_1.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=1
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda/3-2021.11
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python iid_train_v3.py \
--shadow_matte_network_version="v58.70" --shadow_removal_version="v58.29" --shadow_matte_iteration=$ITERATION --shadow_removal_iteration=1 \
--server_config=1 --plot_enabled=0 --img_to_load=-1 --train_mode=train_shadow

conda deactivate