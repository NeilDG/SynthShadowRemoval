#!/bin/bash
#SBATCH -J DOWNLOAD
#SBATCH --partition=serial
#SBATCH --qos=84c-1d_serial
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --output=script_download.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

#About this script:
# Installation of necessary libraries

module load anaconda
module load cuda/10.1_cudnn-7.6.5

#do fresh install
source activate NeilGAN_V2
pip install gdown
gdown https://drive.google.com/uc?id=1B6erEfeB2_n_KYP5GtCQU-DHGE1bKEVE "scratch2/SynthWeather Dataset 10/"
unzip "scratch2/SynthWeather Dataset 10/v26.zip"

conda list
conda deactivate