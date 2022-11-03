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
srun python "gdown_download.py"
zip -F "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/v28_istd.zip" --out "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/v28_istd_fixed.zip"
unzip "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/v28_istd_fixed.zip" -d "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/v28_istd/"

conda list
conda deactivate