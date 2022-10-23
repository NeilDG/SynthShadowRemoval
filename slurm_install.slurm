#!/bin/bash
#SBATCH -J INSTALL
#SBATCH --partition=serial
#SBATCH --qos=84c-1d_serial
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --output=script_install.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

#About this script:
# Installation of necessary libraries

module load anaconda
module load cuda/10.1_cudnn-7.6.5

conda env remove --name NeilGAN_V2
conda create --name NeilGAN_V2 python=3.9
source activate NeilGAN_V2
conda update python

#do fresh install
source activate NeilGAN_V2
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
#conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
#conda install -c conda-forge scikit-image
pip install visdom
pip install kornia
pip install opencv-python
pip install --upgrade pillow
pip install lpips
conda list
conda deactivate

conda deactivate