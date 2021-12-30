#!/bin/bash
#SBATCH -J INSTALL
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=1
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_install_gpu.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

#About this script:
# Installation of necessary libraries

module load anaconda
module load cuda/10.1_cudnn-7.6.5

#conda env remove --name NeilGAN_V2
conda create --name NeilGAN_V2 python=3.9
#source activate NeilGAN_V2
#conda update python

#do fresh install
source activate NeilGAN_V2
conda uninstall pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge scikit-image
pip install visdom
pip install kornia
pip install opencv-python
pip install --upgrade pillow
pip install lpips
conda list
conda deactivate

conda deactivate