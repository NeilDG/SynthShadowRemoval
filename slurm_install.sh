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

#do fresh install
pip-review --local --auto
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install scikit-learn
pip install scikit-image
pip install visdom
pip install kornia
pip install opencv-python
pip install --upgrade pillow
pip install lpips
pip install gputil
