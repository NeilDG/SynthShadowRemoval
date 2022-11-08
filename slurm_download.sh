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
#Download of dataset

pip install gdown
srun python "gdown_download.py"

OUTPUT_DIR="/home/neil_delgallego/SynthWeather Dataset 10/"

zip -F $OUTPUT_DIR+"/v31_istd.zip" --out $OUTPUT_DIR+"/v31_istd_fixed.zip"
unzip $OUTPUT_DIR+"/v31_istd_fixed.zip" -d $OUTPUT_DIR+
mv $OUTPUT_DIR+"/v31_istd_fixed" $OUTPUT_DIR+"/v31_istd"