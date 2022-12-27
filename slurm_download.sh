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

module load anaconda/3-2021.11
module load cuda/10.1_cudnn-7.6.5
source activate NeilGAN_V2

pip install --upgrade --no-cache-dir gdown
python "gdown_download.py"

DATASET_NAME="v48_places"
OUTPUT_DIR="/home/jupyter-neil.delgallego/SynthWeather Dataset 10/"
#OUTPUT_DIR="/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/"
echo "$OUTPUT_DIR/$DATASET_NAME.zip"

zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"
mv "$OUTPUT_DIR/$DATASET_NAME+fixed" "$OUTPUT_DIR/$DATASET_NAME"
rm -rf "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"

python "ccs1_main.py"
