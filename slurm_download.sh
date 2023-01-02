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
SERVER_CONFIG=2 #1 = COARE, 2 = CCS Cloud

module load anaconda/3-2021.11
module load cuda/10.1_cudnn-7.6.5
source activate NeilGAN_V2

pip install --upgrade --no-cache-dir gdown

if [ $SERVER_CONFIG == 1 ]
then
  srun python "gdown_download.py" --server_config=$SERVER_CONFIG
else
  python "gdown_download.py" --server_config=$SERVER_CONFIG
fi

DATASET_NAME="v44_places"

if [ $SERVER_CONFIG == 1 ]
then
  OUTPUT_DIR="/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/"
else
  OUTPUT_DIR="/home/jupyter-neil.delgallego/SynthWeather Dataset 10/"
fi

echo "$OUTPUT_DIR/$DATASET_NAME.zip"

zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"
mv "$OUTPUT_DIR/$DATASET_NAME+fixed" "$OUTPUT_DIR/$DATASET_NAME"
rm -rf "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"

if [ $SERVER_CONFIG == 2 ]
then
  python "ccs2_main.py"
fi