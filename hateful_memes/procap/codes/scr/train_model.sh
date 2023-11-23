#!/bin/bash
#SBATCH --account=ganzha_23
#SBATCH --partition=short
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikolaj.galkowski.stud@pw.edu.pl
#SBATCH --job-name=train_procap
#SBATCH --output=/home2/faculty/mgalkowski/logs/procap/train_22_11_2023_16_05.log

. /home2/faculty/mgalkowski/miniconda3/etc/profile.d/conda.sh
conda activate captions_procap

python main.py --SAVE_NUM 221120231605 --SEED 1111 --DATASET 'mem' --LR_RATE 1.3e-05 --MODEL_NAME roberta-large
