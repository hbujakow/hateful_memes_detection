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
#SBATCH --output=/home2/faculty/mgalkowski/logs/procap/train_100_01_2024_roberta_large_10_epochs_inpainted.log

. /home2/faculty/mgalkowski/miniconda3/etc/profile.d/conda.sh
conda activate captions_procap

# distilroberta-base
# roberta-large
# roberta-base
# vinai/bertweet-large https://huggingface.co/vinai/bertweet-large

python main.py --SEED 1111 --DATASET 'mem' --LR_RATE 1.3e-05 --EPOCHS 10 --MODEL_NAME roberta-large
