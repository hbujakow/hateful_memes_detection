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
#SBATCH --job-name=generate_captions
#SBATCH --output=/home2/faculty/mgalkowski/logs/procap/caption_generation_4.log

. /home2/faculty/mgalkowski/miniconda3/etc/profile.d/conda.sh
conda activate captions_procap

python generate_captions.py
