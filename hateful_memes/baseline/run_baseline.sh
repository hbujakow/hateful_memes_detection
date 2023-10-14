#!/bin/bash
#SBATCH --account=ganzha_23
#SBATCH --job-name=run_gen
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikolaj.galkowski.stud@pw.edu.pl
#SBATCH --output=/home2/faculty/mgalkowski/logs/baseline

conda activate baseline

python main.py --config_path config.json --data_dir ../../data/hateful_memes --log_dir "../logs/baseline" --model_dir "../model/baseline"