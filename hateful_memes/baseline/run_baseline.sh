#!/bin/bash
#SBATCH --account=ganzha_23
#SBATCH --job-name=run_gen_baseline
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikolaj.galkowski.stud@pw.edu.pl
#SBATCH --output=/home2/faculty/mgalkowski/logs/baseline/baseline1.log

. /home2/faculty/mgalkowski/miniconda3/etc/profile.d/conda.sh
conda activate baseline

python main.py --config_path config.json --data_dir ../../data/hateful_memes --log_dir "../logs/baseline"