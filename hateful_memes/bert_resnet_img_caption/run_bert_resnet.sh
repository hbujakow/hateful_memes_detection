#!/bin/bash
#SBATCH --account=ganzha_23
#SBATCH --job-name=run_bert_resnet
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=6
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikolaj.galkowski.stud@pw.edu.pl
#SBATCH --output=/home2/faculty/mgalkowski/logs/bert_resnet/bert_resnet.log

. /home2/faculty/mgalkowski/miniconda3/etc/profile.d/conda.sh
conda activate testowe

python train.py --config_path config.json