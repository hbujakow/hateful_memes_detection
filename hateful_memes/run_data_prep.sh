#!/bin/bash
#SBATCH --account=ganzha_23
#SBATCH --job-name=run_data_prep
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=0
#SBATCH --mem=5G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikolaj.galkowski.stud@pw.edu.pl
#SBATCH --output=/home2/faculty/mgalkowski/logs/baseline/baseline2.log

. /home2/faculty/mgalkowski/miniconda3/etc/profile.d/conda.sh
conda activate baseline

python data_prep_hf.py