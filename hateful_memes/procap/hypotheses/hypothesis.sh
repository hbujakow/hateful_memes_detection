#!/bin/bash
#SBATCH --account=ganzha_23
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wiktor.jakubowski.stud@pw.edu.pl
#SBATCH --job-name=hypothesis_testing
#SBATCH --output=/home2/faculty/wjakubowski/logs/memes/20240112_inpainting_test.log

. /home2/faculty/wjakubowski/miniconda3/etc/profile.d/conda.sh
conda activate llava

python /home2/faculty/wjakubowski/memes_analysis/hateful_memes/procap/hypotheses/hypothesis_testing.py
