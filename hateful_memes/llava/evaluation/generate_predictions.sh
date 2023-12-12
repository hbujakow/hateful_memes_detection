#!/bin/bash
#SBATCH --account=ganzha_23
#SBATCH --partition=long
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=a100:3
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wiktor.jakubowski.stud@pw.edu.pl
#SBATCH --job-name=simple_inference
#SBATCH --output=/home2/faculty/wjakubowski/logs/llava/finetuned_inference.log

. /home2/faculty/wjakubowski/miniconda3/etc/profile.d/conda.sh
conda activate llava

export PYTHONPATH=/home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/

python evaluate_llava.py