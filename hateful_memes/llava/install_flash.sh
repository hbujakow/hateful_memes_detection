#!/bin/bash
#SBATCH --account=ganzha_23
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wiktor.jakubowski.stud@pw.edu.pl
#SBATCH --job-name=install_flash_attn
#SBATCH --output=/home2/faculty/wjakubowski/logs/llava/flash_install.log

. /home2/faculty/wjakubowski/miniconda3/etc/profile.d/conda.sh
conda activate llava

python install_flash.py