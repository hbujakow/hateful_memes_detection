#!/bin/A--account=ganzha_23
#SBATCH --partition=long,short
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=a100:4
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wiktor.jakubowski.stud@pw.edu.pl
#SBATCH --job-name=inference
#SBATCH --output=/home2/faculty/wjakubowski/logs/llava/finetuned_inference_test.log

. /home2/faculty/wjakubowski/miniconda3/etc/profile.d/conda.sh
conda activate llava

export PYTHONPATH=/home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/

python run_inference.py \
    --test-file "/home2/faculty/wjakubowski/memes_analysis/data/dev_seen.jsonl" \
    --dest-file "/home2/faculty/wjakubowski/memes_analysis/data/dev_seen_with_predictions_finetuned.json"