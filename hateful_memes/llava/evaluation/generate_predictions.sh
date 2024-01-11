#!/bin/bash
#SBATCH --account=ganzha_23
#SBATCH --partition=long,short
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wiktor.jakubowski.stud@pw.edu.pl
#SBATCH --job-name=inference
#SBATCH --output=/home2/faculty/wjakubowski/logs/llava/20240112_inference_test.log

. /home2/faculty/wjakubowski/miniconda3/etc/profile.d/conda.sh
conda activate llava

export PYTHONPATH=/home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/

# evaluation of general LLaVA without finetuning

python /home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/evaluation/run_inference.py \
    --test-file "/home2/faculty/wjakubowski/memes_analysis/data/train.jsonl" \
    --dest-file "/home2/faculty/wjakubowski/memes_analysis/data/llava_preds/train/train_with_predictions_not_finetuned.json" \
    --model-path "/home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/checkpoints/llava-v1.5-13b-lora"

python /home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/evaluation/run_inference.py \
    --test-file "/home2/faculty/wjakubowski/memes_analysis/data/dev_seen.jsonl" \
    --dest-file "/home2/faculty/wjakubowski/memes_analysis/data/llava_preds/dev/dev_seen_with_predictions_not_finetuned.json" \
    --model-path "/home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/checkpoints/llava-v1.5-13b-lora"

python /home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/evaluation/run_inference.py \
    --test-file "/home2/faculty/wjakubowski/memes_analysis/data/test_seen.jsonl" \
    --dest-file "/home2/faculty/wjakubowski/memes_analysis/data/llava_preds/test/test_seen_with_predictions_not_finetuned.json" \
    --model-path "/home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/checkpoints/llava-v1.5-13b-lora"

# evaluation of different epochs in finetuning
for (( epoch=2; epoch<=10; epoch+=2 )); do
    python /home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/evaluation/run_inference.py \
    --test-file "/home2/faculty/wjakubowski/memes_analysis/data/train.jsonl" \
    --dest-file "/home2/faculty/wjakubowski/memes_analysis/data/llava_preds/train/train_with_predictions_finetuned_${epoch}_epochs.json" \
    --model-path "/home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/checkpoints/llava-v1.5-13b-lora-${epoch}-epochs"

    python /home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/evaluation/run_inference.py \
        --test-file "/home2/faculty/wjakubowski/memes_analysis/data/dev_seen.jsonl" \
        --dest-file "/home2/faculty/wjakubowski/memes_analysis/data/llava_preds/dev/dev_seen_with_predictions_finetuned_${epoch}_epochs.json" \
        --model-path "/home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/checkpoints/llava-v1.5-13b-lora-${epoch}-epochs"

    python /home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/evaluation/run_inference.py \
        --test-file "/home2/faculty/wjakubowski/memes_analysis/data/test_seen.jsonl" \
        --dest-file "/home2/faculty/wjakubowski/memes_analysis/data/llava_preds/test/test_seen_with_predictions_finetuned_${epoch}_epochs.json" \
        --model-path "/home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/checkpoints/llava-v1.5-13b-lora-${epoch}-epochs"
done