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
#SBATCH --job-name=llava_finetune
#SBATCH --output=/home2/faculty/wjakubowski/logs/llava/train_7_12_2023_1gpu.log

. /home2/faculty/wjakubowski/miniconda3/etc/profile.d/conda.sh
conda activate llava

wandb login 6b0f39a9fb1b517ea0286d1dbf54c20229b387d8

export PYTHONPATH=/home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/

deepspeed /home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/llava/train/train_mem.py \
    --deepspeed /home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/scripts/zero3.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version v1 \
    --data_path /home2/faculty/wjakubowski/memes_analysis/data/llava_train.json \
    --image_folder /home2/faculty/wjakubowski/memes_analysis/data/img \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/checkpoints/llava-v1.5-13b-task \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
