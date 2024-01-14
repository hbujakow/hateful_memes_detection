# Hateful memes detection using LLaVA Vision-Language model

Download the repository and install packages as described in [here](https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#install).

**Disclaimer** If you run into problems, when running `pip install flash-attn --no-build-isolation` on cluster, run the job `finetuning/install_flash.sh` instead. This will install the package with the information about available CUDA version.

## Fine-tuning

1. Navigate to `finetuning/` directory.

2. Generate the hateful memes dataset in the desired format (like [this](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/detail_23k.json)), using the script `create_finetuning_dataset.py`.

3. Run the finetuning script `finetune_task.sh` (modify the paths and training specifications accordingly to your needs).
*Optional*: In order to report the training process in Weights&Biases:
    - create account in the W&B domain,
    - copy API key from [here](https://wandb.ai/settings#api) and paste it in the `finetune_task.sh`,
    - uncommment the line with `reports_to` argument in the script.

## Evaluation

1. Navigate to `evaluation/` directory.

2. Run `generate_predictions.sh` (modify the paths to data & model to your case).

3. Use `evaluate_predictions.py` script to evaluate the performance of the model by calculating metrics.
