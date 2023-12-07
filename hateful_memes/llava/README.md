# Hateful memes detections using LLaVA Vision-Language model

## Fine-tuning
1. Generate the hateful memes dataset in the desired format (like [this](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/detail_23k.json)), using the script `create_finetuning_dataset.py`.

2. Download the repository and install packages as described in [here](https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#install).

**Disclaimer** If you run into problems, when running `pip install flash-attn --no-build-isolation` on cluster, run the job `install_flash.sh` instead. This will install the package with the information about available CUDA version.

3. Run the finetuning script `finetune_task_memems.sh` - (modify the paths).

