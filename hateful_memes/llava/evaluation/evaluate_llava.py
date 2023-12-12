## TO BE REFACTORED -> refactored code will be in run_inference.py

import os
import sys
from pathlib import Path

sys.path.append("../LLaVA/")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from tqdm import tqdm
import pandas as pd

# Data
DATAPATH = "/home2/faculty/wjakubowski/memes_analysis/data/"

# Default model
model_path = "liuhaotian/llava-v1.5-13b"
model_base = None

# Fine-tuned model
model_path = "/home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/checkpoints/llava-v1.5-13b-lora"
model_base = "liuhaotian/llava-v1.5-13b"
# device_map = "auto"
# offload_folder="offload",


prompt = "Is this meme hateful?\nAnswer the question using a single word or phrase."

df_test = pd.read_json(DATAPATH + "dev_seen.jsonl", lines = True) 

for idx, row in tqdm(df_test.iterrows()):
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": model_base,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": DATAPATH + row["img"],
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    df_test.at[idx, "prediction"] = eval_model(args)

df_test.to_json(DATAPATH + 'dev_seen_predictions_finetuned.json')

print('Evaluation completed.')