import os
import shutil

import pandas as pd
from tqdm import tqdm

if not os.path.exists("../data/data_hf"):
    os.makedirs("../data/data_hf")

train_data = pd.read_json("../data/hateful_memes/train.jsonl", lines=True)
dev_data = pd.read_json("../data/hateful_memes/dev_seen.jsonl", lines=True)
test_data = pd.read_json("../data/hateful_memes/test_seen.jsonl", lines=True)


data_split = ["train", "dev", "test"]

data = {"train": train_data, "dev": dev_data, "test": test_data}

for split in tqdm(data):
    if not os.path.exists(f"../data/data_hf/{split}"):
        os.makedirs(f"../data/data_hf/{split}")

    data[split]["img_num"] = data[split]["img"].apply(lambda x: x.split("/")[1])
    curr_data = data[split]

    with open(f"../data/data_hf/{split}/metadata.tsv", "w+", encoding="utf-8") as f:
        f.write("file_name\timg_text\n")
        dst_path = f"../data/data_hf/{split}"
        for row in tqdm(curr_data.iterrows()):
            src_path = f"../data/hateful_memes/img/{row[1]['img_num']}"
            shutil.copy(src_path, dst_path)

            f.write(f"{row[1]['img_num']}\t{row[1]['text']}\n")
