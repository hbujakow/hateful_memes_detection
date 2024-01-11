import os
import pickle as pkl
from argparse import ArgumentParser

import pandas as pd
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from tqdm import tqdm

PATH = "/home2/faculty/mgalkowski/memes_analysis/data/hateful_memes/"
OUTPUT_PATH = "/home2/faculty/mgalkowski/memes_analysis/data/hateful_memes/captions/"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


def generate_prompt_result(model, vis_processors, device, im, ques):
    image = vis_processors["eval"](im).float().unsqueeze(0).to(device)
    ans = model.generate(
        {"image": image, "prompt": ("Question: %s Answer:" % (ques))},
        length_penalty=3.0,
    )
    return ans[0]


def main(args) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
    )

    model = model.float()

    data_path = args.data_path
    train_memes = pd.read_json(os.path.join(data_path, "train.jsonl"), lines=True)
    dev_memes = pd.read_json(os.path.join(data_path, "dev_seen.jsonl"), lines=True)
    test_memes = pd.read_json(os.path.join(data_path, "test_seen.jsonl"), lines=True)

    train_memes = train_memes.loc[:, "img"]
    dev_memes = dev_memes.loc[:, "img"]
    test_memes = test_memes.loc[:, "img"]

    data = {"train": train_memes, "dev": dev_memes, "test": test_memes}

    categories = {
        "race": "what is the race of the person in the image?",
        "gender": "what is the gender of the person in the image?",
        "valid_animal": "is there an animal in the image?",
        "valid_person": "is there a person in the image?",
        "country": "which country does the person in the image come from?",
        "animal": "what animal is in the image?",
        "valid_disable": "are there disabled people in the image?",
        "religion": "what is the religion of the person in the image?",
    }

    generic_caption = {"generic": "describe briefly what is in the image"}

    print(f"Part for: {categories.keys()}")

    for data_split, dataset in tqdm(
        data.items(),
        desc="Generating captions",
        position=1,
        leave=False,
    ):
        if data_split == "test" and generic_caption != {}:
            categories.update(generic_caption)

        for category, question in tqdm(categories.items(), position=0):
            print("Generating captions for: ", category)
            captions = {}
            for i, path in enumerate(dataset):
                if i % 250 == 0:
                    print(
                        f"Already preprocessing {i*100/len(dataset):2f}% of the {data_split} dataset"
                    )
                # image = Image.open(os.path.join(data_path, "inpainted", path))
                image = Image.open(os.path.join(data_path, path))  # not inpainted
                caption = generate_prompt_result(
                    model, vis_processors, device, image, question
                )
                captions[path] = caption
            pkl.dump(
                captions,
                open(
                    os.path.join(args.output_path, f"{category}_{data_split}.pkl"), "wb"
                ),
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()
    main(args)
