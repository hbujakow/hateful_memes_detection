import argparse
import os
import sys
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append("../LLaVA/")

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image
import math
from typing import List


def split_list(lst: List, n: int) -> List[List]:
    """
    Splits a list into n (roughly) equal-sized chunks

    Args:
        lst (List): The list to be split
        n (int): The number of chunks to split the list into

    Returns:
        List[List]: A list of n (roughly) equal-sized chunks
    """
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def predict(args) -> None:
    """
    Runs inference on a pretrained model to make predictions on test data.

    Args:
        args: Command-line arguments.
    """
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    df = pd.read_json(args.test_file, lines=True)

    if model.config.mm_use_im_start_end:
            question = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + args.question
            )
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + args.question

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .cuda()
    )

    input_token_len = input_ids.shape[1]

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    for image_file in tqdm(df["img"]):
        image = Image.open(os.path.join(args.data_folder, image_file))
        image_tensor = process_images(
            [image],
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        
        n_diff_input_output = (
        (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )

        output = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        output = output.strip()
        if output.endswith(stop_str):
            output = output[: -len(stop_str)].strip()

        df.loc[df["img"] == image_file, "prediction"] = output

    df.to_json(args.dest_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home2/faculty/wjakubowski/memes_analysis/hateful_memes/llava/LLaVA/checkpoints/llava-v1.5-13b-lora-10-epochs")
    parser.add_argument("--model-base", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--data-folder", type=str, default="/home2/faculty/wjakubowski/memes_analysis/data/")
    parser.add_argument("--test-file", type=str, default="/home2/faculty/wjakubowski/memes_analysis/data/dev_seen.jsonl")
    parser.add_argument("--dest-file", type=str, default="/home2/faculty/wjakubowski/memes_analysis/data/dev_seen_predictions_10_epochs.json")
    parser.add_argument("--question", type=str, default="Is this meme hateful?\nAnswer the question using a single word or phrase.")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    predict(args)
