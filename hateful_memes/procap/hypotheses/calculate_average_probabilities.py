import os
import re
import pickle
import argparse
import numpy as np
import torch
from typing import List

def get_model_types(directory: str):
    return ['_'.join(subdir.split('_', 2)[2:]) for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]

def calculate_average_probs(input_data_dir: str, output_data_dir: str, model_type_list: List[str], inpainting_type: str, seed_list: List[int]):
    """
    Calculates tensor of average probabilities for each model trained with several different seeds.
    Saves the results to respective pytorch tensor files.
    """
    avg_probs = {model: None for model in model_type_list}
    num_seeds = len(seed_list)


    for model_type in model_type_list:
        for seed in seed_list:
            seed_folder = os.path.join(input_data_dir, f"seed_{seed}", inpainting_type)
            for model_folder in os.listdir(seed_folder):
                if model_type in model_folder:
                    for file in os.listdir(os.path.join(seed_folder, model_folder)):
                        if bool(re.compile(r"probs_\d+").search(file)): # checks for file with probabilities
                            with open(os.path.join(seed_folder, model_folder, file), 'rb') as f:
                                probs = torch.load(f, map_location=torch.device('cpu'))[:, 1]
                            if avg_probs[model_type] is None:
                                avg_probs[model_type] = probs
                            else:
                                avg_probs[model_type] += probs
        avg_probs[model_type] /= num_seeds
        output_file = os.path.join(output_data_dir, inpainting_type, model_type +  "_avg_probs.pt")
        torch.save(torch.tensor(avg_probs[model_type]), output_file)

def main(args):
    input_data_dir = args.input_data_dir
    output_data_dir = args.output_data_dir
    seed_list = [1111, 2222, 3333, 4444, 5555]
    model_type_list = get_model_types(input_data_dir + "/seed_1111/inpainted")

    inpainted_avg_probs = calculate_average_probs(input_data_dir, output_data_dir, model_type_list, "inpainted", seed_list)
    not_inpainted_avg_probs = calculate_average_probs(input_data_dir, output_data_dir, model_type_list, "not_inpainted", seed_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default="/home2/faculty/wjakubowski/memes_analysis/data/procap_probs",
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default="/home2/faculty/wjakubowski/memes_analysis/data/averaged_probs",
    )
    args = parser.parse_args()

    main(args)