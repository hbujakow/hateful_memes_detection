import os
import pickle
import numpy as np
import torch
from typing import List

def get_subdirectories(directory: str):
    return [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]

def calculate_average_probs(input_data_dir: str, output_data_dir: str, model_type_list: List[str], inpainting_type: str, seed_list: List[int]):
    """
    Calculates tensor of average probabilities for each model trained with several different seeds.
    Saves the results to respective pytorch tensor files.
    """
    avg_probs = np.zeros(2)
    num_seeds = len(seed_list)


    for model_type in model_type_list:
        for seed in seed_list:
            seed_folder = os.path.join(input_data_dir, f"seed_{seed}", model_type)

            inpainted_probs_file = f"probs_{seed_folder.split('/')[-1]}.pkl"
            not_inpainted_probs_file = f"probs_query_{seed_folder.split('/')[-1]}.pkl"

            inpainted_probs_path = os.path.join(seed_folder, inpainted_probs_file)
            not_inpainted_probs_path = os.path.join(seed_folder.replace("inpainted", "not_inpainted"), not_inpainted_probs_file)

            if os.path.exists(inpainted_probs_path) and os.path.exists(not_inpainted_probs_path):
                with open(inpainted_probs_path, 'rb') as f:
                    inpainted_probs = pickle.load(f)

                with open(not_inpainted_probs_path, 'rb') as f:
                    not_inpainted_probs = pickle.load(f)

                avg_probs[0] += inpainted_probs
                avg_probs[1] += not_inpainted_probs

        avg_probs /= num_seeds
        
        output_file = os.path.join(output_data_dir, inpainting_type, model_type, "avg_probs.pt")
        torch.save(torch.tensor(avg_probs), output_file)

def main(args):
    input_data_dir = args.input_data_dir
    output_dir = args.output_data_dir
    seed_list = [1111, 2222, 3333, 4444, 5555]
    model_type_list = get_subdirectories(input_data_dir)

    inpainted_avg_probs = calculate_average_probs(input_data_dir, output_data_dir, model_type_list, "inpainted", seed_list)
    not_inpainted_avg_probs = calculate_average_probs(input_data_dir, output_data_dir, model_type_list, "not_inpainted", seed_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default="/home2/faculty/mgalkowski/memes_analysis/hateful_memes/procap/architecture/logits_probs",
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default="/home2/faculty/wjakubowski/memes_analysis/data/averaged_probs",
    )
    args = parser.parse_args()

    main(args)