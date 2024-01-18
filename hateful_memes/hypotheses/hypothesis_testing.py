import argparse
import pandas as pd
import torch
from delong_test import delong_roc_test


def main(args):
    data = pd.read_json(args.data_path, lines=True)
    true_labels = data[args.labels_name].values
    probas_model1 = torch.load(args.probas_first_model_path).cpu()
    probas_model2 = torch.load(args.probas_second_model_path).cpu()

    p_value = delong_roc_test(true_labels, probas_model1, probas_model2)
    print(
        "DeLong test of {} functionality returned p_value = {}".format(
            args.test, p_value
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        type=str,
        default="de_long_test",
        help="Functionality of proposed solution to test.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home2/faculty/wjakubowski/memes_analysis/data/dev_seen.jsonl",
    )
    parser.add_argument("--labels-name", type=str, default="label")
    parser.add_argument(
        "--probas-first-model-path",
        type=str,
        default="/home2/faculty/wjakubowski/memes_analysis/data/averaged_probs/inpainted/vinai_bertweet_large_epochs_15_avg_probs.pt",
        help="Probabilities for the second class predicted by the first model",
    )
    parser.add_argument(
        "--probas-second-model-path",
        type=str,
        default="/home2/faculty/wjakubowski/memes_analysis/data/averaged_probs/not_inpainted/vinai_bertweet_large_epochs_15_avg_probs.pt",
        help="Probabilities for the second class predicted by the second model",
    )
    args = parser.parse_args()

    main(args)
