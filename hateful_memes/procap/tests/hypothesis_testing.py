import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pycmp import delong_roc_test


def perform_delong_test(
    auc_model1, auc_model2, true_labels, probas_model1, probas_model2
) -> float:
    """
    Perform DeLong test for two models.
    Args:
        auc_model1: AUC score of the first model.
        auc_model2: AUC score of the second model.
        true_labels: True labels.
        probas_model1: Predicted probabilities of the first model.
        probas_model2: Predicted probabilities of the second model.
    Returns:
        p_value: p-value of the DeLong test.
    """

    # Calculate p-value using DeLong test
    p_value = delong_roc_test(
        true_labels, probas_model1, probas_model2, auc_model1, auc_model2, alpha=0.05
    )

    return p_value


def main(args):
    # Load true labels and predicted probabilities
    data = pd.read_jsonl(args.datapath_path, lines=True)
    true_labels = data[args.labels_name].values
    probas_model1 = np.load(args.probas_first_model_path)
    probas_model2 = np.load(args.probas_second_model_path)

    auc_model1 = roc_auc_score(true_labels, probas_model1)
    auc_model2 = roc_auc_score(true_labels, probas_model2)

    p_value = perform_delong_test(
        auc_model1, auc_model2, true_labels, probas_model1, probas_model2
    )
    print(
        "DeLong test of {} functionality returned p_value = {}".format(
            args.tested_functionality, p_value
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tested_functionality",
        type=str,
        choices=["inpainting", "distillation"],
        help="Functionality of proposed solution to test.",
    )
    parser.add_argument(
        "--datapath-path",
        type=str,
        default="/home2/faculty/wjakubowski/memes_analysis/data/llava_preds",
    )
    parser.add_argument("labels-name", type=str, default="label")
    parser.add_argument(
        "--probas-first-model-path",
        type=str,
        default="/home2/faculty/wjakubowski/memes_analysis/data/llava_preds",
    )
    parser.add_argument(
        "--probas-second-model-path",
        type=str,
        default="/home2/faculty/wjakubowski/memes_analysis/data/llava_preds",
    )
    args = parser.parse_args()

    main(args)
