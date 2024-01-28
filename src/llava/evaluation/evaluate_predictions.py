import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, Any, List

sys.path.append("../LLaVA/")

DATAPATH = "/home2/faculty/wjakubowski/memes_analysis/data/llava_preds/"


def get_datasets(data_path: str) -> List[str]:
    """
    Recursively traverses through the directory and its subdirectories, adding the full paths of all files to the output variable.
    """
    file_list = []

    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith("json"):
                file_list.append(file_path)
    return file_list


def evaluate_predictions(data_path: str) -> Dict[str, Any]:
    """
    Calculates the AUC and Accuracy for the predictions given by the specific model saved in data_path set.
    """

    data = pd.read_json(data_path)
    assert all(data["prediction"].isin(["Yes", "No"]))

    data["label_pred"] = np.where(data["prediction"] == "Yes", 1, 0)
    y = data["label"]
    y_pred = data["label_pred"].astype("Int64")

    pattern = r"(finetuned(?:_(\d+)_epochs)?|not_finetuned)\.json$"
    match = re.search(pattern, data_path)
    model_name = match.group(1)

    if "train" in data_path:
        set_var = "train"
    elif "dev" in data_path:
        set_var = "dev"
    elif "test" in data_path:
        set_var = "test"
    else:
        set_var = None

    return {
        "model": [model_name],
        f"auc_{set_var}": [round(roc_auc_score(y, y_pred), 4)],
        f"accuracy_{set_var}": [round(accuracy_score(y, y_pred), 4)],
    }


def main(args):
    datasets_list = get_datasets(args.data_path)

    df_results = pd.DataFrame(
        columns=[
            "model",
            "auc_train",
            "accuracy_train",
            "auc_dev",
            "accuracy_dev",
            "auc_test",
            "accuracy_test",
        ]
    )

    for dataset in datasets_list:
        print("evaluating:", dataset)
        results_dict = evaluate_predictions(dataset)
        print(results_dict)

        index = df_results.index[df_results["model"] == results_dict["model"][0]]

        if not index.empty:
            df_results.loc[index, results_dict.keys()] = results_dict.values()
        else:
            df_results = pd.concat(
                [df_results, pd.DataFrame(results_dict)], ignore_index=True
            )

    df_results["no_epochs"] = (
        df_results["model"]
        .apply(lambda x: re.search(r"\d+", x).group() if re.search(r"\d+", x) else "0")
        .astype(int)
    )

    df_results.sort_values("no_epochs").to_csv(DATAPATH + "results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA predictions")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DATAPATH,
        help="path to the data with LLaVA predictions",
    )
    args = parser.parse_args()
    main(args)
