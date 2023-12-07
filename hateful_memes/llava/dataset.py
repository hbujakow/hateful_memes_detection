from pathlib import Path
import pandas as pd
import json


DATAPATH = Path(__file__).resolve().parent.parent.parent / "data"


def create_json_entry(row: pd.Series):
    """
    Creates a json entry for a row in the dataset.
    """
    conversation = [
        {"from": "human", "value": "Is this meme hateful?\n<image>"},
        {"from": "gpt", "value": "Yes" if row["label"] == 1 else "No"},
    ]

    json_entry = {
        "id": str(row["id"]),
        "image": row["img"].replace("img/", ""),
        "conversations": conversation,
    }
    return json_entry


if __name__ == "__main__":
    df = pd.read_json(DATAPATH / "train.jsonl", lines=True)

    json_data = [create_json_entry(row) for _, row in df.iterrows()]

    result_json = json.dumps(json_data, indent=2)

    with open(DATAPATH / "llava_train.json", "w") as json_file:
        json_file.write(result_json)
