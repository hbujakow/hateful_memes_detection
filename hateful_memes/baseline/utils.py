import json
from pathlib import Path


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config
