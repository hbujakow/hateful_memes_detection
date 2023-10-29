import argparse
from datetime import datetime
from pathlib import Path

import fasttext
import mlflow
import pandas as pd
import torch
import torchvision
from data_utils import HatefulMemesDataset
from model import HatefulMemesModel
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import load_config

MLFLOW_TRACKING_URI = "/home2/faculty/mgalkowski/memes_analysis/mlflow_data"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("hateful_memes")
mlflow.pytorch.autolog()


def main(*args, **kwargs):
    arguments = args[0]
    data_dir = Path(arguments.data_dir)
    log_dir = Path(arguments.log_dir)

    config = load_config(Path(arguments.config_path))
    hparams = config["hparams"]

    train_loader, dev_loader, test_loader = data_preparation(data_dir, hparams)

    train_test_model(hparams, train_loader, dev_loader)


def data_preparation(data_dir: Path, hparams: dict):
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev_seen.jsonl"
    test_path = data_dir / "test_seen.jsonl"

    image_dim = hparams.get("image_dim", 224)

    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(image_dim, image_dim)),
            torchvision.transforms.ToTensor(),
            # all torchvision models expect the same
            # normalization mean and std
            # https://pytorch.org/docs/stable/torchvision/models.html
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    train_samples_frame = pd.read_json(train_path, lines=True)

    with open("train_prepared.txt", "w", encoding="utf-8") as f:
        for i in range(len(train_samples_frame)):
            f.write(train_samples_frame.loc[i, "text"] + "\n")

    text_transform = fasttext.train_unsupervised(
        "train_prepared.txt",
        model=hparams.get("fasttext_model", "cbow"),
        dim=hparams.get("embedding_dim", 224),
    )

    train_dataset = HatefulMemesDataset(
        data_path=train_path,
        img_dir=data_dir,
        image_transform=image_transform,
        text_transform=text_transform,
        dev_limit=(hparams.get("dev_limit", None)),
        balance=True,
    )

    dev_dataset = HatefulMemesDataset(
        data_path=dev_path,
        img_dir=data_dir,
        image_transform=image_transform,
        text_transform=text_transform,
        dev_limit=None,
        balance=False,
    )

    test_dataset = HatefulMemesDataset(
        data_path=test_path,
        img_dir=data_dir,
        image_transform=image_transform,
        text_transform=text_transform,
        dev_limit=None,
        balance=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.get("batch_size", 4),
        shuffle=True,
        num_workers=hparams.get("num_workers", 16),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=hparams.get("batch_size", 4),
        shuffle=False,
        num_workers=hparams.get("num_workers", 16),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=hparams.get("batch_size", 4),
        shuffle=False,
        num_workers=hparams.get("num_workers", 16),
    )

    return train_loader, dev_loader, test_loader


def train_test_model(hparams: dict, train_loader: DataLoader, dev_loader: DataLoader):
    model = HatefulMemesModel(hparams)
    optimizer = optim.AdamW(model.parameters(), lr=hparams.get("lr", 0.001))
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = hparams.get("num_epochs", 10)

    timestamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    with mlflow.start_run(run_name=f"baseline_{timestamp}"):
        mlflow.log_artifact("config.json")

        for epoch in range(num_epochs):
            model.train()

            total_loss = 0.0
            num_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                optimizer.zero_grad()

                text, image, label = batch["text"], batch["image"], batch["label"]
                preds = model(text, image)

                loss = loss_fn(preds, label)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_loss / num_batches

            model.eval()

            total_val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for val_batch in tqdm(
                    dev_loader, desc=f"Validation - Epoch {epoch + 1}/{num_epochs}"
                ):
                    text, image, label = (
                        val_batch["text"],
                        val_batch["image"],
                        val_batch["label"],
                    )
                    val_preds = model(text, image)

                    val_loss = loss_fn(val_preds, label)

                    total_val_loss += val_loss.item()
                    num_val_batches += 1

            avg_val_loss = total_val_loss / num_val_batches

            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {avg_val_loss:.4f}"
            )

            metrics = {"train_loss": avg_train_loss, "val_loss": avg_val_loss}

            mlflow.log_metrics(metrics, step=epoch)

        mlflow.pytorch.log_model(model, f"baseline_model_{timestamp}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config_path",
        default="config.json",
        help="Path to a config file listing data, model, and trainer parameters.",
    )
    arg_parser.add_argument(
        "--data_dir",
        default="../../data",
        help="The input data dir. Should contain the data files for the task.",
    )
    arg_parser.add_argument(
        "--log_dir",
        default="../logs/baseline",
        help="The output directory where the logs will be written.",
    )
    args = arg_parser.parse_args()
    main(args)
