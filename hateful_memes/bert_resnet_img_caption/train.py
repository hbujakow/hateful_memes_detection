import argparse
import json
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import torchvision
from datasets import load_dataset
from model import HateClassifier
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, DefaultDataCollator

MLFLOW_TRACKING_URI = "/home2/faculty/mgalkowski/memes_analysis/mlflow_data"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("hateful_memes")
mlflow.pytorch.autolog()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config


def initialize_models(config: dict):
    # easiest way to get features rather than
    # classification is to overwrite last layer
    # with an identity transformation, we'll reduce
    # dimension using a Linear layer, resnet is 2048 out
    bert_model = BertModel.from_pretrained(config["language_model"]["checkpoint"])

    out_features = config["model_dim"]
    vision_model = torchvision.models.resnet152(pretrained=True)
    vision_model.fc = torch.nn.Linear(in_features=2048, out_features=out_features)

    return bert_model, vision_model


def main(args):
    hf_data_loading_script_path = args.hf_data_path
    config = load_config(args.config_path)
    language_model_config = config["language_model"]
    hparams = config["hparams"]

    lang_model, vision_model = initialize_models(config)
    tokenizer = BertTokenizer.from_pretrained(language_model_config["checkpoint"])

    model = HateClassifier(lang_model, vision_model, features_dim=config["model_dim"])

    train_loader, dev_loader, test_loader = prepare_datasets(
        hf_data_loading_script_path, tokenizer, hparams, language_model_config
    )

    train_evaluate_model(model, hparams, train_loader, dev_loader)


def prepare_datasets(path: Path, tokenizer, hparams, language_model_config):
    data = load_dataset(path)

    def tokenization(example):
        return tokenizer(
            example["caption"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    train_data = data["train"].map(tokenization, batched=True)
    val_data = data["validation"].map(tokenization, batched=True)
    test_data = data["test"].map(tokenization, batched=True)

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

    def transforms(examples):
        examples["image"] = [
            image_transform(img.convert("RGB")) for img in examples["image"]
        ]
        return examples

    train_data.set_transform(transforms)
    val_data.set_transform(transforms)
    test_data.set_transform(transforms)

    train_data = train_data.select_columns(
        ["image", "input_ids", "token_type_ids", "attention_mask", "label"]
    )
    val_data = val_data.select_columns(
        ["image", "input_ids", "token_type_ids", "attention_mask", "label"]
    )
    test_data = test_data.select_columns(
        ["image", "input_ids", "token_type_ids", "attention_mask", "label"]
    )

    data_collator = DefaultDataCollator()

    train_loader = DataLoader(
        train_data, batch_size=hparams.get("batch_size", 64), collate_fn=data_collator
    )
    val_loader = DataLoader(
        val_data, batch_size=hparams.get("batch_size", 64), collate_fn=data_collator
    )
    test_loader = DataLoader(
        test_data, batch_size=hparams.get("batch_size", 64), collate_fn=data_collator
    )

    return train_loader, val_loader, test_loader


def train_evaluate_model(model, hparams: dict, train_loader, dev_loader):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=hparams.get("lr", 1e-3))

    num_epochs = hparams.get("num_epochs", 10)

    timestamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    with mlflow.start_run(run_name=f"resnet52_bert_concat_{timestamp}"):
        mlflow.log_artifact("config.json")

        for epoch in range(num_epochs):
            model.train()

            total_loss = 0.0
            num_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                optimizer.zero_grad()

                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                token_type_ids = batch["token_type_ids"]
                image = batch["image"]
                label = batch["labels"]

                preds = model(input_ids, attention_mask, token_type_ids, image)

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
                    input_ids = val_batch["input_ids"]
                    attention_mask = val_batch["attention_mask"]
                    token_type_ids = val_batch["token_type_ids"]
                    image = val_batch["image"]
                    label = val_batch["labels"]

                    val_preds = model(input_ids, attention_mask, token_type_ids, image)

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
        "--hf_data_path",
        default="/home2/faculty/mgalkowski/memes_analysis/hateful_memes/dataset_hf/dataset_hateful_memes.py",
        help="Path to dataset loading script.",
    )

    args = arg_parser.parse_args()
    main(args=args)
