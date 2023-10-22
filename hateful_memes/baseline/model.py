from pathlib import Path

import torch
import torchvision
from torch import nn


class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        loss_fn,
        language_module,
        vision_module,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
    ):
        super(LanguageAndVisionConcat, self).__init__()
        self.language_module = language_module
        self.vision_module = vision_module
        self.fusion = torch.nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim),
            out_features=fusion_output_size,
        )
        self.fc = torch.nn.Linear(
            in_features=fusion_output_size, out_features=num_classes
        )
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, text, image, label=None):
        text_features = torch.nn.functional.relu(self.language_module(text))
        image_features = torch.nn.functional.relu(self.vision_module(image))
        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        logits = self.fc(fused)
        pred = torch.nn.functional.softmax(logits)
        # loss = self.loss_fn(pred, label) if label is not None else label
        return pred


class HatefulMemesModel(nn.Module):
    def __init__(self, hparams):
        super(HatefulMemesModel, self).__init__()

        for data_key in [
            "train_path",
            "dev_path",
            "img_dir",
        ]:
            # ok, there's one for-loop but it doesn't count
            if data_key not in hparams.keys():
                raise KeyError(f"{data_key} is a required hparam in this model")

        self.hparams = hparams

        self.embedding_dim = self.hparams.get("embedding_dim", 300)
        self.language_feature_dim = self.hparams.get("language_feature_dim", 300)
        self.vision_feature_dim = self.hparams.get(
            "vision_feature_dim", self.language_feature_dim
        )
        self.output_path = Path(self.hparams.get("output_path", "model-outputs"))

        self.model = self._build_model()

    def forward(self, text, image, label=None):
        return self.model(text, image, label)

    def _build_model(self):
        # we're going to pass the outputs of our text
        # transform through an additional trainable layer
        # rather than fine-tuning the transform
        language_module = torch.nn.Linear(
            in_features=self.embedding_dim, out_features=self.language_feature_dim
        )

        # easiest way to get features rather than
        # classification is to overwrite last layer
        # with an identity transformation, we'll reduce
        # dimension using a Linear layer, resnet is 2048 out
        vision_module = torchvision.models.resnet152(pretrained=True)
        vision_module.fc = torch.nn.Linear(
            in_features=2048, out_features=self.vision_feature_dim
        )

        return LanguageAndVisionConcat(
            num_classes=self.hparams.get("num_classes", 2),
            loss_fn=torch.nn.CrossEntropyLoss(),
            language_module=language_module,
            vision_module=vision_module,
            language_feature_dim=self.language_feature_dim,
            vision_feature_dim=self.vision_feature_dim,
            fusion_output_size=self.hparams.get("fusion_output_size", 512),
            dropout_p=self.hparams.get("dropout_p", 0.1),
        )
