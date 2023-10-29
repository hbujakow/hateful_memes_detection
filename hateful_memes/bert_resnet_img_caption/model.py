import torch
import torchvision
from torch import nn


class HateClassifier(nn.Module):
    def __init__(
        self,
        language_model,
        vision_model: int = 512,
        features_dim: int = 768,
        fusion_output_size: int = 256,
        dropout_p: float = 0.15,
        num_classes: int = 2,
    ):
        super(HateClassifier, self).__init__()
        self.language_model = language_model
        self.vision_model = vision_model
        self.fusion = torch.nn.Linear(
            in_features=(features_dim * 2), out_features=fusion_output_size
        )
        self.class_layer = torch.nn.Linear(
            in_features=fusion_output_size, out_features=num_classes
        )
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, input_ids, attention_mask, token_type_ids, image):
        text_features = torch.nn.functional.relu(
            self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state[:, 0, :]
        )
        image_features = torch.nn.functional.relu(self.vision_model(image))
        combined = torch.cat([text_features, image_features], dim=1)

        fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        logits = self.class_layer(fused)
        pred = torch.nn.functional.softmax(logits)
        return pred
