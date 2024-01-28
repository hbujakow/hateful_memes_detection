import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM, RobertaTokenizer
from typing import List, Tuple


class PromptHateModel(nn.Module):
    def __init__(
        self,
        label_words: List[str] = ["good", "bad"],
        max_length: int = 320,
        model_name: str = "roberta-large",
    ):
        """
        Prompt-based hate speech detection model.
        """
        super(PromptHateModel, self).__init__()
        self.roberta = RobertaForMaskedLM.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        self.mask_token_id = self.tokenizer.mask_token_id
        self.label_word_list = []
        for word in label_words:
            self.label_word_list.append(
                self.tokenizer._convert_token_to_id(
                    self.tokenizer.tokenize(" " + word)[0]
                )
            )

    def forward_single_cap(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor, mask_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for a single caption.
        Args:
            tokens (torch.Tensor): Input tokens.
            attention_mask (torch.Tensor): Attention mask.
            mask_pos (torch.Tensor): Mask position.
        Returns:
            torch.Tensor: Logits.
        """
        batch_size = tokens.size(0)
        mask_pos = mask_pos.squeeze()

        out = self.roberta(tokens, attention_mask)
        prediction_mask_scores = out.logits[torch.arange(batch_size), mask_pos]
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(
                prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1)
            )
        logits = torch.cat(logits, -1)
        return logits

    def generate_input_tokens(
        self, sents: List[str], max_length: int = 320
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate input tokens.
        Args:
            sents (List[str]): List of sentences.
            max_length (int): Maximum length.
        Returns:
            torch.Tensor: Tokens.
            torch.Tensor: Attention mask.
            torch.Tensor: Mask position.
        """
        token_info = self.tokenizer(
            sents,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = token_info.input_ids
        attention_mask = token_info.attention_mask
        mask_pos = [t.numpy().tolist().index(self.mask_token_id) for t in tokens]
        mask_pos = torch.LongTensor(mask_pos)
        return tokens, attention_mask, mask_pos

    def forward(self, all_texts: List[str]) -> torch.Tensor:
        """
        Passes the input through the model.
        Args:
            all_texts (List[str]): List of texts.
        Returns:
            torch.Tensor: Logits for each text.
        """
        tokens, attention_mask, mask_pos = self.generate_input_tokens(
            all_texts, self.max_length
        )
        logits = self.forward_single_cap(
            tokens.to(self.roberta.device),
            attention_mask.to(self.roberta.device),
            mask_pos.to(self.roberta.device),
        )  # B,2
        return logits
