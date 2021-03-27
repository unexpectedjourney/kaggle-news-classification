import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TextClassifier(nn.Module):
    def __init__(self, n_classes, pretrained_model_name, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            pretrained_model_name,
            return_dict=False
        )
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size, n_classes)
        )

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(pooled_output)
