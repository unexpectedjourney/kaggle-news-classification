import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class BertForSequenceClassification(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes
        )

        self.model = AutoModel.from_pretrained(pretrained_model_name, config=config)

        self.dropout1 = nn.Dropout(dropout)
        self.l1 = nn.Linear(config.hidden_size*6, config.hidden_size)
        self.bn1 = torch.nn.LayerNorm(config.hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.l2 = torch.nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, head_mask=None, text=None):
        assert attention_mask is not None, "attention mask is none"
        bert_output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask,
            output_hidden_states=True, return_dict=True,
        )

        seq_output1 = bert_output['hidden_states'][-1]  # (bs, seq_len, dim)
        seq_output2 = bert_output['hidden_states'][-4]
        seq_output3 = bert_output['hidden_states'][-7]
        # seq_output4 = bert_output['hidden_states'][-10]
        # seq_output2 = detoxify_output['hidden_states'][-1] # (bs, seq_len, dim)

        # mean pooling, i.e. getting average representation of all tokens
        pooled_output1 = seq_output1.mean(axis=1)  # (bs, dim)
        pooled_output2 = seq_output2.mean(axis=1)  # (bs, dim)
        pooled_output3 = seq_output3.mean(axis=1)  # (bs, dim)
        pooled_output4, _ = seq_output1.max(dim=1)  # (bs, dim)
        pooled_output5, _ = seq_output2.max(dim=1)  # (bs, dim)
        pooled_output6, _ = seq_output3.max(dim=1)  # (bs, dim)
        # pooled_output1 = bert_output['pooler_output']
        # pooled_output2 = seq_output2.mean(axis=1)  # (bs, dim)

        # and concat it
        pooled_output = torch.cat([pooled_output1, pooled_output2, pooled_output3, pooled_output4, pooled_output5, pooled_output6], dim=1)  # (bs, 6*dim)
        pooled_output = self.dropout1(pooled_output)  # (bs, dim)
        pooled_output = self.l1(pooled_output)  # (bs, dim)
        pooled_output = self.bn1(pooled_output)  # (bs, dim)
        pooled_output = nn.Tanh()(pooled_output)
        pooled_output = self.dropout2(pooled_output)  # (bs, dim)
        pooled_output = self.l2(pooled_output)  # (bs, num_classes)

        # probs = self.sigmoid(scores)

        return pooled_output

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


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
