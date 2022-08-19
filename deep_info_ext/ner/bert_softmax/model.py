from typing import Optional

import torch
from transformers import BertForTokenClassification


class BertForNER(BertForTokenClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        loss, logits = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )[:2]

        predictions = logits.argmax(dim=-1)
        return loss, predictions
