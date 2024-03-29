import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

from ...common.crf import CRF

class BertWithCRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)
        self.bert = BertModel(config)
        self.config = config
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        sequence_output = outputs[2][-1]
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)
        loss = None
        outputs = [loss, logits]
        if labels is not None:
            loss = -self.crf(
                logits, labels, attention_mask.bool(), reduction="token_mean"
            )
            outputs[0] = loss
        if not self.training:
            outputs[1] = self.crf.decode(logits, mask=attention_mask.bool())
            outputs[1] = torch.tensor(outputs[1], device=input_ids.device)
        return outputs
