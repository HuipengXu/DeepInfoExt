from transformers import PreTrainedModel, BertPreTrainedModel, BertModel
from torchcrf import CRF
import torch.nn as nn


class MyBert(PreTrainedModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError


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
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)
        loss = None
        outputs = [loss, logits]
        if labels is not None:
            loss = - self.crf(logits, labels, attention_mask.bool(), reduction='token_mean')
            outputs[0] = loss
        return outputs
