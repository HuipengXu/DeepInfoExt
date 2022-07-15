from transformers import PreTrainedModel, BertModel


class MyBert(PreTrainedModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError


class BertWithCRF(BertModel):
    
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)
        pass