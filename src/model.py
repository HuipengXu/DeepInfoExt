from transformers import PreTrainedModel


class MyBert(PreTrainedModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
