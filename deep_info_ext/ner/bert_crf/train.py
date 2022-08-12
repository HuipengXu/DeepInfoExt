from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from ...common.train import Trainer


class CRFTrainer(Trainer):
    def build_optimizer(self):  # sourcery skip: invert-any-all
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in param_optimizer if n.endswith("transitions")],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.crf_lr,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        total_steps = self.args.num_train_epochs * len(self.train_dataloader)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

        return lr_scheduler, optimizer
