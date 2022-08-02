from argparse import Namespace
from typing import Optional
from tqdm import tqdm, trange
import prettytable as pt
import logging
import wandb
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from .utils import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        args: Namespace,
        model: PreTrainedModel,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        label_mapping: Optional[dict] = None,
    ):
        self.args = args
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.monitor_metric = 0.0
        if label_mapping:
            self.id2label = {id_: label for label, id_ in label_mapping.items()}

    @torch.no_grad()
    def evaluation(self):
        self.model.eval()
        predictions = []
        labels = []
        val_loss = 0.0
        val_iterator = tqdm(
            self.dev_dataloader, desc="Evaluation", total=len(self.dev_dataloader)
        )

        for batch in val_iterator:
            batch_labels = [
                [self.id2label[label] for label in label_seq]
                for label_seq in batch["labels"].numpy()
            ]
            labels.extend(batch_labels)
            batch_cuda = {
                item: value.to(self.args.device) for item, value in list(batch.items())
            }
            loss, logits = self.model(**batch_cuda)[:2]
            probs = torch.softmax(logits, dim=-1)

            if self.args.ngpus > 1:
                loss = loss.mean()

            val_loss += loss.item()
            batch_predictions = [
                [self.id2label[pred] for pred in pred_seq]
                for pred_seq in probs.argmax(dim=-1).cpu().numpy()
            ]
            predictions.extend(batch_predictions)

        avg_val_loss = val_loss / len(self.dev_dataloader)
        p, r, f1, acc = get_seqeuence_labeling_metrics(labels, predictions)
        metrics = {"p": p, "r": r, "f1": f1, "acc": acc, "avg_val_loss": avg_val_loss}
        return metrics

    def train(self):
        wandb.watch(self.model)

        self.model.to(self.args.device)
        if self.args.ngpus > 1 and self.args.device == "cuda":
            model = nn.DataParallel(self.model)
        else:
            model = self.model

        lr_scheduler, optimizer = self.build_optimizer()
        ema, fgm, pgd, pgd_attack_round = self.add_tricks()

        epoch_iterator = trange(self.args.num_train_epochs)
        global_steps = 0
        accumulated_train_loss = 0.0
        accumulated_train_loss_shadow = 0.0

        for _ in epoch_iterator:

            train_iterator = tqdm(
                self.train_dataloader, desc="Training", total=len(self.train_dataloader)
            )
            model.train()
            for batch in train_iterator:
                batch_cuda = {
                    item: value.to(self.args.device)
                    for item, value in list(batch.items())
                }
                loss = model(**batch_cuda)[0]
                if self.args.ngpus > 1:
                    loss = loss.mean()
                loss.backward()

                self.attack(batch_cuda, model, fgm, pgd, pgd_attack_round)

                self.step(lr_scheduler, optimizer)

                if self.args.ema:
                    ema.update()

                accumulated_train_loss += loss.item()
                global_steps += 1

                train_iterator.set_postfix_str(
                    f"running-training-loss: {loss.item():.4f}"
                )
                lr = lr_scheduler.get_last_lr()[0]
                wandb.log(
                    {"running training loss": loss.item(), "lr": lr}, step=global_steps
                )

                if global_steps % self.args.logging_steps == 0:
                    avg_train_loss = (
                        accumulated_train_loss - accumulated_train_loss_shadow
                    ) / self.args.logging_steps
                    accumulated_train_loss_shadow = accumulated_train_loss

                    if self.args.ema:
                        ema.apply_shadow()
                    metrics = self.evaluation()
                    metrics["avg_train_loss"] = avg_train_loss

                    self.after_eval(ema, global_steps, metrics, model)

        return model

    @staticmethod
    def step(lr_scheduler, optimizer):
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    def after_eval(self, ema, global_steps, metrics, model):
        assert (
            self.args.monitor_metric in metrics
        ), "monitor metric didn't been calculated"
        if metrics[self.args.monitor_metric] > self.monitor_metric:
            model_save_path = os.path.join(
                self.args.output_dir,
                f"checkpoint-{global_steps}-{metrics[self.args.monitor_metric]:.6f}",
            )
            self.model.save_pretrained(model_save_path)
            self.monitor_metric = metrics[self.args.monitor_metric]
        table = pt.PrettyTable(
            [
                "Train Global Steps",
                "Training Loss",
                "Eval Loss",
                "Eval Precision",
                "Eval Recall",
                "Eval F1",
                "Eval Acc",
            ]
        )
        pt_metrics = [
            metrics["avg_train_loss"],
            metrics["avg_val_loss"],
            metrics["p"],
            metrics["r"],
            metrics["f1"],
            metrics["acc"],
        ]
        table.add_row([str(global_steps)] + [f"{metric:.6f}" for metric in pt_metrics])
        logger.info(f"\n\n{table}\n")
        log_wandb_metrics = {
            "train loss": metrics["avg_train_loss"],
            "eval loss": metrics["avg_val_loss"],
            "eval precision": metrics["p"],
            "eval recall": metrics["r"],
            "eval f1": metrics["f1"],
            "eval acc": metrics["acc"],
        }
        wandb.log(log_wandb_metrics, step=global_steps)
        model.train()
        if self.args.ema:
            ema.restore()

    def attack(self, batch_cuda, model, fgm, pgd, pgd_attack_round):
        if self.args.adv == "fgm":
            fgm.attack(epsilon=self.args.eps)
            loss_adv = model(**batch_cuda)[0]
            if self.args.ngpus > 1:
                loss_adv = loss_adv.mean()
            loss_adv.backward()
            fgm.restore()
        elif self.args.adv == "pgd":
            pgd.backup_grad()
            for t in range(pgd_attack_round):
                pgd.attack(
                    epsilon=self.args.eps,
                    alpha=self.args.alpha,
                    is_first_attack=(t == 0),
                )
                if t != pgd_attack_round - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                loss_adv = model(**batch_cuda)[0]
                if self.args.ngpus > 1:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
            pgd.restore()

    def add_tricks(self):
        fgm = None
        pgd = None
        ema = None
        pgd_attack_round = 3
        if self.args.adv == "fgm":
            fgm = FGM(self.model)
        elif self.args.adv == "pgd":
            pgd = PGD(self.model)
        if self.args.ema:
            ema = EMA(self.model, decay=0.999)
        return ema, fgm, pgd, pgd_attack_round

    def build_optimizer(self):
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
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        total_steps = self.args.num_train_epochs * len(self.train_dataloader)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )
        return lr_scheduler, optimizer
