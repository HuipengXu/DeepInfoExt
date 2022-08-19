import os
import wandb
import prettytable as pt
from typing import Optional
from tqdm import tqdm, trange
from argparse import Namespace

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from .utils import *


class Trainer:
    def __init__(
        self,
        args: Namespace,
        model: PreTrainedModel,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        label_mapping: dict,
    ):
        self.args = args
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.monitor_metric = 0.0
        self.id2label = {id_: label for label, id_ in label_mapping.items()}
        self.metric = ChunkEvaluator(self.id2label)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.metric.reset()
        val_loss = 0.0
        val_iterator = tqdm(
            self.dev_dataloader, desc="Evaluating", total=len(self.dev_dataloader)
        )

        for batch in val_iterator:
            lengths = batch["attention_mask"].sum(dim=-1).numpy()
            batch_cuda = {
                item: value.to(self.args.device) for item, value in list(batch.items())
            }
            loss, predictions = self.model(**batch_cuda)[:2]

            if self.args.ngpus > 1:
                loss = loss.mean()

            val_loss += loss.item()

            self.metric.update(lengths, predictions, batch["labels"])

        avg_val_loss = val_loss / len(self.dev_dataloader)
        p, r, f1, acc = self.metric.accumulate()
        return {"p": p, "r": r, "f1": f1, "acc": acc, "avg_val_loss": avg_val_loss}

    def train(self):  # sourcery skip: low-code-quality
        if RANK in {-1, 0}:
            wandb.watch(self.model)

        lr_scheduler, optimizer = self.build_optimizer()

        if self.args.device.type != "cpu":
            model = self.model.to(self.args.device)
            if LOCAL_RANK != -1:
                model = smart_DDP(self.model, local_rank=LOCAL_RANK)
            elif self.args.ngpus > 1:
                model = nn.DataParallel(model)
        else:
            model = self.model

        ema, fgm, pgd, pgd_attack_round = self.add_tricks()

        epoch_iterator = trange(
            self.args.num_train_epochs, desc="Epoch", disable=RANK not in {-1, 0}
        )

        global_steps = 0
        accumulated_train_loss = 0.0
        accumulated_train_loss_shadow = 0.0

        optimizer.zero_grad()
        for epoch in epoch_iterator:

            if RANK != -1:
                self.train_dataloader.sampler.set_epoch(epoch)

            train_iterator = tqdm(
                self.train_dataloader,
                desc="Training",
                total=len(self.train_dataloader),
                disable=RANK not in {-1, 0},
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

                nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                self.step(lr_scheduler, optimizer)

                if ema:
                    ema.update()

                accumulated_train_loss += loss.item()
                global_steps += 1

                train_iterator.set_postfix_str(
                    f"running-training-loss: {loss.item():.4f}"
                )
                lr = lr_scheduler.get_last_lr()[0]

                if RANK in {-1, 0}:
                    wandb.log(
                        {"running training loss": loss.item(), "lr": lr},
                        step=global_steps,
                    )

                if RANK in {-1, 0} and global_steps % self.args.logging_steps == 0:
                    avg_train_loss = (
                        accumulated_train_loss - accumulated_train_loss_shadow
                    ) / self.args.logging_steps
                    accumulated_train_loss_shadow = accumulated_train_loss

                    if ema:
                        ema.apply_shadow()
                    metrics = self.evaluate()
                    metrics["avg_train_loss"] = avg_train_loss

                    self.save_and_log(global_steps, metrics)
                    if ema:
                        ema.restore()
                    model.train()

        return self.model

    @staticmethod
    def step(lr_scheduler, optimizer):
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    def save_and_log(self, global_steps, metrics):
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
        LOGGER.info(f"\n\n{table}\n")
        log_wandb_metrics = {
            "train loss": metrics["avg_train_loss"],
            "eval loss": metrics["avg_val_loss"],
            "eval precision": metrics["p"],
            "eval recall": metrics["r"],
            "eval f1": metrics["f1"],
            "eval acc": metrics["acc"],
        }
        wandb.log(log_wandb_metrics, step=global_steps)

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
        if self.args.ema and RANK in {-1, 0}:
            ema = EMA(self.model, decay=0.999)
        return ema, fgm, pgd, pgd_attack_round

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
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        total_steps = self.args.num_train_epochs * len(self.train_dataloader)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

        return lr_scheduler, optimizer
