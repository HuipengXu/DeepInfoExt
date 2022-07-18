import argparse
from pandas import date_range
import wandb
import time
import os

import torch
from transformers import BertTokenizerFast, BertConfig

from .train import Trainer
from .model import BertWithCRF
from .utils import seed_everything
from .data_module import MSRANERData, MSRACollator


def get_args():
    parser = argparse.ArgumentParser(description="For training")
    parser.add_argument("--debug", default=0, help="whether to debug", type=int)
    parser.add_argument("--task_name", default="NER", help="task name", type=str)
    parser.add_argument("--username", default="xuhuipeng", help="username", type=str)
    parser.add_argument(
        "--overwrite", default=0, help="whether to process data from scratch", type=int
    )
    parser.add_argument("--model_path", default="ptms/rbt3", type=str)
    parser.add_argument(
        "--data_dir", default="./data", type=str, required=False, help="Path to data."
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs",
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--monitor_metric",
        default="f1",
        type=str,
        help="according to monitor metric to save best model, e.g. acc, f, r",
    )
    parser.add_argument(
        "--dev_ratio",
        default=0.1,
        type=float,
        help="dev data ratio in total train data",
    )
    parser.add_argument(
        "--num_workers",
        default=2,
        type=int,
        help="how many subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=2,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--logging_steps",
        default=100,
        type=int,
        help="Total number of steps to log metrics",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.1,
        type=float,
        help="Linear warmup over warmup_ratio * total_steps.",
    )
    parser.add_argument(
        "--ema", default=1, type=int, choices=[0, 1], help="whether to use EMA"
    )
    parser.add_argument("--ema_decay", default=0.999, type=float, help="decay in EMA")
    parser.add_argument(
        "--tau", default=1000, type=int, help="ema decay correction factor"
    )
    parser.add_argument("--adv", default=None, type=str, help="use fgm or pgd")
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cuda:0",
        help="Select which device to train model, defaults to gpu.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if not torch.cuda.is_available():
        args.ngpus = 0
    else:
        args.ngpus = torch.cuda.device_count()

    seed_everything(args.seed)
    args_save = vars(args)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    data_module = MSRANERData(args, tokenizer)
    config = BertConfig.from_pretrained(
        args.model_path, num_labels=len(data_module.label_mapping)
    )
    model = BertWithCRF.from_pretrained(args.model_path, config=config)
    collator = MSRACollator(args.max_seq_length, tokenizer, data_module.label_mapping)
    train_dataloader, dev_dataloader = data_module.create_dataloader(collator=collator)

    wandb.init(
        project=args.task_name,
        entity=args.username,
        config=args_save,
        dir=args.output_dir,
    )
    run_name = wandb.run.name if wandb.run.name else str(time.time())
    args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir)

    trainer = Trainer(args, model, train_dataloader, dev_dataloader, label_mapping=data_module.label_mapping)
    trainer.train()


if __name__ == "__main__":
    main()
