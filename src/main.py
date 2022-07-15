import argparse
import wandb
import os

import torch
from transformers import BertTokenizerFast

from .train import Trainer
from .data_module import BaseDataModule
from .utils import seed_everything


def get_args():
    parser = argparse.ArgumentParser(description='For training')
    parser.add_argument("--debug", default=0, help="whether to debug", type=int)
    parser.add_argument("--task_name", default="", help="task name", type=str)
    parser.add_argument("--username", default="", help="username", type=str)
    parser.add_argument("--overwrite", default=0, help="whether to process data from scratch", type=int)
    parser.add_argument("--model_path", default='model/chinese-roberta-wwm-ext', type=str)
    parser.add_argument("--data_dir", default="./data", type=str, required=False, help="Path to data.")
    parser.add_argument("--output_dir", default="./outputs", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--monitor_metric", default='f1', type=str,
                        help="according to monitor metric to save best model, e.g. acc, f, r", )
    parser.add_argument("--dev_ratio", default=0.1, type=float, help="dev data ratio in total train data", )
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--logging_steps", default=50, type=int, help="Total number of steps to log metrics")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_ratio * total_steps.")
    parser.add_argument("--ema", default=1, type=int, choices=[0, 1], help="whether to use EMA")
    parser.add_argument("--ema_decay", default=0.999, type=float, help="decay in EMA")
    parser.add_argument("--tau", default=1000, type=int, help="ema decay correction factor")
    parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument('--device', choices=['cpu', 'cuda'], default="cuda:0",
                        help="Select which device to train model, defaults to gpu.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        args.ngpus = torch.cuda.device_count()

    seed_everything(args.seed)
    args_save = vars(args)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    data_module = BaseDataModule(args, tokenizer)
    train_dataloader, dev_dataloader = data_module.create_dataloader()

    wandb.init(project=args.task_name, entity=args.username, config=args_save, dir=args.output_dir)
    run_name = wandb.run.name
    args.output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(args.output_dir)

    trainer = Trainer(args, model, train_dataloader, dev_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
