import os
import time
import wandb
import argparse
from argparse import Namespace

import torch
import torch.distributed as dist
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification

from .train import Trainer
from .predict import Predictor
from .model import BertWithCRF
from .data_module import MSRANERData, BaseDataModule
from .utils import seed_everything, select_device, LOGGER, torch_distributed_zero_first


LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def get_args():
    parser = argparse.ArgumentParser(description="For training")
    parser.add_argument("--debug", default=0, help="whether to debug", type=int)
    parser.add_argument("--do_train", default=1, help="train or test", type=int)
    parser.add_argument("--task_name", default="NER", help="task name", type=str)
    parser.add_argument("--username", default="xuhuipeng", help="username", type=str)
    parser.add_argument(
        "--overwrite", default=0, help="whether to process data from scratch", type=int
    )
    parser.add_argument("--model_path", default="ptms/rbt3", type=str)
    parser.add_argument("--save_model_path", default="ptms/rbt3", type=str)
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
        default=50,
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
    parser.add_argument("--eps", default=0.5, type=float, help="epsilon for fgm or pgd")
    parser.add_argument("--alpha", default=0.3, type=float, help="alpha for pgd")
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Select which device to train model, defaults to gpu.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Automatic DDP Multi-GPU argument, do not modify",
    )
    return parser.parse_args()


def do_train(args: Namespace, data_module: BaseDataModule):
    if RANK in {-1, 0}:
        args_save = vars(args)

        wandb.init(
            project=args.task_name,
            entity=args.username,
            config=args_save,
            dir=args.output_dir,
        )
        run_name = wandb.run.name or str(time.time())
        args.output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(args.output_dir)

    config = BertConfig.from_pretrained(
        args.model_path, num_labels=len(data_module.label_mapping)
    )
    # model = BertWithCRF.from_pretrained(args.model_path, config=config)
    model = BertForTokenClassification.from_pretrained(args.model_path, config=config)

    train_dataloader = data_module.create_dataloader(
        data_cache=data_module.train_cache, shuffle=True, rank=LOCAL_RANK
    )
    dev_dataloader = None
    if RANK in {-1, 0}:
        dev_dataloader = data_module.create_dataloader(
            data_cache=data_module.dev_cache, shuffle=False, rank=-1
        )

    trainer = Trainer(
        args,
        model,
        train_dataloader,
        dev_dataloader,
        label_mapping=data_module.label_mapping,
    )
    trainer.train()


def do_predict(args: Namespace, data_module: BaseDataModule):
    test_dataloader = data_module.create_dataloader(
        data_cache=data_module.test_cache, shuffle=False, rank=-1
    )
    predictor = Predictor(args, test_dataloader, data_module.label_mapping)
    predictor.predict()


def main():
    args = get_args()
    args.ngpus = torch.cuda.device_count()
    args.device = select_device(args.device, batch_size=args.batch_size)

    seed_everything(args.seed + 1 + RANK, deterministic=True)

    if LOCAL_RANK != -1 and args.do_train == 1:
        assert (
            args.batch_size % WORLD_SIZE == 0
        ), f"--batch-size {args.batch_size} must be multiple of WORLD_SIZE"
        assert (
            torch.cuda.device_count() > LOCAL_RANK
        ), "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        args.device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_module = MSRANERData(args, tokenizer)

    if args.do_train:
        do_train(args, data_module)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info("Destroying process group... ")
            dist.destroy_process_group()
    else:
        do_predict(args, data_module)


if __name__ == "__main__":
    main()
