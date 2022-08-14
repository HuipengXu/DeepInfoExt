import os
import time
import wandb
from argparse import Namespace

import torch
import torch.distributed as dist
from transformers import BertTokenizerFast, BertConfig

from .train import CRFTrainer
from .model import BertWithCRF
from ...common.predict import Predictor
from ...common.args import get_default_parser
from ...common.data_module import NERDataModule, BaseDataModule
from ...common.utils import (
    seed_everything,
    select_device,
    LOGGER,
    torch_distributed_zero_first,
)


LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def get_args():
    parser = get_default_parser()
    parser.add_argument(
        "--crf_lr", default=0.1, help="learning rate for crf layer", type=float
    )
    return parser.parse_args()


def do_train(args: Namespace, data_module: BaseDataModule, config: BertConfig):
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

    model = BertWithCRF.from_pretrained(args.model_path, config=config)

    train_dataloader = data_module.create_dataloader(
        data_cache=data_module.train_cache, shuffle=True, rank=LOCAL_RANK
    )
    dev_dataloader = None
    if RANK in {-1, 0}:
        dev_dataloader = data_module.create_dataloader(
            data_cache=data_module.dev_cache, shuffle=False, rank=-1
        )

    trainer = CRFTrainer(
        args,
        model,
        train_dataloader,
        dev_dataloader,
        label_mapping=data_module.label_mapping,
    )
    trainer.train()


def do_predict(args: Namespace, data_module: BaseDataModule, config: BertConfig):
    test_dataloader = data_module.create_dataloader(
        data_cache=data_module.test_cache, shuffle=False, rank=-1
    )
    model = BertWithCRF.from_pretrained(args.save_model_path, config=config)
    predictor = Predictor(args, model, test_dataloader, data_module.label_mapping)
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
        data_module = NERDataModule(args, tokenizer)

    config = BertConfig.from_pretrained(
        args.model_path, num_labels=len(data_module.label_mapping)
    )

    if args.do_train:
        do_train(args, data_module, config)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info("Destroying process group... ")
            dist.destroy_process_group()
    else:
        do_predict(args, data_module, config)


if __name__ == "__main__":
    main()
