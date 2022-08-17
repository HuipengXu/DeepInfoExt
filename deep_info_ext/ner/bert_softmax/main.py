import os
from argparse import Namespace

import torch
import torch.distributed as dist
from transformers import BertTokenizerFast, BertConfig

from .model import BertForNER
from ...common.train import Trainer
from ...common.predict import Predictor
from ...common.args import get_default_parser
from ...common.data_module import NERDataModule, BaseDataModule
from ...common.utils import (
    RANK,
    LOGGER,
    LOCAL_RANK,
    WORLD_SIZE,
    ddp_init,
    wandb_init,
    select_device,
    seed_everything,
    torch_distributed_zero_first,
)


def get_args():
    parser = get_default_parser()
    return parser.parse_args()


def do_train(args: Namespace, data_module: BaseDataModule, config: BertConfig):
    if RANK in {-1, 0}:
        wandb_init(args)

    model = BertForNER.from_pretrained(args.model_path, config=config)

    train_dataloader = data_module.create_dataloader(
        data_cache=data_module.train_cache, shuffle=True, rank=LOCAL_RANK
    )

    dev_dataloader = (
        data_module.create_dataloader(
            data_cache=data_module.dev_cache, shuffle=False, rank=-1
        )
        if RANK in {-1, 0}
        else None
    )

    trainer = Trainer(
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
    model = BertForNER.from_pretrained(
        args.save_model_path, config=config
    )
    predictor = Predictor(args, model, test_dataloader, data_module.label_mapping)
    predictor.predict()


def main():
    args = get_args()
    args.ngpus = torch.cuda.device_count()
    args.device = select_device(args.device, batch_size=args.batch_size)

    seed_everything(args.seed + 1 + RANK, deterministic=True)

    ddp_init(args)

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
