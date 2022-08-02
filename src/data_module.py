import os
import pickle
import logging
from tqdm import tqdm
from argparse import Namespace
from dataclasses import dataclass
from collections import defaultdict
from typing import Union, Optional, List

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from .utils import json_dump, json_load

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class InputExample:

    input_ids: list
    token_type_ids: list
    labels: Union[list, int]
    raw_text: Optional[str] = None
    offsets: Optional[List[tuple]] = None


class BaseDataset(Dataset):
    def __init__(self, data: List[InputExample]):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        example = [
            self.data[index].input_ids,
            self.data[index].token_type_ids,
            self.data[index].labels,
        ]
        return example

    def __len__(self):
        num_examples = len(self.data)
        return num_examples


class Collator:
    def __init__(self, max_seq_len: int, tokenizer: PreTrainedTokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def pad_and_truncate(self, input_ids_list, token_type_ids_list, max_seq_len):
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(
                    input_ids_list[i], dtype=torch.long
                )
                token_type_ids[i, :seq_len] = torch.tensor(
                    token_type_ids_list[i], dtype=torch.long
                )
                attention_mask[i, :seq_len] = 1
            else:
                input_ids[i] = torch.tensor(
                    input_ids_list[i][: max_seq_len - 1]
                    + [self.tokenizer.sep_token_id],
                    dtype=torch.long,
                )
                token_type_ids[i] = torch.tensor(
                    token_type_ids_list[i][:max_seq_len], dtype=torch.long
                )
                attention_mask[i] = 1

        return input_ids, token_type_ids, attention_mask

    def process_labels(self, labels: list, max_seq_len: Optional[int] = None):
        labels = torch.tensor(labels, dtype=torch.long)
        return labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, labels_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(
            input_ids_list, token_type_ids_list, max_seq_len
        )

        labels = self.process_labels(labels_list, max_seq_len)

        data_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return data_dict


class BaseDataModule:
    def __init__(
        self,
        args: Namespace,
        tokenizer: PreTrainedTokenizer,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.label_mapping = None
        self.train_cache = None
        self.dev_cache = None
        self.test_cache = None
        self.train_cache_path = os.path.join(self.args.data_dir, "train_data.pkl")
        self.dev_cache_path = os.path.join(self.args.data_dir, "dev_data.pkl")
        self.test_cache_path = os.path.join(self.args.data_dir, "test_data.pkl")
        self.setup()

    def prepare(self, *args, **kwargs):
        """预处理原始数据.

        将原始文本编码为数值 id，分割成训练集和验证集，并进行缓存
        """
        raise NotImplementedError

    def setup(self):
        label_mapping_path = os.path.join(self.args.data_dir, "label_mapping.json")
        if os.path.exists(self.train_cache_path):
            self.label_mapping = json_load(label_mapping_path)
        if self.args.overwrite or not os.path.exists(self.train_cache_path):
            self.prepare()
        else:
            self.from_pickle()

    def create_dataloader(
        self,
        dataset: Dataset = BaseDataset,
        collator: Collator = Collator,
    ):
        """返回训练集和验证集的 DataLoader 对象."""
        if self.args.do_train:
            train_dataset = dataset(self.train_cache)
            dev_dataset = dataset(self.dev_cache)
            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                collate_fn=collator,
            )
            dev_dataloader = DataLoader(
                dataset=dev_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=collator,
            )
            return train_dataloader, dev_dataloader
        else:
            test_dataset = dataset(self.test_cache)
            test_dataloader = DataLoader(
                dataset=test_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=collator,
            )
            return test_dataloader

    @staticmethod
    def pkl_dump(data: dict, data_path: str):
        with open(data_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def pkl_load(data_path: str):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data

    def to_pickle(self):
        self.pkl_dump(self.train_cache, self.train_cache_path)
        self.pkl_dump(self.dev_cache, self.dev_cache_path)
        if self.test_cache is not None:
            self.pkl_dump(self.test_cache, self.test_cache_path)

    def from_pickle(self):
        assert os.path.exists(
            self.train_cache_path
        ), "Train data cache doesn't exist, please run `prepare`"
        assert os.path.exists(
            self.dev_cache_path
        ), "Dev data cache doesn't exist, please run `prepare`"
        self.train_cache = self.pkl_load(self.train_cache_path)
        self.dev_cache = self.pkl_load(self.dev_cache_path)
        if os.path.exists(self.test_cache_path):
            self.test_cache = self.pkl_load(self.test_cache_path)


class MSRANERData(BaseDataModule):
    def read_raw_data(self, file_path, num_examples=0):
        length = []
        with open(file_path, "r", encoding="utf8") as f:
            examples = f.read().strip().split("\n\n")
            cnt = 0
            data = defaultdict(list)
            assert (
                len(examples) == num_examples
            ), f"Read data incorrectly, {len(examples)}:{num_examples}"

            if self.args.debug:
                examples = examples[: int(0.01 * len(examples))]

            for ex in tqdm(
                examples, desc="Preprocessing raw data", total=len(examples)
            ):
                words, tags = [], []
                lines = ex.split("\n")
                length.append(len(lines))
                for line in lines:
                    items = line.split()
                    if len(items) == 2:
                        word, tag = items
                    else:
                        cnt += 1
                        word = items[0]
                        tag = "O"
                    words.append(word)
                    tags.append(tag)
                text = "".join(words)
                data["texts"].append(text)
                data["tags"].append(tags)
                data["raw_examples"].append(ex)
            logger.info(
                f"Total have {cnt} blanks, 0.99 quantile: {np.quantile(length, 0.99):.2f}, 0.95 quantile: {np.quantile(length, 0.95):.2f}"
            )
            return data

    def encode(self, data: dict):
        texts = data["texts"]
        tags = data["tags"]
        raw_examples = data["raw_examples"]

        tag_set = set(t for tag in tags for t in tag)
        if not self.label_mapping:
            self.label_mapping = {tag: i for i, tag in enumerate(tag_set)}
            label_mapping_path = os.path.join(self.args.data_dir, "label_mapping.json")
            json_dump(label_mapping_path, self.label_mapping)

        encoded_data = []
        for j, (text, tag) in tqdm(
            enumerate(zip(texts, tags)), desc="Encoding", total=len(texts)
        ):

            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_token_type_ids=True,
                return_attention_mask=False,
                return_offsets_mapping=True,
            )

            offset_mapping = inputs.offset_mapping[1:-1]
            merged_tag = []
            for offset in offset_mapping:
                span_tag = []
                for i in range(offset[0], offset[-1]):
                    if not span_tag:
                        span_tag.append(tag[i])
                    elif tag[i] != span_tag[-1]:
                        span_tag.append(tag[i])
                if len(span_tag) == 1:
                    cur_tag = span_tag[0]
                elif (
                    len(span_tag) == 2
                    and span_tag[0].startswith("B")
                    and span_tag[-1].startswith("I")
                ):
                    cur_tag = span_tag[0]
                else:
                    logger.info(
                        f"Entity is cut error: {text[:offset[0]]}-{text[offset[0]: offset[-1]]}-{text[offset[-1]:]}"
                    )
                    cur_tag = "O"

                if (
                    cur_tag.startswith("I")
                    and len(merged_tag) > 0
                    and merged_tag[-1] == "O"
                ):
                    cur_tag == "O"
                merged_tag.append(cur_tag)

            merged_tag = ["O"] + merged_tag + ["O"]
            assert len(merged_tag) == len(
                inputs.input_ids
            ), "Label's length must be equal to input's length"

            tag_ids = [self.label_mapping[t] for t in merged_tag]

            example = InputExample(
                inputs.input_ids,
                inputs.token_type_ids,
                tag_ids,
                raw_examples[j],
                offset_mapping,
            )
            encoded_data.append(example)

        return encoded_data

    def prepare(self):
        train_data_path = os.path.join(self.args.data_dir, "msra_train_bio.txt")
        test_data_path = os.path.join(self.args.data_dir, "msra_test_bio.txt")
        train_data = self.read_raw_data(train_data_path, num_examples=45000)
        test_data = self.read_raw_data(test_data_path, num_examples=3442)

        encoded_train_data = self.encode(train_data)
        self.test_cache = self.encode(test_data)

        np.random.shuffle(encoded_train_data)
        num_dev_examples = int(len(encoded_train_data) * self.args.dev_ratio)
        self.dev_cache = encoded_train_data[:num_dev_examples]
        self.train_cache = encoded_train_data[num_dev_examples:]

        self.to_pickle()
        logger.info("Data have been cached")


class MSRACollator(Collator):
    def __init__(
        self,
        max_seq_len: int,
        tokenizer: PreTrainedTokenizer,
        label_mapping: dict,
        train: bool = True,
    ):
        super().__init__(max_seq_len, tokenizer)
        self.label_mapping = label_mapping
        self.train = train

    def process_labels(self, labels: list, max_seq_len: Optional[int] = None):
        labels = [
            label[:max_seq_len]
            if len(label) >= max_seq_len
            else label + [self.label_mapping["O"]] * (max_seq_len - len(label))
            for label in labels
        ]
        return torch.tensor(labels, dtype=torch.long)

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, labels_list = list(zip(*examples))
        max_seq_len = max(len(input_id) for input_id in input_ids_list)
        if self.train:
            max_seq_len = min(max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(
            input_ids_list, token_type_ids_list, max_seq_len
        )

        labels = self.process_labels(labels_list, max_seq_len)

        data_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return data_dict
