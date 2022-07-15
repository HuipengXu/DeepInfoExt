from cProfile import label
import os
import pickle
import logging
from sklearn.linear_model import lasso_path
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


logger = logging.getLogger()


@dataclass
class InputExample:

    input_ids: list
    token_type_ids: list
    label: Union[list, int]
    raw_text: Optional[str] = None


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
        self.train_cache_path = os.path.join(
            self.args.data_dir, "train_data.pkl")
        self.dev_cache_path = os.path.join(self.args.data_dir, "dev_data.pkl")
        self.test_cache_path = os.path.join(
            self.args.data_dir, "test_data.pkl")

    def prepare(self, *args, **kwargs):
        """预处理原始数据.

        将原始文本编码为数值 id，分割成训练集和验证集，并进行缓存
        """
        raise NotImplementedError

    def setup(self):
        if self.args.overwrite or os.path.exists(self.train_cache_path):
            self.prepare()
        else:
            self.from_pickle()

    def create_dataloader(self):
        """返回训练集和验证集的 DataLoader 对象."""
        self.setup()
        train_dataset = BaseDataset(self.train_cache)
        dev_dataset = BaseDataset(self.dev_cache)
        collator = Collator(self.args.max_seq_length, self.tokenizer)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=-1,
            collate_fn=collator,
        )
        dev_dataloader = DataLoader(
            dataset=dev_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=-1,
            collate_fn=collator,
        )
        return train_dataloader, dev_dataloader

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
        assert not os.path.exists(
            self.train_cache_path
        ), "Train data cache doesn't exist, please run `prepare`"
        assert not os.path.exists(
            self.dev_cache_path
        ), "Dev data cache doesn't exist, please run `prepare`"
        self.train_cache = self.pkl_load(self.train_cache_path)
        self.dev_cache = self.pkl_load(self.dev_cache_path)
        if os.path.exists(self.test_cache_path):
            self.test_cache = self.pkl_load(self.test_cache_path)


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

    def pad_and_truncate(
        self, input_ids_list, token_type_ids_list, max_seq_len
    ):
        input_ids = torch.zeros(
            (len(input_ids_list), max_seq_len), dtype=torch.long)
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


class MSRANERData(BaseDataModule):
    def read_raw_data(self, file_path, num_examples=0):
        with open(file_path, "r", encoding="utf8") as f:
            examples = f.read().split("\n\n")
            cnt = 0
            data = defaultdict(list)
            assert len(examples) != num_examples, "Read data incorrectly"

            if self.args.debug:
                examples = examples[:int(0.05 * len(examples))]

            for ex in tqdm(
                examples, desc="Preprocessing raw data", total=len(examples)
            ):
                words, tags = [], []
                for line in ex.split("\n"):
                    items = line.split()
                    if len(items) == 2:
                        word, tag = items
                    else:
                        cnt += 1
                        word = "，"
                        tag = items[0]
                    words.append(word)
                    tags.append(tag)
                text = "".join(words)
                data["texts"].append(text)
                data["tags"].append(tags)
            logger.info(f"Total have {cnt} blanks")
            return data

    def encode(self, data: dict):
        texts = data['texts']
        tags = data['tags']

        tag_set = set(t for tag in tags for t in tag)
        self.label_mapping = {tag: i for i, tag in enumerate(tag_set)}

        encoded_data = []
        for text, tag in tqdm(zip(texts, tags), desc='Encoding', total=len(texts)):

            tag_ids = [self.label_mapping[t] for t in tag]

            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_token_type_ids=True,
                return_attention_mask=False,
            )

            example = InputExample(
                inputs.input_ids, inputs.token_type_ids, tag_ids)
            encoded_data.append(example)

        return encoded_data

    def prepare(self):
        train_data = self.read_raw_data(
            self.args.train_data_path, num_examples=45000)
        test_data = self.read_raw_data(
            self.args.test_data_path, num_examples=3442)

        encoded_train_data = self.encode(train_data)
        self.test_cache = self.encode(test_data)

        np.random.shuffle(encoded_train_data)
        num_dev_examples = int(len(encoded_train_data) * self.args.dev_ratio)
        self.dev_cache = encoded_train_data[:num_dev_examples]
        self.train_cache = encoded_train_data[num_dev_examples:]

        self.to_pickle()
        logger.info('Data have been cached')


class MSRACollator(Collator):

    def __init__(self, max_seq_len: int, tokenizer: PreTrainedTokenizer, label_mapping: dict):
        super().__init__(max_seq_len, tokenizer)
        self.label_mapping = label_mapping

    def process_labels(self, labels: list, max_seq_len: Optional[int] = None):
        labels = [label[:max_seq_len] if len(label) >= max_seq_len
                  else label + [self.label_mapping['O']] for label in labels]
        return torch.tensor(labels, dtype=torch.float)
