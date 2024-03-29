import os
import sys
import pickle
import traceback
from tqdm import tqdm
from argparse import Namespace
from dataclasses import dataclass
from collections import defaultdict
from typing import Union, Optional, List

import numpy as np

import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .utils import json_dump, json_load, LOGGER, LOCAL_RANK, RANK, WORLD_SIZE


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
        return [
            self.data[index].input_ids,
            self.data[index].token_type_ids,
            self.data[index].labels,
        ]

    def __len__(self):
        return len(self.data)


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
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


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
        if not self.args.overwrite and os.path.exists(self.train_cache_path):
            self.label_mapping = json_load(label_mapping_path)
        if self.args.overwrite or not os.path.exists(self.train_cache_path):
            self.prepare()
        else:
            self.from_pickle()

    def create_dataloader(
        self,
        data_cache: Optional[list] = None,
        shuffle=True,
        rank=-1,
        CustomDataset: Dataset = BaseDataset,
    ):
        """返回训练集,验证集的 DataLoader 对象."""
        batch_size = self.args.batch_size // WORLD_SIZE
        if not shuffle:  # eval or test
            batch_size *= 2
        nd = torch.cuda.device_count()  # number of CUDA devices
        nw = min(
            [
                os.cpu_count() // max(nd, 1),
                batch_size if batch_size > 1 else 0,
                self.args.num_workers,
            ]
        )  # number of workers
        dataset = CustomDataset(data_cache)
        sampler = None if rank == -1 else DistributedSampler(dataset, shuffle=shuffle)
        collator = NERCollator(
            self.args.max_seq_length,
            self.tokenizer,
            self.label_mapping,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=nw,
            pin_memory=True,
            collate_fn=collator,
        )

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


class NERDataModule(BaseDataModule):
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
                examples = examples[: int(self.args.debug_ratio * len(examples))]

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
                        word = "，"
                        tag = "O"
                    words.append(word)
                    tags.append(tag)
                text = "".join(words)
                assert len(text) == len(
                    tags
                ), f"words: {len(words)}, text: {len(text)}, tags: {len(tags)} text length must equal to tag's"
                data["texts"].append(text)
                data["tags"].append(tags)
                data["raw_examples"].append(ex)
            LOGGER.info(
                f"Total have {cnt} blanks, 0.99 quantile: {np.quantile(length, 0.99):.2f}, 0.95 quantile: {np.quantile(length, 0.95):.2f}"
            )
            return data

    def encode(self, data: dict):
        texts = data["texts"]
        tags = data["tags"]
        raw_examples = data["raw_examples"]
        tag_set = sorted({t for tag in tags for t in tag})

        if not self.label_mapping:
            self.label_mapping = {tag: i for i, tag in enumerate(tag_set)}
            label_mapping_path = os.path.join(self.args.data_dir, "label_mapping.json")
            json_dump(label_mapping_path, self.label_mapping)

        encoded_data = []
        problem_data = []
        for raw_example, text, tag in tqdm(
            zip(raw_examples, texts, tags), desc="Encoding", total=len(texts)
        ):
            example = self.encode_step(raw_example, text, tag)
            if isinstance(example, InputExample):
                encoded_data.append(example)
            else:
                problem_data.append(raw_example)

        if problem_data:
            problem_data_path = os.path.join(self.args.data_dir, "problem_data.txt")
            with open(problem_data_path, "a+", encoding="utf8") as f:
                f.write("*" * 50 + "\n")
                f.write("\n\n".join(problem_data))
            LOGGER.info(f"Problem data have been write to {problem_data_path}.")

        return encoded_data

    def encode_step(self, raw_example, text, tag):
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
        merged_tag, cut_error = self.merge_tag(text, tag, offset_mapping)
        assert len(merged_tag) == len(
            inputs.input_ids
        ), "Label's length must be equal to input's length"

        try:
            tag_ids = [self.label_mapping[t] for t in merged_tag]
        except KeyError:
            LOGGER.info(traceback.format_exc())
            LOGGER.info("Maybe your debug ratio is to low, try to scale it up a little")
            sys.exit(1)

        return cut_error or InputExample(
            inputs.input_ids,
            inputs.token_type_ids,
            tag_ids,
            raw_example,
            offset_mapping,
        )

    def merge_tag(self, text, tag, offset_mapping):
        assert len(text) == len(tag), "text length must equal to tag's"
        merged_tag = []
        cut_error = False
        for offset in offset_mapping:
            span_tag = []
            start, end = offset[0], offset[-1]
            for i in range(start, end):
                if not span_tag or tag[i] != span_tag[-1]:
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
                LOGGER.info(f"Entity is cut error! error segment: {text[start: end]}")
                cut_error = True
                cur_tag = "O"

            if cur_tag.startswith("I") and merged_tag and merged_tag[-1] == "O":
                cur_tag == "O"
            merged_tag.append(cur_tag)

        merged_tag = ["O"] + merged_tag + ["O"]
        return merged_tag, cut_error

    def prepare(self):
        train_data_path = os.path.join(self.args.data_dir, self.args.train_file)
        test_data_path = os.path.join(self.args.data_dir, self.args.test_file)
        train_data = self.read_raw_data(
            train_data_path, num_examples=self.args.num_train_examples
        )
        test_data = self.read_raw_data(
            test_data_path, num_examples=self.args.num_test_examples
        )

        encoded_train_data = self.encode(train_data)
        self.test_cache = self.encode(test_data)

        np.random.shuffle(encoded_train_data)
        num_dev_examples = int(len(encoded_train_data) * self.args.dev_ratio)
        self.dev_cache = encoded_train_data[:num_dev_examples]
        self.train_cache = encoded_train_data[num_dev_examples:]

        self.to_pickle()
        LOGGER.info("Data have been cached")


class NERCollator(Collator):
    def __init__(
        self, max_seq_len: int, tokenizer: PreTrainedTokenizer, label_mapping: dict
    ):
        super().__init__(max_seq_len, tokenizer)
        self.label_mapping = label_mapping

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
        max_seq_len = min(max_seq_len, self.max_seq_len)
        # lengths = [min(self.max_seq_len - 2, len(input_id) - 2) for input_id in input_ids_list]
        input_ids, token_type_ids, attention_mask = self.pad_and_truncate(
            input_ids_list, token_type_ids_list, max_seq_len
        )

        labels = self.process_labels(labels_list, max_seq_len)
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
