from contextlib import contextmanager
import os
import math
import json
import random
import logging
import platform
import pkg_resources as pkg

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch
import evaluate
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def json_load(path: str):
    with open(path, "r", encoding="utf8") as f:
        result = json.load(f)
    return result


def json_dump(path: str, obj: dict):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f)


class LabelSmoothingLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.01):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = torch.log_softmax(x, dim=-1)

        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.5, emb_name="embeddings"):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="embeddings"):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(
        self, epsilon=1.0, alpha=0.3, emb_name="word_embeddings", is_first_attack=False
    ):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name="word_embeddings"):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class EMA:
    def __init__(self, model, decay, tau=2000, global_steps=0):
        self.model = model
        self.decay = lambda steps: decay * (1 - math.exp(-steps / tau))
        self.global_steps = global_steps
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self.global_steps += 1
        decay = self.decay(self.global_steps)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def get_classification_metrics(y_true, y_pred):
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return p, r, f1, acc


seqeval_evaluator = evaluate.load("seqeval")


def get_seqeuence_labeling_metrics(y_true, y_pred):
    results = seqeval_evaluator.compute(
        predictions=y_pred, references=y_true, scheme="IOB2"
    )
    return (
        results["overall_precision"],
        results["overall_recall"],
        results["overall_f1"],
        results["overall_accuracy"],
    )


def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)


set_logging()  # run before defining LOGGER
LOGGER = logging.getLogger(
    "msra-ner"
)  # define globally (used in train.py, val.py, detect.py, etc.)
# for fn in LOGGER.info, LOGGER.warning:
#     _fn, fn = fn, lambda x: _fn(emojis(x))  # emoji safe logging


def emojis(str=""):
    # Return platform-dependent emoji-safe version of string
    return (
        str.encode().decode("ascii", "ignore")
        if platform.system() == "Windows"
        else str
    )


def select_device(device="", batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f"MSRA-NER 🚀   Python-{platform.python_version()} torch-{torch.__version__} \n"
    device = (
        str(device).strip().lower().replace("cuda:", "").replace("none", "")
    )  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not (cpu or mps) and torch.cuda.is_available():  # prefer GPU if available
        devices = (
            device.split(",") if device else "0"
        )  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert (
                batch_size % n == 0
            ), f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    elif (
        mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available()
    ):  # prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)


def check_version(
    current="0.0.0",
    minimum="0.0.0",
    name="version ",
    pinned=False,
    hard=False,
    verbose=False,
):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f"{name}{minimum} required by YOLOv5, but {name}{current} is currently installed"  # string
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result


def seed_everything(seed, deterministic=False):
    import torch.backends.cudnn as cudnn

    if deterministic and check_version(
        torch.__version__, "1.12.0"
    ):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def smart_DDP(model, local_rank):
    # Model DDP creation with checks
    if check_version(torch.__version__, '1.11.0'):
        return DDP(model, device_ids=[local_rank], output_device=local_rank, static_graph=True)
    else:
        return DDP(model, device_ids=[local_rank], output_device=local_rank)