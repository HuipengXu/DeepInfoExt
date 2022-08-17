from argparse import ArgumentParser


def get_default_parser():
    parser = ArgumentParser(description="For training and evaluation")
    parser.add_argument("--debug", default=0, help="whether to debug", type=int)
    parser.add_argument("--do_train", default=1, help="train or test", type=int)
    parser.add_argument("--task_name", default="NER", help="task name", type=str)
    parser.add_argument("--train_file", default="", help="train file name", type=str)
    parser.add_argument("--test_file", default="", help="train file name", type=str)
    parser.add_argument(
        "--num_train_examples", default=0, help="numbers of training examples", type=int
    )
    parser.add_argument(
        "--num_test_examples", default=0, help="numbers of test examples", type=int
    )
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
        "--debug_ratio",
        default=0.01,
        type=float,
        help="debug data ratio in total data",
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
        "--max_grad_norm", default=5.0, type=float, help="max_grad_norm for clip"
    )
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
    return parser
