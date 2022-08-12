import os
import json
from tqdm import tqdm
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForTokenClassification

from .model import BertWithCRF
from .utils import json_load, get_seqeuence_labeling_metrics, LOGGER


class Predictor:
    def __init__(
        self, args: Namespace, test_dataloader: DataLoader, label_mapping: dict
    ) -> None:
        self.args = args
        self.test_raw_path = os.path.join(args.data_dir, "msra_test_bio.txt")
        self.test_dataloader = test_dataloader
        self.id2label = {id_: label for label, id_ in label_mapping.items()}
        config = BertConfig.from_pretrained(
            args.model_path, num_labels=len(label_mapping)
        )
        self.model = BertForTokenClassification.from_pretrained(args.save_model_path, config=config)
        self.model.to(args.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self):
        predictions = []
        labels = []
        test_loss = 0.0
        test_iterator = tqdm(
            self.test_dataloader, desc="Testing", total=len(self.test_dataloader)
        )

        for batch in test_iterator:
            batch_labels = [
                [self.id2label[label] for label in label_seq[1:-1]]
                for label_seq in batch["labels"].numpy()
            ]
            labels.extend(batch_labels)
            batch_cuda = {
                item: value.to(self.args.device) for item, value in list(batch.items())
            }
            loss, logits = self.model(**batch_cuda)[:2]
            probs = torch.softmax(logits, dim=-1)

            if self.args.ngpus > 1:
                loss = loss.mean()

            test_loss += loss.item()
            batch_predictions = [
                [self.id2label[pred] for pred in pred_seq[1:-1]]
                for pred_seq in probs.argmax(dim=-1).cpu().numpy()
            ]
            predictions.extend(batch_predictions)

        avg_test_loss = test_loss / len(self.test_dataloader)
        p, r, f1, acc = get_seqeuence_labeling_metrics(labels, predictions)
        metrics = {"p": p, "r": r, "f1": f1, "acc": acc, "avg_test_loss": avg_test_loss}

        self.save_bad_cases(predictions, labels)
        LOGGER.info(f"\n{json.dumps(metrics, indent=2,  ensure_ascii=False)}")

        return metrics

    def save_bad_cases(self, predictions, labels):
        with open(self.test_raw_path, "r", encoding="utf8") as f:
            examples = f.read().split("\n\n")

        bad_cases = []
        good_cases = []
        for i, (label, prediction) in enumerate(zip(labels, predictions)):
            example = examples[i]
            tokens = example.split("\n")
            # unfold tag
            new_prediction = ["O"] * len(tokens)
            offsets = self.test_dataloader.dataset.data[i].offsets
            offsets = offsets[:len(prediction)]
            for j, offset in enumerate(offsets):
                cur_tag = prediction[j]
                if cur_tag.startswith("I"):
                    new_prediction[offset[0] : offset[-1]] = [cur_tag] * (
                        offset[-1] - offset[0]
                    )
                elif cur_tag.startswith("B"):
                    new_prediction[offset[0]] = cur_tag
                    entity = cur_tag.split("-")[-1]
                    new_prediction[offset[0] + 1 : offset[-1]] = [f"I-{entity}"] * (
                        offset[-1] - offset[0] - 1
                    )

            tokens_with_pred = [
                f"{token.strip()} {new_prediction[k]}" for k, token in enumerate(tokens)
            ]
            
            example_with_pred = "\n".join(tokens_with_pred)
            if label != prediction:
                bad_cases.append(example_with_pred)
            else:
                good_cases.append(example_with_pred)

        bad_cases_path = os.path.join(self.args.data_dir, "bad_cases.txt")
        good_cases_path = os.path.join(self.args.data_dir, "good_cases.txt")
        with open(bad_cases_path, "w", encoding="utf8") as f:
            bad_cases = "\n".join(bad_cases)
            f.write(bad_cases)
                                          
        with open(good_cases_path, "w", encoding="utf8") as f:
            good_cases = "\n".join(good_cases)
            f.write(good_cases)
