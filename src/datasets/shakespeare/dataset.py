from transformers import PreTrainedTokenizerBase
from datasets import load_dataset, load_from_disk
from pathlib import Path
import json
import random
import torch


def process(
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_name: str,
    output_dir: str,
):
    def tokenize_sample(sample):
        text = sample["text"]
        tokenized = tokenizer.encode(text, return_tensors="pt").squeeze()
        return {"input_ids": tokenized}

    ds = load_dataset("karpathy/tiny_shakespeare")

    save_location = Path(output_dir)
    save_meta_location = save_location / "meta.json"

    ds = ds.map(tokenize_sample)
    ds = ds.remove_columns("text")
    ds.set_format("torch", columns=["input_ids"])
    ds.save_to_disk(save_location)
    with open(save_meta_location, "w") as f:
        json.dump({"tokenizer_name": tokenizer_name}, f)


class Dataset:
    def __init__(self, dataset_path: str, max_seq_len: int):
        self.ds = load_from_disk(dataset_path)
        self.max_seq_len = min(max_seq_len, len(self.ds["input_ids"][0]))

    def __len__(self):
        return len(self.ds["input_ids"][0])

    def get_batch(self, batch_size: int):
        batch_seq_len = random.randint(1, self.max_seq_len)
        batch = []

        for i in range(batch_size):
            start_idx = random.randint(0, len(self.ds["input_ids"][0]) - batch_seq_len)
            batch.append(self.ds["input_ids"][0][start_idx : start_idx + batch_seq_len])

        return torch.stack(batch)
