from transformers import PreTrainedTokenizerBase
from datasets import load_dataset, load_from_disk
from pathlib import Path
import json
import random
import torch
#from src.datasets.dataset_base import DatasetBase
from torch.utils.data import Dataset as DatasetBase

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


class Dataset(DatasetBase):
    def __init__(self, dataset_path: str, max_seq_len: int):
        self.ds = load_from_disk(dataset_path)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.ds["input_ids"][0])

    def __getitem__(self, idx):
        return self.ds["input_ids"][0][idx: idx + self.max_seq_len]
