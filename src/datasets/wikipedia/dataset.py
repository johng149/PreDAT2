from transformers import PreTrainedTokenizerBase
from datasets import load_dataset
from pathlib import Path
import json
import random
import torch
from src.datasets.dataset_base import DatasetBase


def process(
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_name: str,
    output_dir: str,
):
    def tokenize_sample(sample):
        text = sample["text"]
        encoded = tokenizer.encode(text, return_tensors="pt").squeeze(0)
        return {"input_ids": encoded}

    wiki_en = load_dataset("wikipedia", "20220301.en", split="train")
    wiki_split = wiki_en.train_test_split(test_size=0.1)
    wiki_en_train = wiki_split["train"]
    wiki_en_validation = wiki_split["test"]

    save_location = Path(output_dir)
    save_meta_location = save_location / "meta.json"

    wiki_en_train = wiki_en_train.map(tokenize_sample)
    wiki_en_validation = wiki_en_validation.map(tokenize_sample)
    
    wiki_en_train = wiki_en_train.remove_columns(["id", "title", "text", "url"])
    wiki_en_validation = wiki_en_validation.remove_columns(["id", "title", "text", "url"])
    
    wiki_en_train.set_format("torch", columns=["input_ids"])
    wiki_en_validation.set_format("torch", columns=["input_ids"])
    
    wiki_en_train.save_to_disk(save_location / "train")
    wiki_en_validation.save_to_disk(save_location / "test")
    with open(save_meta_location, "w") as f:
        json.dump(
            {
                "tokenizer_name": tokenizer_name
            },
            f
        )

class Dataset(DatasetBase):
    def __init__(self, dataset_path: str, max_seq_len: int):
        super().__init__(dataset_path, max_seq_len)

    def __len__(self):
        return len(self.ds)
    
    def get_batch(self, batch_size: int):
        batch = []
        batch_min_len = float("inf")

        for i in range(batch_size):
            index = random.randint(0, len(self.ds) - 1)
            sample = self.ds.select([index])["input_ids"].flatten()
            batch.append(sample)
            batch_min_len = min(batch_min_len, len(sample))

        max_len = min(self.max_seq_len, batch_min_len)
        sliced_batch = []
        for sample in batch:
            start_idx = random.randint(0, len(sample) - max_len)
            sliced_batch.append(sample[start_idx : start_idx + max_len])
        
        return torch.stack(sliced_batch)