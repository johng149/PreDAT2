from transformers import PreTrainedTokenizerBase
from datasets import load_dataset, load_from_disk
from pathlib import Path
import json
import random
import torch
from torch.utils.data import Dataset as DatasetBase

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
        self.ds = load_from_disk(dataset_path)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        sample = self.ds.select([idx])["input_ids"].flatten()
        length = len(sample)
        farthest_start_idx = max(0, length - self.max_seq_len)
        start_idx = random.randint(0, farthest_start_idx)
        sample = sample[start_idx : start_idx + self.max_seq_len]
        
        return sample