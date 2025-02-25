from abc import ABC, abstractmethod
import random
import torch
from datasets import load_from_disk


class DatasetBase(ABC):
    def __init__(self, dataset_path: str, max_seq_len: int):
        self.ds = load_from_disk(dataset_path)
        self.max_seq_len = max_seq_len

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_batch(self, batch_size: int):
        pass
