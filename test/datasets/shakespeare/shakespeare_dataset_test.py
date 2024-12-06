import pytest
from src.datasets.shakespeare.dataset import process, Dataset
from transformers import AutoTokenizer
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def setup():
    dataset_path = "data/shakespeare"
    train_path = Path(dataset_path) / "train"
    if not train_path.exists():
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        process(tokenizer, "gpt2", dataset_path)


def test_dataset():
    ds = Dataset("data/shakespeare/train", 100)
    assert ds.max_seq_len == 100

    batch = ds.get_batch(2)
    batch_size, seq_len = batch.shape
    assert batch_size == 2
    assert seq_len <= 100
