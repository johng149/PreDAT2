from src.datasets.wikipedia.dataset import process, Dataset
from transformers import AutoTokenizer
from pathlib import Path

def setup(location):
    dataset_path = location
    train_path = Path(dataset_path) / "train"
    if not train_path.exists():
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        process(tokenizer, "gpt2", dataset_path)

if __name__ == "__main__":
    setup()