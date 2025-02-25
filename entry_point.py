from transformers import AutoTokenizer
from src.datasets.dataloader import DataLoader
# from src.datasets.shakespeare.dataset import Dataset as ShakespeareDataset
from src.datasets.wikipedia.dataset import Dataset as WikipediaDataset
from src.nn.models.transformer import Transformer
from src.training.train import train
from src.training.checkpoint import load_checkpoint
from torch.optim import AdamW as Adam
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
from src.tokenizer.model import Tokenizer
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = Tokenizer(tokenizer)

# Load dataset
dataset_path = "data/wikipedia"
max_seq_len = 96
train_ds = WikipediaDataset(f"{dataset_path}/train", max_seq_len)
test_ds = WikipediaDataset(f"{dataset_path}/test", max_seq_len)

# Load dataloader
batch_size = 24
min_ratio: int = 3
max_ratio: int = 3
max_num_spans: int = 1
max_span_fill: float = 0.15
min_num_spans: int = 1
min_span_fill: float = 0.15
hard_fill = True

train_dl = DataLoader(
    ds=train_ds,
    batch_size=batch_size,
    enc_span_idx=tokenizer.enc_span_token,
    target_span_idx=tokenizer.targ_span_token,
    fill_idx=tokenizer.mask_token,
    eos_idx=tokenizer.eos_token,
    bos_idx=tokenizer.bos_token,
    min_ratio=min_ratio,
    max_ratio=max_ratio,
    max_num_spans=max_num_spans,
    max_span_fill=max_span_fill,
    min_num_spans=min_num_spans,
    min_span_fill=min_span_fill,
    hard_fill=hard_fill,
)

test_dl = DataLoader(
    ds=test_ds,
    batch_size=batch_size,
    enc_span_idx=tokenizer.enc_span_token,
    target_span_idx=tokenizer.targ_span_token,
    fill_idx=tokenizer.mask_token,
    eos_idx=tokenizer.eos_token,
    bos_idx=tokenizer.bos_token,
    min_ratio=min_ratio,
    max_ratio=max_ratio,
    max_num_spans=max_num_spans,
    max_span_fill=max_span_fill,
    min_num_spans=min_num_spans,
    min_span_fill=min_span_fill,
    hard_fill=hard_fill,
)

# Make or Load model

emb_dim = 512
vocab_size = tokenizer.vocab_size
max_enc_len = max_seq_len + 2  # because we prepend and append bos and eos tokens
max_dec_len = (max_seq_len + 2) * max_ratio
n_heads = 8
n_enc = 3
n_dec = 3
mlp_dim = emb_dim * 3
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

checkpoint_path = "checkpoints"
checkpoint_name = "wikipedia_one_span.pth"
writer_path = "runs/wikipedia_one_span"

try:
    epoch, model, optimizer, writer = load_checkpoint(
        checkpoint_path=f"{checkpoint_path}/{checkpoint_name}",
        optim_class=Adam,
        open_writer=True,
        device=device,
    )
except FileNotFoundError:
    model = Transformer(
        emb_dim=emb_dim,
        vocab_size=vocab_size,
        max_enc_len=max_enc_len,
        max_dec_len=max_dec_len,
        n_heads=n_heads,
        n_enc=n_enc,
        n_dec=n_dec,
        mlp_dim=mlp_dim,
        dropout=dropout,
    )
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter(writer_path)
    epoch = 0

print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")

target_epochs = 1_560_000
save_every = 5000
test_every = 32
grad_clip_norm = None

mask_percent = 0.9
use_glancing = True


train(
    model=model,
    tk=tokenizer,
    train_dl=train_dl,
    test_dl=test_dl,
    optimizer=optimizer,
    writer=writer,
    device=device,
    mask_percent=mask_percent,
    save_path=checkpoint_path,
    save_name=checkpoint_name,
    start_epoch=epoch,
    end_epoch=target_epochs,
    save_every=save_every,
    test_every=test_every,
    use_glancing=use_glancing,
    grad_clip_norm=grad_clip_norm,
)
