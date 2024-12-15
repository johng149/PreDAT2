import torch
from torch.optim import Optimizer
from src.nn.models.transformer import Transformer
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path
import os
from typing import Tuple


def save_checkpoint(
    model: Transformer,
    optimizer: Optimizer,
    writer: SummaryWriter,
    epoch: int,
    checkpoint_dir: str,
    checkpoint_name: str,
):
    checkpoint_backup_name = f"{checkpoint_name}.bak"
    checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint_backup = os.path.join(checkpoint_dir, checkpoint_backup_name)
    if os.path.exists(checkpoint_backup):
        os.remove(checkpoint_backup)
    if os.path.exists(checkpoint):
        os.rename(checkpoint, checkpoint_backup)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "writer": writer.get_logdir(),
            "epoch": epoch,
            "kwargs": model.kwargs,
        },
        checkpoint,
    )


def load_checkpoint(
    checkpoint_path: str,
    optim_class=None,
    open_writer: bool = False,
    device: str = "cpu",
) -> Tuple[int, Transformer, Optimizer | None, SummaryWriter | None]:
    checkpoint = torch.load(checkpoint_path)
    model = Transformer(**checkpoint["kwargs"])
    model.to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = optim_class(model.parameters()) if optim_class else None
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    writer = SummaryWriter(checkpoint["writer"]) if open_writer else None
    return epoch, model, optimizer, writer
