import torch
from torch.optim import Optimizer
from src.nn.models.transformer import Transformer
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path
import os
from typing import Tuple
from accelerate import Accelerator


def save_checkpoint(
    model: Transformer,
    optimizer: Optimizer,
    writer: SummaryWriter,
    epoch: int,
    checkpoint_dir: str,
    checkpoint_name: str,
    accelerator: Accelerator
):
    if accelerator.is_main_process:
        checkpoint_backup_name = f"{checkpoint_name}.bak"
        checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
        checkpoint_backup = os.path.join(checkpoint_dir, checkpoint_backup_name)
        if os.path.exists(checkpoint_backup):
            os.remove(checkpoint_backup)
        if os.path.exists(checkpoint):
            os.rename(checkpoint, checkpoint_backup)
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            {
                "model": unwrapped_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "writer": writer.get_logdir() if writer is not None else None,
                "epoch": epoch,
                "kwargs": getattr(unwrapped_model, "kwargs", None),
            },
            checkpoint,
        )


def load_checkpoint(
    accelerator: Accelerator,
    checkpoint_path: str,
    optim_class=None,
    open_writer: bool = False,
) -> Tuple[int, Transformer, Optimizer | None, SummaryWriter | None]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=accelerator.device if accelerator else None)
    except RuntimeError as e:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = Transformer(**checkpoint["kwargs"])
    model.load_state_dict(checkpoint["model"])
    optimizer = optim_class(model.parameters()) if optim_class else None
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    writer = SummaryWriter(checkpoint["writer"]) if open_writer and accelerator.is_main_process else None
    return epoch, model, optimizer, writer


