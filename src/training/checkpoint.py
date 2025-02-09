import torch
from torch.optim import Optimizer
from src.nn.models.transformer import Transformer
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path
import os
from typing import Tuple
from accelerate import Accelerator
import shutil

def get_filename(name: str) -> str:
    return Path(name).stem

def get_backupname(name: str) -> str:
    return f"{name}.bak"

def get_sharded_dir(name: str) -> str:
    return f"{name}_sharded"

def save_checkpoint_non_sharded(
    model: Transformer,
    optimizer: Optimizer,
    writer: SummaryWriter,
    epoch: int,
    checkpoint_dir: str,
    checkpoint_name: str,
    accelerator: Accelerator
):
    container_dir = get_filename(checkpoint_name)
    checkpoint = os.path.join(checkpoint_dir, container_dir, checkpoint_name)
    checkpoint_backup = get_backupname(checkpoint_name)
    checkpoint_backup = os.path.join(checkpoint_dir, container_dir, checkpoint_backup)
    if accelerator.is_main_process:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(os.path.dirname(checkpoint)):
            os.makedirs(os.path.dirname(checkpoint))
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
            checkpoint
        )
    
def save_checkpoint_sharded(
    model: Transformer,
    optimizer: Optimizer,
    writer: SummaryWriter,
    epoch: int,
    checkpoint_dir: str,
    checkpoint_name: str,
    accelerator: Accelerator
):
    container_dir = get_filename(checkpoint_name)
    checkpoint = os.path.join(checkpoint_dir, container_dir, checkpoint_name)
    checkpoint_backup = get_backupname(checkpoint_name)
    checkpoint_backup = os.path.join(checkpoint_dir, container_dir, checkpoint_backup)
    if accelerator.is_main_process:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(os.path.dirname(checkpoint)):
            os.makedirs(os.path.dirname(checkpoint))
        if os.path.exists(checkpoint_backup):
            os.remove(checkpoint_backup)
        if os.path.exists(checkpoint):
            os.rename(checkpoint, checkpoint_backup)
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            {
                "writer": writer.get_logdir() if writer is not None else None,
                "epoch": epoch,
                "kwargs": getattr(unwrapped_model, "kwargs", None),
            },
            checkpoint
        )
    shard_path = get_sharded_dir(checkpoint)
    shard_path_backup = get_sharded_dir(checkpoint_backup)
    if accelerator.is_main_process:
        if os.path.exists(shard_path_backup):
            shutil.rmtree(shard_path_backup)
        if os.path.exists(shard_path):
            os.rename(shard_path, shard_path_backup)
    accelerator.wait_for_everyone()
    accelerator.save_state(shard_path)
    accelerator.wait_for_everyone()

def save_checkpoint(
    model: Transformer,
    optimizer: Optimizer,
    writer: SummaryWriter,
    epoch: int,
    checkpoint_dir: str,
    checkpoint_name: str,
    accelerator: Accelerator
):
    # https://github.com/huggingface/accelerate/issues/2000
    # i'll assume you are using fully sharded fsdp
    if accelerator.state.fsdp_plugin is not None:
        save_checkpoint_sharded(model, optimizer, writer, epoch, checkpoint_dir, checkpoint_name, accelerator)
    else:
        save_checkpoint_non_sharded(model, optimizer, writer, epoch, checkpoint_dir, checkpoint_name, accelerator)
    accelerator.wait_for_everyone()

def load_checkpoint_non_sharded(
        accelerator: Accelerator,
        checkpoint_path: str,
        checkpoint_name: str,
        optim_class=None,
        open_writer: bool = False,
) -> Tuple[int, Transformer, Optimizer | None, SummaryWriter | None]:
    container_dir = get_filename(checkpoint_name)
    checkpoint = os.path.join(checkpoint_path, container_dir, checkpoint_name)
    try:
        checkpoint = torch.load(checkpoint, map_location=accelerator.device if accelerator else None)
    except RuntimeError as e:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model = Transformer(**checkpoint["kwargs"])
    model.load_state_dict(checkpoint["model"])
    optimizer = optim_class(model.parameters()) if optim_class else None
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    writer = SummaryWriter(checkpoint["writer"]) if open_writer and accelerator.is_main_process else None

    model, optimizer = accelerator.prepare(model, optimizer)
    return epoch, model, optimizer, writer

def load_checkpoint_sharded(
    accelerator: Accelerator,
    checkpoint_path: str,
    checkpoint_name: str,
    optim_class=None,
    open_writer: bool = False,
) -> Tuple[int, Transformer, Optimizer | None, SummaryWriter | None]:
    container_dir = get_filename(checkpoint_name)
    checkpoint_path = os.path.join(checkpoint_path, container_dir, checkpoint_name)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=accelerator.device if accelerator else None)
    except RuntimeError as e:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = Transformer(**checkpoint["kwargs"])
    optimizer = optim_class(model.parameters()) if optim_class else None
    epoch = checkpoint["epoch"]
    writer = SummaryWriter(checkpoint["writer"]) if open_writer and accelerator.is_main_process else None

    model, optimizer = accelerator.prepare(model, optimizer)
    shard_path = get_sharded_dir(checkpoint_path)
    accelerator.load_state(shard_path)
    accelerator.wait_for_everyone()

    return epoch, model, optimizer, writer

def load_checkpoint(
    accelerator: Accelerator,
    checkpoint_path: str,
    checkpoint_name: str,
    optim_class=None,
    open_writer: bool = False,
) -> Tuple[int, Transformer, Optimizer | None, SummaryWriter | None]:
    # https://huggingface.co/docs/accelerate/usage_guides/fsdp#state-dict
    container_dir = get_filename(checkpoint_name)
    checkpoint = os.path.join(checkpoint_path, container_dir, checkpoint_name)
    shard_path = get_sharded_dir(checkpoint)

    if os.path.exists(shard_path):
        return load_checkpoint_sharded(accelerator, checkpoint_path, checkpoint_name, optim_class, open_writer)
    else:
        return load_checkpoint_non_sharded(accelerator, checkpoint_path, checkpoint_name, optim_class, open_writer)
