import torch
from torch import Tensor
from torch.utils.data import DataLoader
from src.nn.models.transformer import Transformer
from src.training.loss_fn import dag_loss
from src.training.glancing import findPathBatched, backtrace
from src.common.npercent_mask import batchwise_npercent_mask
from typing import Iterator
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from src.training.checkpoint import save_checkpoint
from tqdm.auto import tqdm
from typing import Tuple
from src.tokenizer.model import Tokenizer
import os
from accelerate import Accelerator



def test_step(
    model: Transformer,
    dataloader_iter: Iterator[
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]
    ],
    writer: SummaryWriter,
    epoch: int,
    device: str,
) -> bool:
    try:
        with torch.no_grad():
            model.eval()
            (
                batch,
                enc,
                targ,
                dec_pos,
                dec_v,
                target_lens,
                vertex_lens,
                target_span_indices,
                ratio,
            ) = next(dataloader_iter)

            transition_probs, emission_probs = model(
                enc_x=enc, dec_x_vocab=dec_v, dec_x_pos=dec_pos, vertex_lens=vertex_lens
            )
            loss = dag_loss(
                targets=targ,
                transition_matrix=transition_probs,
                emission_probs=emission_probs,
                target_lens=target_lens,
                vertex_lens=vertex_lens,
            )
            if loss.isnan():
                raise ValueError("Loss is NaN")
            if writer is not None:
                writer.add_scalar("Loss/Test", loss.item(), epoch)
            model.train()
            return False
    except StopIteration:
        return True


def glancing_step(
    model: Transformer,
    data: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int],
    mask_percent: float,
    device: str,
    fill_idx: int,
) -> Tensor:
    with torch.no_grad():
        model.train()
        (
            batch,
            enc,
            targ,
            dec_pos,
            dec_v,
            target_lens,
            vertex_lens,
            target_span_indices,
            ratio,
        ) = data


        transition_probs, emission_probs = model(
            enc_x=enc, dec_x_vocab=dec_v, dec_x_pos=dec_pos, vertex_lens=vertex_lens
        )
        _, back = findPathBatched(
            transition_matrix=transition_probs,
            emission_probs=emission_probs,
            target_seq=targ,
            target_span_indices=target_span_indices,
        )
        trace, _ = backtrace(
            backtrace=back, target_lens=target_lens, vertex_lens=vertex_lens
        )
        mask = batchwise_npercent_mask(original=trace, percent=mask_percent)
        assignments = torch.where(mask, -1, trace)
        # assignments = assignments.to(device)
        # mask = mask.to(device)
        # trace = trace.to(device)
        scattered_vocab = dec_v.scatter(1, trace, torch.where(mask, -1, targ))
        scattered_vocab[scattered_vocab == -1] = fill_idx
        #return assignments.to(device), scattered_vocab.to(device)
        return assignments, scattered_vocab


def train_step(
    accelerator: Accelerator,
    model: Transformer,
    dataloader_iter: Iterator[
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]
    ],
    optimizer: Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: str,
    fill_idx: int,
    mask_percent: float,
    pbar: tqdm | None = None,
    use_glancing: bool = False,
    grad_clip_norm: float = 0.1,
) -> bool:
    try:
        model.train()
        (
            batch,
            enc,
            targ,
            dec_pos,
            dec_v,
            target_lens,
            vertex_lens,
            target_span_indices,
            ratio,
        ) = next(dataloader_iter)

        assignments, dec_v = (
            (None, dec_v)
            if not use_glancing
            else glancing_step(
                model=model,
                data=(
                    batch,
                    enc,
                    targ,
                    dec_pos,
                    dec_v,
                    target_lens,
                    vertex_lens,
                    target_span_indices,
                    ratio,
                ),
                mask_percent=mask_percent,
                device=device,
                fill_idx=fill_idx,
            )
        )

        optimizer.zero_grad()
        transition_probs, emission_probs = model(
            enc_x=enc, dec_x_vocab=dec_v, dec_x_pos=dec_pos, vertex_lens=vertex_lens
        )
        loss = dag_loss(
            targets=targ,
            transition_matrix=transition_probs,
            emission_probs=emission_probs,
            target_lens=target_lens,
            vertex_lens=vertex_lens,
            assignments=assignments,
        )
        if loss.isnan():
            raise ValueError("Loss is NaN")
        if loss.isinf():
            # print("Loss is inf, saving data for debugging...")
            # torch.save(
            #     {
            #         "enc": enc,
            #         "targ": targ,
            #         "dec_pos": dec_pos,
            #         "dec_v": dec_v,
            #         "target_lens": target_lens,
            #         "vertex_lens": vertex_lens,
            #         "target_span_indices": target_span_indices,
            #         "ratio": ratio,
            #         "assignments": assignments,
            #     },
            #     "inf_data.pt",
            # )
            raise ValueError("Loss is inf")
        #loss.backward()
        accelerator.backward(loss)
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if writer is not None:
            writer.add_scalar("Loss/Training", loss.item(), epoch)
        if pbar is not None:
            pbar.set_postfix({"Loss": loss.item()})
        return False
    except StopIteration:
        return True


def train(
    accelerator: Accelerator,
    model: Transformer,
    tk: Tokenizer,
    train_dl: DataLoader,
    test_dl: DataLoader,
    optimizer: Optimizer,
    writer: SummaryWriter,
    device: str,
    mask_percent: float,
    save_path: str,
    save_name: str,
    start_epoch: int = 0,
    end_epoch: int = 100,
    save_every: int = 256,
    test_every: int = 32,
    use_glancing: bool = False,
    grad_clip_norm: float = 0.1,
):
    epoch = start_epoch
    exit_saved = False
    try:
        # ensure save_path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_iter = iter(train_dl)
        test_iter = iter(test_dl)
        pbar = tqdm(range(start_epoch, end_epoch))
        for epoch in pbar:

            train_iter_fail = train_step(
                accelerator=accelerator,
                model=model,
                dataloader_iter=train_iter,
                optimizer=optimizer,
                writer=writer,
                epoch=epoch,
                device=device,
                fill_idx=tk.mask_token,
                pbar=pbar,
                mask_percent=mask_percent,
                use_glancing=use_glancing,
                grad_clip_norm=grad_clip_norm,
            )
            if train_iter_fail:
                # this happens when the dataloader is exhausted
                train_iter = iter(train_dl)
                train_step(
                    accelerator=accelerator,
                    model=model,
                    dataloader_iter=train_iter,
                    optimizer=optimizer,
                    writer=writer,
                    epoch=epoch,
                    device=device,
                    fill_idx=tk.mask_token,
                    pbar=pbar,
                    mask_percent=mask_percent,
                    use_glancing=use_glancing,
                    grad_clip_norm=grad_clip_norm,
                )
            if epoch % test_every == 0:
                test_iter_fail = test_step(
                    model=model,
                    dataloader_iter=test_iter,
                    writer=writer,
                    epoch=epoch,
                    device=device,
                )
                if test_iter_fail:
                    test_iter = iter(test_dl)
                    test_step(
                        model=model,
                        dataloader_iter=test_iter,
                        writer=writer,
                        epoch=epoch,
                        device=device,
                    )
            if epoch % save_every == 0:
                save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    writer=writer,
                    epoch=epoch,
                    checkpoint_dir=save_path,
                    checkpoint_name=save_name,
                )
    except KeyboardInterrupt:
        print("Training interrupted, saving model. With Accelerator you may encounter an error, but the checkpoint should be fine")
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            writer=writer,
            epoch=epoch,
            checkpoint_dir=save_path,
            checkpoint_name=save_name,
        )
        exit_saved = True
    if not exit_saved:
        print("Training complete, saving model...")
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            writer=writer,
            epoch=epoch,
            checkpoint_dir=save_path,
            checkpoint_name=save_name,
        )
