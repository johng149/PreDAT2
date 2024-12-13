import torch
from src.common.span_masking import flow, mask_span
import random
from src.common.target_span_indices import targetSpanIndices


def collate_fn(
    batch,
    enc_span_idx: int,
    target_span_idx: int,
    fill_idx: int,
    eos_idx: int,
    bos_idx: int,
    min_ratio: int = 2,
    max_ratio: int = 4,
    max_num_spans: int = 6,
    max_span_fill: float = 0.8,
    min_num_spans: int = 0,
    min_span_fill: float = 0,
    hard_fill=True,
):
    """
    Given a batch of tensors, this function will apply the flow function to each tensor
    to produce the encoder and target masked tensor pairs for each tensor in the batch.

    The target tensor will also be accompanied by a decoder input tensor which has
    length dependent on a ratio of the length of the target tensor. Once a ratio
    has been determined, all batches will use that ratio.

    There are two decoder input tensors: one for positions and one for vocabulary.
    The position tensor will be a tensor of shape (seq_len,) where each element is
    in ascending order from 0 to seq_len-1. The vocabulary tensor will be a tensor
    of shape (seq_len,) filled with the `fill_idx` value. I recommend that the
    `fill_idx` value be different from the `enc_span_idx` and `target_span_idx` values,
    such as the masking / padding / unk token index of whatever tokenizer you are using.

    The batch should be a list of tensors or a 2D tensor where each row is a tensor.

    Args:
        batch (List[Tensor] | Tensor): The batch of tensors to mask
        enc_span_idx (int): The index to represent encoder masked spans
        target_span_idx (int): The index to represent target masked spans
        fill_idx (int): The index to fill the decoder input tensor for vocab
        mask_percent (float): The percentage of elements to mask
        min_ratio (int): The minimum ratio to determine the length of the decoder input tensor
        max_ratio (int): The maximum ratio to determine the length of the decoder input tensor
        max_num_spans (int): The maximum number of spans to mask
        max_span_fill (float): The maximum proportion of the span that should be masked
        min_num_spans (int): The minimum number of spans to mask
        min_span_fill (float): The minimum proportion of the span that should be masked
        hard_fill (bool): Whether to ensure that no more than `max_span_fill` of the span is masked
            If False, the percentage of masked tokens may be higher than `max_span_fill`

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, int]: A tuple of the encoder tensor, target tensor,
        decoder input tensor for positions, the decoder input tensor for vocab,
        and the ratio used to determine the length of the decoder input tensor
    """
    ratio = random.randint(min_ratio, max_ratio)
    encs, targs, decs_pos, decs_vocab, target_span_indices = [], [], [], [], []
    mask = mask_span(
        tokens=batch[0],
        max_num_spans=max_num_spans,
        max_span_fill=max_span_fill,
        min_num_spans=min_num_spans,
        min_span_fill=min_span_fill,
        hard_fill=hard_fill,
    )
    eos_postfix = torch.tensor([eos_idx])
    bos_prefix = torch.tensor([bos_idx])
    for tensor in batch:
        enc, targ = flow(tensor, mask, enc_span_idx, target_span_idx)

        enc = torch.cat((bos_prefix, enc, eos_postfix))
        targ = torch.cat((bos_prefix, targ, eos_postfix))

        span_indices = targetSpanIndices(
            targets=targ,
            is_span=(targ == target_span_idx),
            ratio=ratio,
        )

        dec_len = len(targ) * ratio
        dec_pos = torch.arange(dec_len)
        dec_vocab = torch.full((dec_len,), fill_idx)

        encs.append(enc)
        targs.append(targ)
        decs_pos.append(dec_pos)
        decs_vocab.append(dec_vocab)
        target_span_indices.append(span_indices)

    encs = torch.stack(encs)
    targs = torch.stack(targs)
    decs_pos = torch.stack(decs_pos)
    decs_vocab = torch.stack(decs_vocab)
    target_span_indices = torch.stack(target_span_indices)
    return encs, targs, decs_pos, decs_vocab, target_span_indices, ratio
