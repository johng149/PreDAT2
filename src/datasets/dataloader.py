import torch
from src.datasets.dataset_base import DatasetBase
from src.datasets.collate_fn import collate_fn


class DataLoader:

    def __init__(
        self,
        ds: DatasetBase,
        batch_size: int,
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
        assert batch_size > 0
        self.ds = ds
        self.batch_size = batch_size
        self.enc_span_idx = enc_span_idx
        self.target_span_idx = target_span_idx
        self.fill_idx = fill_idx
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.max_num_spans = max_num_spans
        self.max_span_fill = max_span_fill
        self.min_num_spans = min_num_spans
        self.min_span_fill = min_span_fill
        self.hard_fill = hard_fill
        self._current_idx = 0
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx

    def get_batch(self):

        batch = self.ds.get_batch(self.batch_size)
        enc, targ, dec_pos, dec_v, target_span_indices, ratio = collate_fn(
            batch=batch,
            enc_span_idx=self.enc_span_idx,
            target_span_idx=self.target_span_idx,
            fill_idx=self.fill_idx,
            eos_idx=self.eos_idx,
            bos_idx=self.bos_idx,
            min_ratio=self.min_ratio,
            max_ratio=self.max_ratio,
            max_num_spans=self.max_num_spans,
            max_span_fill=self.max_span_fill,
            min_num_spans=self.min_num_spans,
            min_span_fill=self.min_span_fill,
            hard_fill=self.hard_fill,
        )
        # target_lens has shape (batch_size,)
        # each element is the length of its respective target sequence
        # however, since we are using bucketing, it should all be the same
        batch_size, targ_len = targ.shape
        target_lens = torch.full((batch_size,), targ_len)
        # something similar for vertex_lens, but it is for the decoder
        _, dec_len = dec_pos.shape
        vertex_lens = torch.full((batch_size,), dec_len)
        return (
            batch,
            enc,
            targ,
            dec_pos,
            dec_v,
            target_lens,
            vertex_lens,
            target_span_indices,
            ratio,
        )

    def __len__(self):
        return len(self.ds) // self.batch_size

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= len(self):
            self._current_idx = 0
            raise StopIteration
        self._current_idx += 1
        return self.get_batch()
