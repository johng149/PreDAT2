from transformers import PreTrainedTokenizerBase


class Tokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.ensure_mask_token()
        self.ensure_bos_token()
        self.ensure_eos_token()
        self.ensure_enc_span_token()
        self.ensure_targ_span_token()

    def ensure_mask_token(self):
        if self.tokenizer.mask_token is None:
            self.mask_token = self.vocab_size
            self.vocab_size += 1
        else:
            self.mask_token = self.tokenizer.mask_token_id

    def ensure_bos_token(self):
        if self.tokenizer.bos_token is None:
            self.bos_token = self.vocab_size
            self.vocab_size += 1
        else:
            self.bos_token = self.tokenizer.bos_token_id

    def ensure_eos_token(self):
        if self.tokenizer.eos_token is None:
            self.eos_token = self.vocab_size
            self.vocab_size += 1
        else:
            self.eos_token = self.tokenizer.eos_token_id

    def ensure_enc_span_token(self):
        self.enc_span_token = self.vocab_size
        self.vocab_size += 1

    def ensure_targ_span_token(self):
        self.targ_span_token = self.vocab_size
        self.vocab_size += 1
