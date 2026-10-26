import torch

from torchtitan.components.loss import IGNORE_INDEX


class TextPretrainCollator:
    """Collate token samples into the Titan pretraining batch format

    Samples shorter than seq_len (e.g. the tail of a shard) are padded with
    eos to the fixed seq_len; the padded label positions are masked with
    IGNORE_INDEX by real length
    """

    def __init__(self, *, seq_len: int, eos_token_id: int) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if eos_token_id is None or eos_token_id < 0:
            raise ValueError(
                f"eos_token_id must be a non-negative int, got {eos_token_id}"
            )
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id

    def __call__(
        self,
        samples: list[dict[str, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if not samples:
            raise ValueError("TextPretrainCollator received no samples")

        rows = []
        real_lens = []
        for sample_idx, sample in enumerate(samples):
            if "tokens" not in sample:
                raise ValueError(f"Sample {sample_idx} is missing 'tokens'")
            tokens = sample["tokens"]
            if not isinstance(tokens, torch.Tensor):
                raise ValueError(
                    f"Sample {sample_idx} tokens must be a torch.Tensor, "
                    f"got {type(tokens).__name__}"
                )
            if tokens.ndim != 1:
                raise ValueError(
                    f"Sample {sample_idx} tokens must be 1D, got shape {tuple(tokens.shape)}"
                )
            row = tokens[: self.seq_len].to(dtype=torch.long)
            real_len = row.numel()
            if real_len < self.seq_len:
                row = torch.cat(
                    [row, row.new_full((self.seq_len - real_len,), self.eos_token_id)]
                )
            rows.append(row)
            real_lens.append(real_len)

        input_ids = torch.stack(rows, dim=0)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[:, :-1] = input_ids[:, 1:]
        # mask padded labels by real length, not by token value
        for row_idx, real_len in enumerate(real_lens):
            if real_len < self.seq_len:
                labels[row_idx, max(real_len - 1, 0) :] = IGNORE_INDEX

        # calculations for mask_type == 'block_causal' support
        col = torch.arange(self.seq_len, dtype=torch.int32).expand_as(input_ids)
        doc_start = torch.zeros_like(input_ids, dtype=torch.bool)
        doc_start[:, 0] = True
        doc_start[:, 1:] = input_ids[:, :-1] == self.eos_token_id
        last_start = torch.cummax(
            torch.where(doc_start, col, torch.zeros_like(col)), dim=1
        ).values
        positions = (col - last_start).contiguous()

        return {"input": input_ids, "positions": positions}, labels
