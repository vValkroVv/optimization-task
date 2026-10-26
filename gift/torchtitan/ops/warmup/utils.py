from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class DeepGemmWarmupCfg:
    cache_dir: str
    device: str
    dtype: torch.dtype

    hid_dim: int
    int_dim: int
    num_groups: int

    # IMPORTANT: step/min_size/max_size interpretation:
    #   - step: granularity (tokens) you want to cover (e.g., 128)
    #   - min_size: MIN TOTAL (m_sum / k_sum) to warm (not per-expert)
    #   - max_size: MAX PER-EXPERT tokens you want to allow in synthesized group sizes
    #              (buffers are sized as max_size * num_groups)
    step: int
    min_size: int
    max_size: int

    # Cap the token-buffer allocation to avoid OOM when max_size * num_groups
    # is very large.  The sweep is limited to totals that fit within this
    # budget.  None = no cap (default: max_size * num_groups).
    max_buffer_tokens: int | None = field(default=None)
    num_sms: int | None = field(default=None)
