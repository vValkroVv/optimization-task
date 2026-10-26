# positional encoding
import torch


class PositionProjector3d:
    @classmethod
    def _3d(cls, data: torch.Tensor) -> torch.Tensor:
        length, h, w, d_latent = data.shape

        x = torch.arange(h * w).reshape((h, w)) % w
        y = torch.arange(h * w).reshape((h, w)) // w
        t = torch.arange(length).repeat_interleave(h * w)

        x = x.unsqueeze(0).repeat_interleave(length, dim=0)
        y = y.unsqueeze(0).repeat_interleave(length, dim=0)
        t = t.reshape(length, h, w)

        return torch.stack([t, y, x], -1)

    @classmethod
    def _1d(cls, data: torch.Tensor) -> torch.Tensor:
        """input: Tensor (length, )"""
        length = data.shape[0]
        return torch.arange(length)[:, None].repeat(1, 3)

    @classmethod
    def offset(cls, position: torch.Tensor, offset: int | torch.Tensor) -> torch.Tensor:
        return position + offset

    @classmethod
    def synchronize(
        cls, posititions: torch.Tensor, ref_positions: torch.Tensor
    ) -> torch.Tensor:
        """positions:"""
        ref_d = ref_positions[..., 0][-1].max() - ref_positions[..., 0][0].min()
        ref_min = ref_positions[..., 0][0].min()

        d = posititions[..., 0][-1].max() - posititions[..., 0][0].min()
        min_ = posititions[..., 0][0].min()

        return (posititions - min_) / d * ref_d + ref_min

    # still need to add center offset and, in a separate function, sequence offset
    @classmethod
    def process(cls, data: torch.Tensor) -> torch.Tensor:
        n_dims = len(data.shape)
        if n_dims == 4:
            return cls._3d(data)
        if n_dims in [1, 2]:
            return cls._1d(data)
        return cls._1d(data)

    @classmethod
    def __call__(cls, data: torch.Tensor) -> torch.Tensor:
        return cls.process(data)
