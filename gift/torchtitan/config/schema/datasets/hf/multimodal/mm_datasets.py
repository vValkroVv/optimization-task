from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.components.dataloader import ParallelAwareDataloaderConfig


@config(type="hf_multimodal")
class MMDataLoaderConfig(ParallelAwareDataloaderConfig):
    _owner_spec: ClassVar[str] = (
        "torchtitan.datasets.hf.multimodal.mm_datasets:MMDataLoader"
    )

    dataset: str = "cc12m-test"

    dataset_subset: str = ""
    """Dataset subset/config name."""

    infinite: bool = True

    # Batching configs
    packing_buffer_size: int = 0
    """Set to a value >0 to enable sample packing."""

    max_images_per_batch: int
    """Max images per batch to bound vision encoder memory."""

    # Preprocessing configs
    patch_size: int

    temporal_patch_size: int
    """Temporal patch size for video processing."""

    spatial_merge_size: int
    """Spatially merge visual tokens after encoder. e.g. 2 means 2x2=4 patches merged."""

    min_pixels: int

    max_pixels: int

    image_mean: tuple[float, ...]
    """Per-channel mean for image normalization."""

    image_std: tuple[float, ...]
    """Per-channel std for image normalization."""

    video_dir: str = ""
    """Base directory for video files (for datasets with video filename references)."""

    video_fps: float = 2.0
    """Target frames per second for video sampling."""

    video_min_frames: int = 4

    video_max_frames: int = 768
