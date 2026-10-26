import enum
from typing import Literal

FLASH_ATTN_BLOCK_SIZE = 64
# Queue get/put timeout so workers can observe stop_event.
IPC_TIMEOUT = 1.0
# A blocking get/result longer than this is a stall once the loader is warm.
SLOW_GET = 0.01
EMPTY_QUEUE_WARNING_TIMEOUT = 600.0
SLOW_GET_WARN_INTERVAL = 60.0
DIFFUSION_GENERATION = ["<gen_image>", "<gen_video>", "<gen_audio>"]

# same-modality inputs must be adjacent
ENCODER_ORDER = [
    "text",
    "semantic_visual",
    "pixel_visual",
    "gen_visual",
    "semantic_audio",
    "pixel_audio",
    "gen_audio",
]
ENCODER_MODALITY_MAPPING = {
    "text": "text",
    "semantic_visual": "visual",
    "pixel_visual": "visual",
    "gen_visual": "visual",
    "semantic_audio": "audio",
    "pixel_audio": "audio",
    "gen_audio": "audio",
}

assert set(ENCODER_MODALITY_MAPPING) == set(ENCODER_ORDER)

AUXILIARY_TOKENS = {
    "<text>": ("", ""),
    # semantic tokens
    "<semantic_image>": ("<semantic_image/>", "</semantic_image>"),
    "<semantic_video>": ("<semantic_video/>", "</semantic_video>"),
    "<semantic_audio>": ("<semantic_audio/>", "</semantic_audio>"),
    # pixel tokens
    "<pixel_image>": ("<pixel_image/>", "</pixel_image>"),
    "<pixel_video>": ("<pixel_video/>", "</pixel_video>"),
    "<pixel_audio>": ("<pixel_audio/>", "</pixel_audio>"),
    # start generation
    "<gen_image>": ("<gen_image/>", "</gen_image>"),
    "<gen_video>": ("<gen_video/>", "</gen_video>"),
    "<gen_audio>": ("<gen_audio/>", "</gen_audio>"),
}
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
TIMESTEP_TOKEN = "<dtimestep>"
PAD_TOKEN = "<pad>"

ENCODER_MAPPING = {
    "text": "text",
    "semantic_image": "semantic_visual",
    "semantic_video": "semantic_visual",
    "pixel_image": "pixel_visual",
    "pixel_video": "pixel_visual",
    "gen_image": "gen_visual",
    "gen_video": "gen_visual",
    "gen_audio": "gen_audio",
}


class TokenMode(enum.StrEnum):
    # random tokens (for tests)
    SYNTHETIC = "synthetic"
    # S3 images/video
    TEXT = "text"
    # MDS streaming text
    VISION = "vision"


# in a layout string, in `template`, in `prpr_mapper`
DataModality = Literal[
    "text",
    "semantic_image",
    "semantic_video",
    "semantic_audio",
    "pixel_image",
    "pixel_video",
    "pixel_audio",
    "gen_image",
    "gen_video",
    "gen_audio",
]
