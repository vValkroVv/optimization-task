from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.gift_moe._base.model import GiftMoEModelConfig


@config(type="gift_3_5")
class GiftHybridModelConfig(GiftMoEModelConfig):
    """GDN + MLA hybrid; GIFT 4 is a preset of it (plain norms, softcapped head)."""

    _owner_spec: ClassVar[str] = (
        "torchtitan.models.gift_moe.gift_3_5.model:GiftHybridModel"
    )
    _runtime_spec: ClassVar[str] = "torchtitan.models.gift_moe.versions:HYBRID"
