from dataclasses import dataclass


@dataclass(frozen=True)
class MoeModules:
    DeepseekGMMMoeBlock: type
    SonicGMMMoeBlock: type
