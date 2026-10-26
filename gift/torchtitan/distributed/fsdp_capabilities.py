"""Device eligibility shared by FSDP runtime and in-container preflight."""


def fsdp_symmetric_memory_supported() -> bool:
    """FSDP symmetric-memory communication requires NVIDIA SM90 or newer."""
    import torch
    import torch.version

    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability() >= (9, 0)
    )
