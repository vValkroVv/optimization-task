import inspect
import logging
import os
import warnings

import torch
from packaging import version

logger = logging.getLogger(__name__)
_warned_missing_params: set[tuple[str, int | tuple]] = set()

warnings.simplefilter("once", UserWarning)


def _sm90_and_triton_version_check() -> None:
    import triton

    device_capability = torch.cuda.get_device_capability()
    triton_version = version.parse(triton.__version__)
    if device_capability != (9, 0) or triton_version != version.parse("3.1.0"):
        logger.warning(
            f"For triton kernels expected SM90 and Triton 3.1.0, but got {device_capability} and {triton_version}."
            + " Results may be suboptimal, and rewarmup may be required."
            + " Check llmfoundry/models/ops/warmup/README.md (Triton section) for more details."
        )


def params_to_kernel_kwargs(
    static_params: dict, key: int | tuple, **kwargs: object
) -> dict:
    if key not in static_params:
        frame = inspect.currentframe()
        try:
            caller = frame.f_back if frame is not None else None
            base_name = (
                os.path.basename(caller.f_code.co_filename)
                if caller is not None
                else "unknown"
            )
        finally:
            del frame
            del caller
        warning_key = (base_name, key)
        if warning_key not in _warned_missing_params:
            warnings.warn(
                f"Key for static parameters not found: {key} in {base_name}."
                + " Default parameters are set from triton kernel documentation."
                + " Results may be not optimal, please check"
                + " /llmfoundry/models/ops/warmup/README.md (Triton section) for more information."
            )
            _warned_missing_params.add(warning_key)
        if kwargs:
            return kwargs

        return dict(num_warps=4, num_stages=3, num_ctas=1, maxnreg=None)
    return static_params[key]
