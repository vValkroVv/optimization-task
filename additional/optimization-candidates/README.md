# Preserved optimization candidates

Read the [approach overview and code map](../../docs/optimization-candidates.md).

| Collection | Pinned GIFT revision | Contents |
| --- | --- | --- |
| [Kindulov](kindulov/manifest.json) | `b983803337dfcc34fdd571b38aaf5e01df583bbe` | Softcap/CE kernels, FFN/expert recomputation, integration, recipe and tests |
| [Hidden z-loss](hidden-z-loss/manifest.json) | `81054a3383dc349a6690cc2b4aa0bce0dc73b5e9` | GIFT FP8 integration, CudaCherry API adapter and dependency pin, tests |
| [Vision-related runtime](vision-runtime/manifest.json) | `860152d111e4bac25daecbe3cc0cd84c68c8af9c` | CompactCE, streaming prefetch, GFSDP, frozen storage and metrics |

These are selected, byte-preserved upstream files, including their license.
`source/` preserves upstream paths; `manifest.json` records each file's SHA-256
and Git blob ID. They are reference data, outside the active training import path,
not runnable checkouts or newly validated implementations. Upstream documents
retain their original claims and relative links; use the overview for maintained
local navigation and applicability to the crop.

The external CudaCherry kernels at the recorded dependency revision are not
included. The vision collection has not been confirmed as the promised team list.
These limits are detailed in the overview.
