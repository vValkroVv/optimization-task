"""Record the common compiler prerequisite before native campaign startup."""
import json
import os
from pathlib import Path

import torch
import torch._inductor.config as inductor_config

options = {
    "rank": int(os.environ["RANK"]),
    "torch_version": torch.__version__,
    "triton.mix_order_reduction": inductor_config.triton.mix_order_reduction,
    "triton.cudagraphs": inductor_config.triton.cudagraphs,
    "graph_partition": inductor_config.graph_partition,
}
assert options["triton.mix_order_reduction"] is False
requested_graph = os.environ["TORCHINDUCTOR_CUDAGRAPHS"]
assert requested_graph in {"0", "1"}
assert options["triton.cudagraphs"] is (requested_graph == "1")
assert options["graph_partition"] is True
output_dir = Path(__file__).resolve().parent / "compiler-options"
output_dir.mkdir(exist_ok=True)
with (output_dir / f"rank-{options['rank']:05d}.json").open("x") as output:
    json.dump(options, output, indent=2)
    output.write("\n")
print("OPT004 compiler prerequisite: " + json.dumps(options), flush=True)

from torchtitan.experiments.cuda_graph_campaign.train_scan_fix import main

if __name__ == "__main__":
    main()
