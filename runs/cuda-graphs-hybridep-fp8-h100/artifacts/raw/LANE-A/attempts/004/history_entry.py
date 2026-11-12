"""Enable built-in CUDA Graph Trees allocation history for this diagnostic."""

import torch._inductor.config as inductor_config

inductor_config.triton.cudagraph_trees_history_recording = True

from torchtitan.experiments.cuda_graph_campaign.train_scan_fix import main


if __name__ == "__main__":
    main()
