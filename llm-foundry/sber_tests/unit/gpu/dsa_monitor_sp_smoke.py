from types import SimpleNamespace

import torch
import torch.distributed as torch_dist

from llmfoundry.callbacks.dsa_kl_monitor import DSAMonitor


class _Logger:
    def __init__(self):
        self.metrics = None

    def log_metrics(self, metrics):
        self.metrics = metrics


def main():
    torch_dist.init_process_group("nccl")
    rank = torch_dist.get_rank()
    world_size = torch_dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    attention_stats = (
        [[0.8, 1.0, 4.0, 10.0], [1.6, 1.0, 4.0, 10.0]]
        if rank == 0
        else [[1.0, 2.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]]
    )
    indexer_stats = (
        [[2.0, 3.0, 1.0, 1.0, 4.0, 10.0], [4.0, 5.0, 2.0, 1.0, 4.0, 10.0]]
        if rank == 0
        else [[6.0, 9.0, 3.0, 2.0, 0.0, 0.0], [8.0, 10.0, 4.0, 2.0, 0.0, 0.0]]
    )
    layer = SimpleNamespace(
        attention_mass_monitor_stats=torch.tensor(attention_stats, device=device),
        indexer_topk_monitor_stats=torch.tensor(indexer_stats, device=device),
    )
    callback = DSAMonitor(
        attention_mass={"enabled": True, "topk_values": [2, 4]},
        indexer_topk={"enabled": True, "topk_values": [2, 4]},
    )
    callback.layers = [layer]
    callback.tags = ["layer_3"]
    callback._device = device
    callback._attention_mass_now = True
    callback._indexer_topk_now = True
    logger = _Logger()

    callback.batch_end(None, logger)

    monitored_q_count = 1.0 + 2.0 * (world_size - 1)
    assert logger.metrics is not None
    torch.testing.assert_close(
        torch.tensor(
            logger.metrics["dsa_attention_mass/topk_2/attention_mass/layer_3"]
        ),
        torch.tensor((0.8 + 1.0 * (world_size - 1)) / monitored_q_count),
    )
    torch.testing.assert_close(
        torch.tensor(
            logger.metrics["dsa_attention_mass/topk_4/attention_mass/layer_3"]
        ),
        torch.tensor((1.6 + 2.0 * (world_size - 1)) / monitored_q_count),
    )
    torch.testing.assert_close(
        torch.tensor(logger.metrics["dsa_indexer_topk/topk_2/position_range/layer_3"]),
        torch.tensor((2.0 + 6.0 * (world_size - 1)) / monitored_q_count),
    )
    torch.testing.assert_close(
        torch.tensor(logger.metrics["dsa_attention_mass/monitored_q_count"]),
        torch.tensor(monitored_q_count),
    )
    torch.testing.assert_close(
        torch.tensor(logger.metrics["dsa_indexer_topk/monitored_q_count"]),
        torch.tensor(monitored_q_count),
    )
    assert logger.metrics["dsa_attention_mass/eligible_q_count"] == 4
    assert logger.metrics["dsa_attention_mass/total_q_count"] == 10
    if rank == 0:
        print("DSA monitor SP all-reduce smoke passed")

    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
