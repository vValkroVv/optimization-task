"""Make the existing n8 gather comparison use real two-rank head collectives."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import frontier_gather
import replay


def overlay(function, changes, destination):
    source = inspect.getsource(function)
    for before, after in changes:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    destination.write_text(source)
    methods = {}
    exec(compile(source, str(destination), "exec"), vars(replay), methods)
    return methods[function.__name__]


def main():
    arm, output_prefix = sys.argv[1:]
    assert os.environ["WORLD_SIZE"] == os.environ["LOCAL_WORLD_SIZE"] == "2"
    rank = int(os.environ["RANK"])
    assert rank == int(os.environ["LOCAL_RANK"]) in (0, 1)
    prefix = Path(output_prefix + f".rank-{rank}")
    output = Path(str(prefix) + ".json")
    replay._dual_rank = rank
    replay.make_head = overlay(
        replay.make_head,
        (
            ('init_device_mesh("cuda", (1,),', 'init_device_mesh("cuda", (2,),'),
            ("dp_shard=1", "dp_shard=2"),
            ("world_size=1", "world_size=2"),
        ),
        Path(str(prefix) + ".head-overlay.py"),
    )
    replay.main = overlay(
        replay.main,
        (
            ("torch.cuda.set_device(0)", "torch.cuda.set_device(_dual_rank)"),
            (
                '    rendezvous = args.output.with_suffix(".rendezvous").resolve().as_uri()\n'
                '    dist.init_process_group("nccl", init_method=rendezvous, rank=0, world_size=1)',
                '    dist.init_process_group("nccl", device_id=torch.device("cuda", _dual_rank))\n'
                '    assert dist.get_world_size() == 2 and dist.get_rank() == _dual_rank',
            ),
            ('"name": torch.cuda.get_device_name(0)', '"name": torch.cuda.get_device_name(_dual_rank)'),
            (
                '"singleton_reduction": "force-sum" if args.sum_reduction else "native-factor-1",',
                '"reduction": "NCCL SUM reduce-scatter over two ranks, FP32, divide factor 1",',
            ),
            (".view(128256, 4096).cpu()", ".view(-1, 4096).cpu()"),
        ),
        Path(str(prefix) + ".main-overlay.py"),
    )

    count = 0
    dist = replay.dist
    native_reduce = dist.reduce_scatter_tensor

    def counted_reduce(output, input, op=dist.ReduceOp.SUM, group=None, **kwargs):
        nonlocal count
        assert group.rank() == rank and group.size() == 2 and dist.get_backend(group) == "nccl"
        assert op == dist.ReduceOp.SUM
        assert output.dtype == input.dtype == replay.torch.float32
        assert output.numel() == 64128 * 4096 and input.numel() == 128256 * 4096
        count += 1
        return native_reduce(output, input, op=op, group=group, **kwargs)

    dist.reduce_scatter_tensor = counted_reduce
    sys.argv[1:] = [arm, "8", str(output)]
    frontier_gather.main()
    result = json.loads(output.read_text())
    assert result["gather_probe"]["group_rank"] == rank
    assert result["gather_probe"]["group_size"] == 2
    assert count == 23, count
    assert replay.torch.cuda.current_device() == rank
    result["device"].update(rank=rank, current_device=rank, visible_uuid=os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rank])
    result["native_fp32_sum_reduce_scatter_calls"] = count
    result["dual_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
