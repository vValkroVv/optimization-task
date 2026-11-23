"""Test whether moving unique-count readback into the existing count exchange helps.

The fixed-shape route is an experimental substitution for the embedding replay.
Its performance is unmeasured. Enable it explicitly with --fixed-shape-route.
"""

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import json
import os
import runpy

import torch
from torch import distributed as dist


SOURCE_ROOT = Path(
    os.environ.get("ROW_OWNER_REPLAY_SOURCE_ROOT", Path(__file__).resolve().parents[2])
).resolve()


def fixed_shape_route(owner, ids):
    """Return Original A's ordered IDs, inverse, requests, and splits."""
    world = owner.mesh_info.shard_mesh_size
    rank = owner.mesh_info.shard_mesh_rank
    group = owner.mesh_info.shard_process_group
    device = owner.weight.device
    with owner._stage("partition_ids"):
        valid_type = ids.dtype == torch.int64 and ids.device == device
        flat = (
            ids.reshape(-1)
            if valid_type
            else torch.empty(0, dtype=torch.int64, device=device)
        )
        valid = ((flat >= 0) & (flat < owner.num_embeddings)).all()
        if not valid_type:
            valid.zero_()
        ordered, permutation = flat.clamp(0, owner.num_embeddings - 1).sort()
        length = flat.numel()
        starts = torch.ones(length, dtype=torch.bool, device=device)
        starts[1:] = ordered[1:] != ordered[:-1]
        prefix = flat.new_empty(length + 1)
        prefix[:1].zero_()
        torch.cumsum(starts, dim=0, dtype=torch.int64, out=prefix[1:])
        run = prefix[1:] - 1

        # Run starts fill [0, U). At a duplicate, i - run is the one-based
        # duplicate count. Thus duplicates fill [U, N) from the end once each.
        position = torch.arange(length, dtype=torch.int64, device=device)
        destination = torch.where(starts, run, length - position + run)
        packed = torch.empty_like(flat).scatter_(0, destination, ordered)
        inverse = torch.empty_like(flat).scatter_(0, permutation, run)

        boundaries = torch.arange(world + 1, dtype=torch.int64, device=device)
        owner_offsets = torch.searchsorted(ordered, boundaries * owner.rows_per_owner)
        metadata = flat.new_empty(world + 1)
        metadata[:world] = prefix.index_select(0, owner_offsets).diff()
        metadata[world] = valid

    with owner._stage("counts"):
        matrix = metadata.new_empty(world * (world + 1))
        dist.all_gather_into_tensor(matrix, metadata, group=group)
        host = matrix.view(world, world + 1).cpu()
        if not bool(host[:, world].all()):
            raise ValueError(
                "row-owner IDs must be signed int64 on the weight device and in "
                f"[0, {owner.num_embeddings}) on every rank"
            )
        send = tuple(host[rank, :world].tolist())
        receive = tuple(host[:, rank].tolist())
    with owner._stage("request_exchange"):
        # The existing count read supplies U, so no dynamic device output is
        # needed to obtain the same exact-length NCCL input view.
        unique = packed.narrow(0, 0, sum(send))
        requested = unique.new_empty(sum(receive))
        dist.all_to_all_single(
            requested, unique, list(receive), list(send), group=group
        )
    return unique, inverse, requested, send, receive


@contextmanager
def use_fixed_shape_route():
    """Install the experiment in this process for an explicitly bounded replay."""
    from torchtitan.distributed.gfsdp.row_owner_embedding import RowOwnerEmbedding

    with patch.object(RowOwnerEmbedding, "_route", fixed_shape_route):
        yield


def main(argv=None):
    harness = SOURCE_ROOT / "runs/ffn-row-owner/components.py"
    replay = runpy.run_path(str(harness))
    parser = replay["parser"]()
    parser.add_argument(
        "--fixed-shape-route", action="store_true", help="enable the experimental route"
    )
    args = parser.parse_args(argv)
    if args.fixed_shape_route and args.variant != "owner":
        parser.error("--fixed-shape-route requires --variant owner")
    replay["validate"](args)
    if not args.execute:
        plan = replay["plan"](args)
        plan["fixed_shape_route"] = args.fixed_shape_route
        print(json.dumps(plan, indent=2))
    elif args.fixed_shape_route:
        with use_fixed_shape_route():
            digest = sha256(Path(__file__).read_bytes()).hexdigest()
            print(
                f"Experimental fixed-shape row-owner route active: sha256={digest}",
                flush=True,
            )
            replay["execute"](args)
    else:
        replay["execute"](args)


if __name__ == "__main__":
    main()
