"""Check additional cuBLASLt algorithm 66 configurations for the 100B C8 dW chain."""

import argparse
from collections import Counter
from contextlib import ExitStack
import ctypes as c
import hashlib
from itertools import product
import json
import logging
import os
from pathlib import Path
from statistics import median
import subprocess

import torch

from tuning.lt.lt_bench import (
    EXACT_CE_SHA256,
    bit_mismatches,
    chain,
    check_beta,
    load_operands,
    measure_chains,
    measure_single_arm_memory,
    screen,
    write_json,
)
from tuning.lt.lt_writer import Search, make_dw_writer, native, operand_key


CASE = "100b-c8"
CHUNKS = 8
WORKSPACE = 4
MAX_NEW_PER_ORDER = 48
TILE, CUSTOM, CLUSTER = 1, 5, 8
CAP_TILE_IDS, CAP_CUSTOM_MAX, CAP_ALIGNMENT_A = 6, 7, 16
# AUTO and explicit portable cluster shapes with at most eight CTAs on H100.
CLUSTERS = (0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12)


class Neighborhood(Search):
    def __init__(self, shape, alignments, stream, resources):
        super().__init__(shape, alignments, stream, resources)
        library = c.CDLL(f"libcublasLt.so.{torch.version.cuda.split('.')[0]}")
        self.set_config = library.cublasLtMatmulAlgoConfigSetAttribute
        self.set_config.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_size_t]
        self.set_config.restype = c.c_int
        self.capability_attr = library.cublasLtMatmulAlgoCapGetAttribute
        self.capability_attr.argtypes = [*self.set_config.argtypes, c.POINTER(c.c_size_t)]
        self.capability_attr.restype = c.c_int

    def capability(self, algorithm, attr, value):
        native._check(self.capability_attr, c.byref(algorithm), attr, c.byref(value), c.sizeof(value), None)
        return value.value

    def supported_tiles(self, algorithm):
        size = c.c_size_t()
        native._check(self.capability_attr, c.byref(algorithm), CAP_TILE_IDS, None, 0, c.byref(size))
        values = (c.c_uint32 * (size.value // c.sizeof(c.c_uint32)))()
        native._check(self.capability_attr, c.byref(algorithm), CAP_TILE_IDS, values, size.value, c.byref(size))
        return list(values)

    def expand(self, seeds):
        results = list(seeds)
        counts, proposed, capabilities = {}, {}, {}
        for order in self.layouts:
            local = [seed for seed in seeds if seed.order == order]
            if any(seed.config["id"] != 66 or seed.config["split_k"] != 1 or
                   seed.config["reduction"] != 0 or seed.config["stages"] != 35 or
                   seed.config["swizzle"] != 0 or seed.config["inner"] != 0 for seed in local):
                raise ValueError("Unexpected configuration in the recorded algorithm 66 seeds")
            anchor = next(seed for seed in local if seed.config["tile"] == 29)
            tiles = self.supported_tiles(anchor.algorithm)
            custom_max = self.capability(anchor.algorithm, CAP_CUSTOM_MAX, c.c_int32())
            required = [self.capability(anchor.algorithm, CAP_ALIGNMENT_A + i, c.c_uint32()) for i in range(4)]
            align_a, align_b, align_cd = self.alignments
            if order == "column":
                align_a, align_b = align_b, align_a
            if any(alignment < minimum for alignment, minimum in
                   zip((align_a, align_b, align_cd, align_cd), required, strict=True)):
                raise ValueError(f"Algorithm 66 requires stronger {order} buffer alignment")
            capabilities[order] = {"tiles": tiles, "custom_max": custom_max, "alignment_abcd": required}
            used = {(seed.config["tile"], seed.config["custom"], seed.config["cluster"]) for seed in local}
            # The nearby cluster/custom combinations come first, then new tiles
            # with the same settings as the fastest heuristic for this layout.
            customs = sorted(range(custom_max + 1), key=lambda custom: abs(custom - anchor.config["custom"]))
            cluster = anchor.config["cluster"]
            tiles_first = [t for t in tiles if t != 29] or [seed.config["tile"] for seed in local if seed != anchor]
            neighborhood = [(29, custom, cluster) for custom in customs]
            neighborhood += [(29, anchor.config["custom"], cluster_id) for cluster_id in CLUSTERS]
            neighborhood += [(tile, anchor.config["custom"], cluster) for tile in tiles_first]
            neighborhood += list(product((29,), customs, CLUSTERS))
            for seed in local:
                tile, custom = seed.config["tile"], seed.config["custom"]
                neighborhood += list(product((tile,), (custom,), CLUSTERS))
                neighborhood += list(product((tile,), customs, (seed.config["cluster"],)))
            proposed[order] = len(set(neighborhood) - used)
            stats = counts[order] = Counter()
            for key in neighborhood:
                if key in used:
                    continue
                used.add(key)
                stats["attempted"] += 1
                tile, custom, cluster_id = key
                algorithm = native._Algo.from_buffer_copy(bytes(anchor.algorithm))
                try:
                    for attr, value in ((TILE, c.c_uint32(tile)), (CUSTOM, c.c_uint32(custom)),
                                        (CLUSTER, c.c_uint16(cluster_id))):
                        native._set_attribute(self.set_config, algorithm, attr, value)
                    candidate = self._candidate(order, algorithm, -1, WORKSPACE)
                except RuntimeError as error:
                    stats[str(error)] += 1
                    continue
                if (candidate.config["tile"], candidate.config["custom"], candidate.config["cluster"]) != key:
                    stats["normalized_configuration"] += 1
                    continue
                if candidate.config["split_k"] != 1 or candidate.config["reduction"] != 0:
                    raise RuntimeError(f"Unexpected split or reduction in {candidate.receipt()}")
                results.append(candidate)
                stats["accepted"] += 1
                if stats["accepted"] == MAX_NEW_PER_ORDER:
                    break
        return results, {"counts": counts, "proposed": proposed, "capabilities": capabilities}


def full_chain_parity(factories, operands, slot, stream):
    expected = torch.empty_like(slot)
    expected.fill_(float("nan"))
    chain(factories["fixed_5_5"](), operands, expected, stream)
    mismatches = {}
    activation = {}
    for name, factory in factories.items():
        writer = factory()
        slot.fill_(float("nan"))
        chain(writer, operands, slot, stream)
        mismatches[name] = int(bit_mismatches(slot, expected))
        if hasattr(writer, "receipt"):
            activation[name] = writer.receipt()
            if activation[name]["tuned_calls_beta0_beta1"] != [1, 7]:
                raise RuntimeError(f"Calibrated dispatch did not activate for {name}")
            writer.close()
    return mismatches, activation


def timed_match(factories, operands, slot, stream, samples):
    with ExitStack() as resources:
        arms = {name: factory() for name, factory in factories.items()}
        for writer in arms.values():
            if hasattr(writer, "close"):
                resources.callback(writer.close)
        return measure_chains(arms, operands, slot, stream, samples)


def comparison(timings, name):
    result = {}
    for metric, read in (("graph_ms", lambda arm: arm["graph_ms"]),
                         ("wall_ms", lambda arm: [item["wall_ms"] for item in arm["eager"]])):
        first, repeat, candidate = (read(timings[key]) for key in ("fixed_5_5", "fixed_repeat", name))
        delta = lambda left, right: median(a - b for a, b in zip(left, right, strict=True))
        result[metric] = {"control_repeat_minus_fixed": delta(repeat, first),
                          "candidate_minus_fixed": delta(candidate, first),
                          "candidate_minus_repeat": delta(candidate, repeat)}
    return result


def reproducible_gain(discovery, confirmation):
    for phase in (discovery, confirmation):
        for metric in phase.values():
            noise = abs(metric["control_repeat_minus_fixed"])
            if max(metric["candidate_minus_fixed"], metric["candidate_minus_repeat"]) >= -noise:
                return False
    return True


def run(root, out, stream):
    fixture = root / "inputs" / "100b.pt"
    previous = json.loads((root / "tuning/lt/results/100b-c8.json").read_text())
    operands, provenance = load_operands(CASE, fixture, root / "native-ce/centered-folded-v1.so")
    slot = torch.empty((operands[0][0].size(1), operands[0][1].size(1)),
                       dtype=torch.float32, device=stream.device)
    shape, alignments = operand_key(*operands[0], slot)
    if [operand_key(*pair, slot) for pair in operands] != [(shape, alignments)] * CHUNKS:
        raise RuntimeError("All eight operand pairs must match the measured shape and alignments")
    with ExitStack() as resources:
        search = Neighborhood(shape, alignments, stream, resources)
        if (previous["shape"], previous["alignments"], previous["cublasLt_version"], previous["provenance"]) != (
            list(shape), list(alignments), search.version, provenance
        ):
            raise RuntimeError("Live fixture or runtime disagrees with the recorded heuristic search")
        seeds = [search.restore(item) for item in previous["candidates"]]
        candidates, enumeration = search.expand(seeds)
        print(f"{CASE} AlgoCheck {enumeration['counts']}", flush=True)
        if len(candidates) == len(seeds):
            raise RuntimeError("AlgoCheck did not find any additional compatible configurations")
        screens = screen(search, candidates, operands, slot, 4)

        def fixed():
            writer = native.make_head_dw_writer("new", CHUNKS, *shape, stream)
            if writer.pair != "5.5":
                raise RuntimeError("Native C8 control did not activate plan 5.5")
            return writer

        beta_mismatches = check_beta(search, candidates, screens, fixed(), operands, slot)
        best_beta = [[i for i in screens[beta]["ranked"] if beta_mismatches[beta][i] == 0][:5]
                     for beta in ("0", "1")]
        known = json.loads((root / "tuning/lt/results/100b-c8-plan-bitwise.json").read_text())
        shortlist = []
        for i, j in product(*best_beta):
            if max(i, j) < len(seeds):
                continue
            ms = screens["0"]["candidates"][i]["median_ms"] + 7 * screens["1"]["candidates"][j]["median_ms"]
            shortlist.append((ms, i, j))
        plans = {"calibrated_previous": known}
        for rank, (_, i, j) in enumerate(sorted(shortlist)[:4], 1):
            plans[f"extended_{rank}"] = {**known, "pair": [candidates[i].receipt(), candidates[j].receipt()],
                                         "selected": f"extended_{rank}"}
            del plans[f"extended_{rank}"]["numerics"]

        factories = {"fixed_5_5": fixed, "fixed_repeat": fixed}
        factories.update({name: lambda plan=plan: make_dw_writer(receipt=plan) for name, plan in plans.items()})
        parity, activation = full_chain_parity(factories, operands, slot, stream)
        tested = {name: factory for name, factory in factories.items() if parity[name] == 0}
        if "fixed_repeat" not in tested or "calibrated_previous" not in tested:
            raise RuntimeError("Control parity failed")
        discovery = timed_match(tested, operands, slot, stream, 8)
        compared = {name: comparison(discovery, name) for name in tested if name.startswith("extended_")}
        finalist = min(compared, key=lambda name: max(
            compared[name]["wall_ms"]["candidate_minus_fixed"],
            compared[name]["wall_ms"]["candidate_minus_repeat"],
        ), default=None)
        confirmation, confirmed_delta, selected, isolated = {}, {}, None, {}
        if finalist is not None:
            finalists = {name: tested[name] for name in ("fixed_repeat", "fixed_5_5", finalist)}
            confirmation = timed_match(finalists, operands, slot, stream, 12)
            confirmed_delta = comparison(confirmation, finalist)
            isolated = measure_single_arm_memory({"fixed_5_5": fixed, finalist: tested[finalist]},
                                                  operands, slot, stream)
            if reproducible_gain(compared[finalist], confirmed_delta):
                selected = finalist
                plans[finalist]["numerics"] = {"control_bit_mismatches": parity[finalist],
                                               "reference": "native 5.5"}
                write_json(out / f"{CASE}-extended-plan.json", plans[finalist])

        receipt = {
            "case": CASE, "shape": shape, "alignments": alignments, "provenance": provenance,
            "fixture": str(fixture), "fixture_bytes": fixture.stat().st_size,
            "ce_sha256": EXACT_CE_SHA256, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cublasLt_version": search.version, "device": torch.cuda.get_device_name(stream.device),
            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                              text=True).splitlines()[0],
            "native_source_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "scope": "One H100, eight preallocated dW writes with BF16 operands ready; warm single-stream graph and eager wall spans. No projection, copies, or collectives within timings.",
            "acceptance": "Exact beta and chain parity. Both gate metrics beat both identical 5.5 controls by more than their paired median difference in both windows. Workspace capped at four bytes.",
            "heuristic_count": len(seeds), "enumeration": enumeration,
            "candidates": [item.receipt() for item in candidates], "screens": screens,
            "per_beta_control_bit_mismatches": beta_mismatches, "shortlist_plans": plans,
            "chain_mismatches": parity, "activation": activation, "discovery": discovery,
            "discovery_deltas": compared, "confirmation": confirmation,
            "confirmation_deltas": confirmed_delta, "isolated_memory": isolated,
            "finalist": finalist, "selected": selected,
        }
        write_json(out / f"{CASE}-extended.json", receipt)
        print(f"{CASE} finalist={finalist} selected={selected} discovery={compared} confirmation={confirmed_delta}",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one authorized GPU")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    with torch.cuda.stream(torch.cuda.Stream()):
        run(args.root, args.out, torch.cuda.current_stream())


if __name__ == "__main__":
    main()
