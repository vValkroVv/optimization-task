"""Run the preserved isolated head and verify each effective plan on the rank."""

import json
import os
from pathlib import Path
import sys

from manifests import ENV_FILES, dw_plans, expected_routes, pin, read_pinned


GRID = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(GRID))
import replay


def signatures(pair):
    fields = ("order", "config", "algorithm", "workspace_bytes")
    return [tuple(item[field] for field in fields) for item in pair]


def main():
    check_base = replay.check_environment
    activate_base = replay.candidate_activation
    run_base = replay.run
    resolved = {}
    manifests = {}

    def check(args):
        digest = check_base(args)
        rank = int(os.environ["RANK"])
        if digest:
            for name, variable in ENV_FILES.items():
                resolved[name] = pin(os.environ[variable], os.environ[variable + "_SHA256"])
                manifests[name] = read_pinned(resolved[name])
            ce = "GIFT_EXACT_FOLDED_CE_LIBRARY"
            resolved["ce"] = pin(os.environ[ce], digest)
        elif any(key in os.environ for variable in ENV_FILES.values()
                 for key in (variable, variable + "_SHA256")):
            raise RuntimeError("Baseline received a tuned plan manifest")
        replay.write_json(args.out / f"resolved-{rank}.json", {
            "rank": rank, "local_rank": int(os.environ["LOCAL_RANK"]),
            "world_size": int(os.environ["WORLD_SIZE"]), "variant": args.variant,
            "model": args.model, "chunks": args.chunks, "seq": args.seq, "batch": args.batch,
            "cuda_devices": os.environ["CUDA_VISIBLE_DEVICES"].split(","),
            "manifests": resolved,
        })
        return digest

    def install(writers):
        from head_run import install_linear_ops
        from torchtitan.distributed.gfsdp import _staged_head_dw as staged
        from tuning.lt.lt_writer import make_dw_writer

        def factory(*unused):
            writer = make_dw_writer(policy="replay", receipt=resolved["dw"]["path"])
            writers.append(writer)
            return writer

        staged.make_head_dw_writer = factory
        return install_linear_ops(["linear"], Path(resolved["linear"]["path"]))

    def activate(args, output, counts, policy, writers, digest):
        from tuning.proj.lt_ops import CONFIG_FIELDS

        activation = activate_base(args, output, counts, policy, writers, digest)
        expected = expected_routes(args.model, args.chunks, args.seq,
                                   dw_plans(manifests["dw"]), manifests["linear"])
        dw = activation["plans"][0]
        steps = args.warmup + args.samples + 3
        expected_dw_calls = [steps, steps * (args.chunks - 1)] if expected["dw"] else [0, 0]
        if dw["tuned_calls_beta0_beta1"] != expected_dw_calls:
            raise RuntimeError(f"dW plan activation differs from {resolved['dw']}: {dw}")
        for plan in dw["plans"]:
            if expected["dw"]:
                if signatures(plan["pair"]) != signatures(expected["dw"]["pair"]):
                    raise RuntimeError(f"Executed a different dW algorithm pair: {plan}")
            elif plan["pair"] != "native-generic":
                raise RuntimeError(f"Expected native generic dW for {plan['shape']}")

        actual = {}
        for match in policy.cache.values():
            if match is None:
                raise RuntimeError(f"Selected linear descriptor fell back: {resolved['linear']}")
            plan, index = match
            capability = replay.torch.cuda.get_device_capability(plan.device)
            key = json.dumps([plan.library.head_lt_version(), list(capability), list(plan.problem)])
            config = plan.configs[index]
            selected = manifests["linear"].get(key)
            if selected is None or any(config[field] != selected[field] for field in CONFIG_FIELDS[:-1]):
                raise RuntimeError(f"Executed a linear algorithm absent from {resolved['linear']}: {key}")
            actual[key] = config
        for op, keys in expected["linear"].items():
            if bool(keys) != any(key in actual for key in keys) or counts[op] != steps * args.chunks * bool(keys):
                raise RuntimeError(f"Unexpected effective {op} descriptor or call count: {counts}, {actual}")
        if Path(policy.manifest).resolve() != Path(resolved["linear"]["path"]):
            raise RuntimeError(f"Linear policy used a different manifest: {policy.manifest}")
        return dict(activation, manifests=resolved, linear_descriptors=actual)

    def run(args):
        run_base(args)
        rank = int(os.environ["RANK"])
        path = args.out / f"rank-{rank}.json"
        output = json.loads(path.read_text())
        stats = replay.torch.cuda.memory_stats(rank)
        output["allocator_retry_count_process"] = stats["num_alloc_retries"]
        replay.write_json(path, output)

    replay.check_environment = check
    replay.install_candidate = install
    replay.candidate_activation = activate
    replay.run = run
    replay.main()


if __name__ == "__main__":
    main()
