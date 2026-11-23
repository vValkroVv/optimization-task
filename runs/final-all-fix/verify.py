"""Check a completed one-update probe against the final recipe and per-rank observations."""

import argparse
import json
import math
import re
from pathlib import Path

import yaml


def verify(attempt: Path, ep: int, loss_chunks: int = 2) -> dict:
    def check(condition, message):
        if not condition:
            raise ValueError(message)

    check((attempt / "torchrun.exit").read_text().strip() == "0", "torchrun failed")
    config = yaml.safe_load(
        (attempt / "native/config/resolved_config.yaml").read_text()
    )
    check(
        config["compile"]["enable"]
        and set(config["compile"]["components"]) >= {"model", "loss"},
        "model or loss compilation is disabled",
    )
    check(
        config["activation_checkpoint"]["mode"] == "selective"
        and config["loss"]["num_chunks"] == loss_chunks,
        "unexpected checkpointing or head configuration",
    )
    check(config["dataloader"]["num_prefetch_batches"] == 32, "prefetch depth")
    par = config["parallelism"]
    check(
        par["expert_parallel_degree"] == ep and par["data_parallel_shard_degree"] == 8,
        "EP / DP mesh",
    )
    for key in (
        "split_layer0_ffn",
        "split_mtp_ffn",
        "row_owner_embeddings",
        "mtp_transition_prefetch",
        "reuse_bf16_expert_weights",
        "fp8_expert_weights",
        "direct_expert_dw",
        "retain_tail_weights",
    ):
        check(par["gfsdp_" + key], key)
    check(par["gfsdp_compact_expert_owner"] == (ep == 4), "expert transport selection")
    for kind in ("gdn_moe", "mla_moe"):
        dispatcher = config["model"]["config"]["blocks"][kind]["moe"]["experts"][
            "token_dispatcher"
        ]
        selector = dispatcher["token_drop"]
        check(
            dispatcher["type"] == "hybridep"
            and selector["branchless_capacity_drop"]
            and selector["drop_by_group_capacity"]
            and selector["expert_capacity_tokens"] == 50000,
            f"{kind} capacity policy",
        )
    result = {"ep": ep, "ranks": {}}
    for rank in range(8):
        record = json.loads((attempt / f"activation-rank-{rank}.json").read_text())
        check(record["rank"] == rank, "receipt rank")
        check(
            record["completed_microbatches"] == record["microbatches_per_update"] == 32,
            "accumulation",
        )
        queue = record["prefetch"]
        check(queue["depth"] == 32 and queue["live"], f"rank {rank}: V2 producer")
        check(
            record["embedding"]
            and all(v["compact_grad_used"] for v in record["embedding"].values()),
            f"rank {rank}: compact row-owner coalescing",
        )
        experts = list(record["experts"].values())
        check(
            len(experts) == 7
            and all(
                g["slot"] == ("torch.bfloat16" if ep == 4 else "torch.float32")
                and g["compact_owner"] == (ep == 4)
                and g["fp8_generation"] >= 0
                and g["cache_ready"]
                and g["retain_tail_weights"]
                for g in experts
            ),
            f"rank {rank}: expert slots/cache",
        )
        check(
            record["mesh"]["efsdp"] == list(range(rank % ep, 8, ep)),
            "expert process group",
        )
        check(record["dense_fp8_tail_layers"], f"rank {rank}: eligible FP8 tail layers")
        tail_inputs = record["dense_tail_quantizer_inputs"]
        for shape in ("[10944, 4096]", "[4096, 10944]"):
            check(
                tail_inputs.get(f"cudacherry::quantize_blk128x128 {shape}", 0) > 0,
                f"rank {rank}: direct dense tail quantization {shape}",
            )
        (stdout,) = (attempt / "logs").glob(f"*/attempt_0/{rank}/stdout.log")
        logs = re.sub(r"\x1b\[[0-9;]*m", "", stdout.read_text())
        expected_log = [
            "LM_HEAD_EXACT_FOLDED_CE_ACTIVE",
            "LM_HEAD_MATRIX_DW_WRITER",
            "LM_HEAD_DIRECT_BF16_DHIDDEN_ACTIVE",
            "LM_HEAD_MATRIX_FUSED_Z_LOSS_ACTIVE",
            "GFSDP buffer region transition active",
            "dedicated NCCL communicator active max_ctas=4",
            "GFSDP FFN overlap active: layers.0.feed_forward",
            "GFSDP FFN overlap active: mtp.layers.0.block.feed_forward",
        ]
        for marker in expected_log:
            check(marker in logs, f"rank {rank}: missing {marker}")
        folded_rows = 4 * 4096 // loss_chunks
        writer_plan = "9.14" if loss_chunks == 2 else "5.5"
        check(
            f"backend=inductor rows={folded_rows} vocab=128256" in logs
            and f"chunks={loss_chunks} writer=cublasLtMatmul pair={writer_plan}" in logs,
            f"rank {rank}: exact CE shape or staged head dW writer",
        )
        if ep == 4:
            check(
                "GFSDP compact BF16 owner transport active" in logs,
                "compact owner initialization",
            )
        if rank == 0:
            check("LM_HEAD_MATRIX_DW_REDUCED" in logs, "staged head dW reduction")
        metrics = re.search(r"step:\s*1\s+loss:\s*(\S+)\s+grad_norm:\s*(\S+)", logs)
        check(
            metrics and all(math.isfinite(float(x)) for x in metrics.groups()),
            f"rank {rank}: update metrics",
        )
        events = record["events"]
        routes = {}
        for marker in (
            "CPU torchtitan::gdn_fla_backward",
            "CUDA _gdn_short_conv_backward_kernel",
            "CUDA _gdn_output_norm_quantize_128_kernel",
            "CUDA _gdn_output_norm_backward_kernel",
            "CUDA _mla_rope_pack",
            "CUDA _mla_rope_unpack",
            "CUDA _capacity_mask_kernel",
            "CPU torchtitan::bind_prepared_fp8_weight",
            "CPU torchtitan::write_direct_expert_dw",
            "CPU torchtitan::cute_gated_norm_pair",
            "CPU torchtitan::cute_gated_norm_bwd_skip",
            "CPU torchtitan::cute_gated_norm_residual_fwd",
            "CPU torchtitan::cute_gated_norm_fwd_stats",
            "CPU torchtitan::exact_folded_ce_consume",
        ):
            count = events.get(marker, 0)
            check(count, f"rank {rank}: missing sampled {marker}")
            routes[marker] = count
        check(
            routes["CPU torchtitan::write_direct_expert_dw"] == 2 * len(experts),
            f"rank {rank}: expected two direct writes per expert group",
        )
        rounded_writes = sum(
            n for name, n in events.items()
            if name.startswith("CUDA void deep_gemm::sm90_k_grouped_fp8_gemm_rounded_f32_impl")
        )
        owner_writes = events.get("CUDA _write_owner", 0)
        expected = 2 * len(experts)
        check(
            (rounded_writes, owner_writes)
            == ((expected, 0) if ep == 8 else (0, expected)),
            f"rank {rank}: unexpected direct expert dW kernels for EP{ep}",
        )
        routes["CUDA rounded FP32 expert slot"] = rounded_writes
        routes["CUDA _write_owner"] = owner_writes
        result["ranks"][rank] = {
            "routes": routes,
            "dense_tail_quantizer_inputs": tail_inputs,
            "queue": queue,
            "expert_groups": len(experts),
            "slot": experts[0]["slot"],
            "loss": float(metrics.group(1)),
            "grad_norm": float(metrics.group(2)),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--ep", type=int, choices=(8, 4), required=True)
    parser.add_argument("--loss-chunks", type=int, choices=(2, 8, 32), default=2)
    args = parser.parse_args()
    print(json.dumps(verify(args.attempt, args.ep, args.loss_chunks), indent=2))
