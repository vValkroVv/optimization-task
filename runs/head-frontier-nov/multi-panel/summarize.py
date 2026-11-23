"""Check the native GIFT timing receipts, panel activation and full numerical oracles."""
import json
from pathlib import Path
from statistics import median


def load(path):
    return json.loads(path.read_text())


def main():
    base = Path(__file__).resolve().parent
    summary = {}
    for c in (2, 8):
        oracle = load(base/f"c{c}-oracle.json")
        numeric = {}
        for width, entry in oracle["candidates"].items():
            mismatch = {
                key:sum(item[key][str(start)]["different_elements"]
                        for chunk in oracle["per_chunk"]
                        for item in [chunk["split_checks"][width]]
                        for start in range(0,128256,int(width)))
                for key in ("projection","bf16_dlogits")
            }
            assert not any(entry["vs_control"][k]["nonfinite"] for k in entry["vs_control"])
            numeric[width] = {
                "mismatched_logit_and_gradient_elements":mismatch,
                "equal_outputs":{k:v["different_elements"]==0 for k,v in entry["vs_control"].items()},
                "dhidden_rel_l2_to_fp64":entry["dx_fp64"]["relative_l2"],
                "dhidden_rel_l2_to_rounded_fp64":entry["dx_rounded_fp64"]["relative_l2"],
                "dw_rel_l2_to_fp64":entry["dw_fp64"]["relative_l2"],
                "dx_component_integrated_differences":entry["standalone_bf16_dx_vs_integrated"]["different_elements"]
            }
        summary[f"precision-c{c}"] = numeric
        for suffix in ("","-alias"):
            group={}
            gpu=set()
            for label,panels in (("A",1),("B",4),("D",8)):
                runs=[load(base/f"c{c}-{label}{i}{suffix}.json") for i in range(2)]
                for r in runs:
                    gpu.add(r["device"]["visible"])
                    assert r["chunks"]==c and len(r["elapsed_ms"])==r["warmup"]==10
                    assert r["median_ms"]==median(r["elapsed_ms"])
                    assert r["peak_total_gpu_gib"]==max(r["total_gpu_gib"])
                    assert r["gather_probe"]["mode"]==("alias" if suffix else "stack")
                    assert r["panel"]["active"]==(panels>1)
                    calls=r["gather_probe"]["native_all_gather_into_tensor_calls"]
                    assert r["dw_writer"]["calls"]==(calls-1)*panels*c
                    assert r["panel"]["projection_calls_per_microbatch"]==(2*panels-1)*c
                    assert not any(r["allocator_retries_and_ooms"].values())
                    assert r["activation"]["staged_chunk_calls"]==[True]*c
                group[str(panels)]={
                    "process_medians_ms":[r["median_ms"] for r in runs],
                    "median_ms":median(r["median_ms"] for r in runs),
                    "peak_total_gib":max(r["peak_total_gpu_gib"] for r in runs),
                    **{f"peak_{metric}_gib":max(m[f"{metric}_peak_bytes"] for r in runs for m in r["memory"])/2**30
                       for metric in ["allocated","reserved"]}
                }
            assert len(gpu)==1
            summary[f"c{c}{suffix}"]=group
    (base/"results.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    main()
