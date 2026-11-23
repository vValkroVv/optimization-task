"""Verify fixed-chunk isolated replay receipts and aggregate unprofiled process medians."""
import json
from pathlib import Path
from statistics import median


def load(file):
    return json.loads(file.read_text())


def summarize(root, chunks):
    arms = {}
    digests = {}
    batches = {}
    versions = set()
    for name in ("control-a", "control-b", "pair-a", "pair-b"):
        path = root / f"c{chunks}-{name}.json"
        run = load(path)
        digest = load(path.with_suffix(".digest.json"))
        assert set(digest) == {"dw", "dx", "loss", "z_loss", "z_dx"}
        assert run["chunks"] == chunks and run["warmup"] == len(run["elapsed_ms"]) == 10
        assert median(run["elapsed_ms"]) == run["median_ms"]
        assert max(run["total_gpu_gib"]) == run["peak_total_gpu_gib"]
        assert run["native_inplace_ce_chunks"] == chunks and run["expandable_segments"]
        assert run["activation"]["staged_chunk_calls"] == [True] * chunks
        assert run["gather"]["mode"] == "stack" and run["gather"]["native_all_gather_into_tensor_calls"] == 24
        assert not any(run["allocator_retries_and_ooms"].values())
        assert run["writer"]["beta_calls"] == {"0": 23, "1": 23*(chunks-1)}
        if name.startswith("pair"):
            assert run["writer"]["extra"] == {"stores":23*chunks//2,"pending":False}
        else:
            assert run["writer"]["pair"] == {2:"9.14",8:"5.5"}[chunks]
        versions.add((run["source"],run["device"]["visible"],run["harness_sha256"],
                      tuple(sorted(run["source_hashes"].items()))))
        digests[name] = digest
        batches[name] = run
    assert len(versions) == 1
    for kind in ("control", "pair"):
        arms[kind] = {}
        runs = [batches[f"{kind}-{s}"] for s in ("a","b")]
        arms[kind]["process_medians_ms"] = [r["median_ms"] for r in runs]
        arms[kind]["median_ms"] = median(arms[kind]["process_medians_ms"])
        arms[kind]["peak_total_gib"] = max(r["peak_total_gpu_gib"] for r in runs)
        for measure in ("allocated", "reserved"):
            arms[kind][f"peak_{measure}_gib"] = max(
                sample[f"{measure}_peak_bytes"] for r in runs for sample in r["memory"])/2**30
        assert digests[kind+"-a"] == digests[kind+"-b"]
    arms["bitwise_equal"] = {
        k:digests["control-a"][k] == digests["pair-a"][k] for k in digests["control-a"]
    }
    return arms


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    summary = {str(c):summarize(root,c) for c in (2,8)}
    (root/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary, indent=2))
