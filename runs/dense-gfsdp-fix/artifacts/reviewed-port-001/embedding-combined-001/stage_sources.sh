#!/usr/bin/env bash
set -euo pipefail
ws=/workspace-SR008.nfs2/vkropotin
staging=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
git_dir="$ws/optimization-task"
head=e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f
cd "$staging"
sha256sum --check bundle.sha256
for name in dg-n4 dg-n8 dg-n4c; do
  test ! -e "$ws/$name"
  git -C "$git_dir" worktree add --quiet --detach "$ws/$name" "$head"
  git -C "$ws/$name" apply --check "$staging/$name.patch"
  git -C "$ws/$name" apply "$staging/$name.patch"
  git -C "$ws/$name" diff --check -- gift/
  (cd "$ws/$name"; raw=runs/overnight-streaming-perf/artifacts/raw; git check-ignore -q -- "$raw/"; mkdir -p -- "$raw"; git check-ignore -q -- "$raw")
done
/opt/torchtitan-venv/bin/python3.12 -B - "$ws" <<'PY'
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import hashlib, json, sys

ws = Path(sys.argv[1])
expected = {"dg-n4": "5fe3b16adb1bb165c281f1b292570097904e5908363497de3b39a07787f9b1b3", "dg-n8": "cb2cb7f319abd06b74946cf62b2f7480217081639605b38257a82b4829b58af4", "dg-n4c": "be51c5be2791a042dbda2dfb8df015090e004634b777fa50026eb5331de429b6"}
records = {}
for name, digest in expected.items():
    root = ws / name
    spec = spec_from_file_location("_owner_snapshot", root / "runs/overnight-streaming-perf/run.py")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    snapshot = module.source_snapshot(root)
    assert snapshot["tracked_and_untracked_sha256"] == digest, (name, snapshot)
    owner = root / "gift/torchtitan/distributed/gfsdp/row_owner_embedding.py"
    snapshot["owner_sha256"] = hashlib.sha256(owner.read_bytes()).hexdigest()
    records[name] = snapshot
    print(name, digest, snapshot["owner_sha256"], flush=True)
with Path("source-pins.json").open("x") as out:
    json.dump(records, out, indent=2)
    out.write("\n")
PY
sha256sum --check queue-files.sha256
cat queue-files.sha256 > sources.sha256
sha256sum source-pins.json >> sources.sha256
bash -n queue_embedding_combined.sh
/opt/torchtitan-venv/bin/python3.12 -B narrow_native_acceptance.py --help >/dev/null
/opt/torchtitan-venv/bin/python3.12 -B narrow_owner_exact.py run --help >/dev/null
