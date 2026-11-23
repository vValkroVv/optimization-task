"""Measure early release of chunk logits in the existing compiled replay."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

import replay


def install_early_release():
    from torchtitan.components import loss

    method = loss.ChunkedCELoss._chunked_loss_and_grad
    source_file = inspect.getsourcefile(method)
    source_lines, first_line = inspect.getsourcelines(method)
    source = textwrap.dedent("".join(source_lines))
    anchor = "                h_chunk.grad = None\n"
    assert source.count(anchor) == 1, source_file
    source = source.replace(anchor, anchor + "            del logits\n")
    code = compile("\n" * (first_line - 1) + source, source_file, "exec")
    methods = {}
    exec(code, vars(loss), methods)
    loss.ChunkedCELoss._chunked_loss_and_grad = methods[method.__name__]


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("lifetime", choices=("native", "release", "reuse", "combined"))
    args, remainder = parser.parse_known_args()
    assert remainder[0] == "current"
    output = Path(remainder[1])
    sys.argv[1:] = remainder
    release = args.lifetime in ("release", "combined")
    if args.lifetime in ("reuse", "combined"):
        sys.argv.extend(("--storage", "reuse"))
    make_head = replay.make_head

    def make_experimental_head(weight):
        if release:
            install_early_release()
        return make_head(weight)

    if release:
        replay.make_head = make_experimental_head

    if "--dump" in remainder:
        dump_path = Path(remainder[remainder.index("--dump") + 1])
        native_save = replay.torch.save

        def save_digest(tensors, destination, *args, **kwargs):
            if destination != dump_path:
                return native_save(tensors, destination, *args, **kwargs)
            digest = {}
            for name, tensor in tensors.items():
                flat = tensor.reshape(-1).contiguous()
                buffer = flat.view(replay.torch.uint8).numpy()
                digest[name] = {
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": hashlib.sha256(buffer).hexdigest(),
                }
            dump_path.write_text(json.dumps(digest, indent=2) + "\n")

        replay.torch.save = save_digest

    replay.main()
    receipt = json.loads(output.read_text())
    receipt["lifetime"] = args.lifetime
    receipt["early_release_gift_loop"] = release
    receipt["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
