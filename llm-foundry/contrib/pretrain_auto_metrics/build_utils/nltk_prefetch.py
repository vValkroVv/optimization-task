import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import nltk


def download_resource(resource: str, retries: int) -> None:
    for _ in range(retries + 1):
        if nltk.download(resource):
            return
        time.sleep(2)
    raise RuntimeError(f"nltk download failed for: {resource}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resources", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")

    resources = [r.strip() for r in args.resources.split(",") if r.strip()]
    if not resources:
        return

    with ThreadPoolExecutor(max_workers=min(args.jobs, len(resources))) as ex:
        futs = [
            ex.submit(download_resource, resource, args.retries)
            for resource in resources
        ]
        for f in as_completed(futs):
            f.result()


if __name__ == "__main__":
    main()
