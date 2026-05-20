"""Download a specific shard range from karpathy/climbmix-400b-shuffle.

Useful for expanding the training corpus in batches without re-downloading
shards we already have. By contrast, `python -m nanochat.dataset -n N`
always starts from shard_00000 and stops at N-1.

Usage (set NANOCHAT_DATA_DIR to the destination directory):

  NANOCHAT_DATA_DIR=$HOME/.cache/nanochat/base_data_climbmix_extra3 \
  http_proxy=http://sys-proxy-rd-relay.byted.org:8118 \
  https_proxy=http://sys-proxy-rd-relay.byted.org:8118 \
  python -m scripts.scaling.download_range --start 101 --end 201 --workers 4

End is exclusive (Python range semantics). The example above downloads
shards 00101 through 00200 (100 shards).

The validation shard (MAX_SHARD = 6542) is NOT downloaded automatically
by this script — use `python -m nanochat.dataset -n 1` for that.
"""
import os
import sys
import argparse
from multiprocessing import Pool

from nanochat.dataset import download_single_file, DATA_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True, help="first shard idx (inclusive)")
    p.add_argument("--end", type=int, required=True, help="last shard idx (exclusive)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    ids = list(range(args.start, args.end))
    print(f"DATA_DIR={DATA_DIR}")
    print(f"Downloading shards {args.start}..{args.end-1} ({len(ids)} shards) with {args.workers} workers")

    with Pool(processes=args.workers) as pool:
        results = pool.map(download_single_file, ids)

    ok = sum(1 for r in results if r)
    failed = [i for i, r in zip(ids, results) if not r]
    print(f"\n{ok}/{len(results)} succeeded")
    if failed:
        print(f"FAILED shards: {failed}")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
