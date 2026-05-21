"""Download a code-dataset shard pack for v3 Run A' (3e18 / d=12 / code-mix).

Streams an ungated, public code dataset (default
`codeparrot/github-code-clean`) and repackages it into parquet shards
matching the ClimbMix shard layout — one column `text` with row-group
size 1000. The output drops into `$NANOCHAT_DATA_DIR` so existing
dataloaders pick it up.

Each output shard targets ~25 MB on disk (matches a ClimbMix shard) so
that downstream byte/token accounting stays sane.

Default dataset chain (first match wins, all ungated):
  1. codeparrot/github-code-clean  (Apache-2.0, parquet, multi-language
     when `language` is None / "all", else a single language)
  2. HuggingFaceTB/smollm-corpus    (ODC-BY-1.0, the actual code subset
     SmolLM2 used; pass --dataset HuggingFaceTB/smollm-corpus
     --config python-edu)

Usage:

  NANOCHAT_DATA_DIR=$HOME/.cache/nanochat/base_data_code_raw \
  http_proxy=http://sys-proxy-rd-relay.byted.org:8118 \
  https_proxy=http://sys-proxy-rd-relay.byted.org:8118 \
  python -m scripts.scaling.download_code --num-shards 40

To use SmolLM2's python-edu subset instead:

  python -m scripts.scaling.download_code \
      --dataset HuggingFaceTB/smollm-corpus --config python-edu \
      --text-column text --num-shards 40
"""
import os
import sys
import argparse

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nanochat.dataset import DATA_DIR  # noqa: E402


DEFAULT_ROWS_PER_GROUP = 1000
DEFAULT_SHARD_TARGET_MB = 25


def stream_dataset(dataset, config, split, text_column):
    """Yield raw text docs from a HF dataset (streaming)."""
    from datasets import load_dataset
    kwargs = dict(split=split, streaming=True)
    if config:
        kwargs["name"] = config
    ds = load_dataset(dataset, **kwargs)
    for row in ds:
        text = row.get(text_column)
        if text:
            yield text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="codeparrot/github-code-clean",
                   help="HF dataset id (must be ungated and public)")
    p.add_argument("--config", default="all-all",
                   help="HF dataset config / subset name (default: all-all)")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="code",
                   help="column to extract text from (codeparrot uses `code`, others `text`)")
    p.add_argument("--num-shards", type=int, default=40)
    p.add_argument("--shard-target-mb", type=int, default=DEFAULT_SHARD_TARGET_MB)
    p.add_argument("--rows-per-group", type=int, default=DEFAULT_ROWS_PER_GROUP)
    p.add_argument("--prefix", default="shard_code",
                   help="shard filename prefix (default: shard_code)")
    p.add_argument("--skip-docs", type=int, default=0,
                   help="skip the first N docs of the stream (use to fetch fresh content on re-runs)")
    p.add_argument("--start-idx", type=int, default=0,
                   help="output shard index to start at (default 0, raises so we do not overwrite)")
    args = p.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"DATA_DIR     = {DATA_DIR}", flush=True)
    print(f"dataset      = {args.dataset} (config={args.config}, split={args.split})", flush=True)
    print(f"text_column  = {args.text_column}", flush=True)
    print(f"target       = {args.num_shards} shards × ~{args.shard_target_mb} MB", flush=True)

    target_bytes = args.shard_target_mb * 1_000_000

    shard_idx = args.start_idx
    stop_at = args.start_idx + args.num_shards
    buf_docs = []
    buf_bytes = 0

    def flush(idx, docs):
        out = os.path.join(DATA_DIR, f"{args.prefix}_{idx:05d}.parquet")
        tbl = pa.table({"text": pa.array(docs, type=pa.large_string())})
        pq.write_table(tbl, out, compression="zstd",
                       row_group_size=args.rows_per_group)
        print(f"  -> {out} docs={len(docs):,} on_disk={os.path.getsize(out)/1e6:.1f} MB",
              flush=True)

    n_emitted = 0
    n_skipped = 0
    for doc in stream_dataset(args.dataset, args.config, args.split, args.text_column):
        if n_skipped < args.skip_docs:
            n_skipped += 1
            if n_skipped % 50_000 == 0:
                print(f"  skip cursor at {n_skipped:,} / {args.skip_docs:,}", flush=True)
            continue
        if shard_idx >= stop_at:
            break
        buf_docs.append(doc)
        buf_bytes += len(doc.encode("utf-8", errors="replace"))
        n_emitted += 1
        if buf_bytes >= target_bytes:
            flush(shard_idx, buf_docs)
            shard_idx += 1
            buf_docs = []
            buf_bytes = 0

    if buf_docs and shard_idx < stop_at:
        flush(shard_idx, buf_docs)
        shard_idx += 1

    print(f"\nWrote {shard_idx - args.start_idx} shards (idx {args.start_idx}..{shard_idx-1}), "
          f"docs streamed = {n_emitted:,}, docs skipped = {n_skipped:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
