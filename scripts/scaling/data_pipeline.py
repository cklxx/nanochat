"""
Data pipeline for minimum-scale scaling law validation on V100.

Stages:
  1. inventory  - list local parquet shards and report basic file stats
  2. stats      - read each shard, compute doc/char/byte/word stats,
                  language-ish heuristic, length percentiles
  3. clean      - apply filters (length, non-text ratio, near-duplicate
                  hash on doc prefix) and write cleaned shards to a new dir
  4. validate   - re-run stats on the cleaned shards, compare, and
                  emit a JSON report + a few small CSV/PNG figures

Usage:
    python -m scripts.scaling.data_pipeline inventory
    python -m scripts.scaling.data_pipeline stats   --split train
    python -m scripts.scaling.data_pipeline clean   --in base_data_climbmix --out base_data_clean
    python -m scripts.scaling.data_pipeline validate

The cleaned shards stay parquet-compatible with the rest of nanochat
(they have a `text` column with row groups of size 1000), so existing
dataloaders pick them up via NANOCHAT_BASE_DIR.

All figures and CSVs go under $NANOCHAT_BASE_DIR/scaling_v100/data_report/
so they can be archived for the paper.
"""

import os
import io
import sys
import json
import math
import time
import hashlib
import argparse
import statistics
from collections import Counter
from contextlib import contextmanager
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

# nanochat helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nanochat.common import get_base_dir  # noqa: E402


BASE_DIR = get_base_dir()
DEFAULT_RAW_DIR = os.path.join(BASE_DIR, "base_data_climbmix")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "base_data_clean")
REPORT_DIR = os.path.join(BASE_DIR, "scaling_v100", "data_report")
os.makedirs(REPORT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Filtering rules. Tuned for ClimbMix (already moderately clean) — we are
# mainly guarding against absurd outliers and exact-prefix duplicates.

MIN_CHARS = 200            # drop very short docs (often boilerplate / 404 pages)
MAX_CHARS = 200_000        # drop pathologically long docs (a few MB each)
MIN_AVG_WORD_LEN = 2.0     # filter sequences that look like keysmash
MAX_AVG_WORD_LEN = 20.0    # filter docs that are one giant token
MIN_LETTER_RATIO = 0.6     # at least 60% of chars are letters/space/punct
PREFIX_HASH_LEN = 256      # prefix length used for near-duplicate detection


# -----------------------------------------------------------------------------
# helpers

def list_parquets(data_dir):
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )


def iter_docs(parquet_path, batch_size=1000):
    pf = pq.ParquetFile(parquet_path)
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx, columns=["text"])
        for t in rg.column("text").to_pylist():
            yield t


def letter_ratio(s):
    if not s:
        return 0.0
    n_letters = sum(1 for c in s if c.isalpha() or c.isspace() or c in ".,;:!?'\"-")
    return n_letters / len(s)


def avg_word_len(s):
    words = s.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


@contextmanager
def stopwatch(label):
    t0 = time.time()
    yield
    print(f"[{label}] {time.time() - t0:.2f}s", flush=True)


# -----------------------------------------------------------------------------
# subcommands

def cmd_inventory(args):
    paths = list_parquets(args.dir)
    rows = []
    total_bytes = 0
    for p in paths:
        st = os.stat(p)
        pf = pq.ParquetFile(p)
        rows.append({
            "file": os.path.basename(p),
            "size_mb": round(st.st_size / 1e6, 2),
            "row_groups": pf.num_row_groups,
            "rows": pf.metadata.num_rows,
        })
        total_bytes += st.st_size
    out = {
        "dir": args.dir,
        "n_shards": len(paths),
        "total_size_mb": round(total_bytes / 1e6, 2),
        "shards": rows,
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(REPORT_DIR, "inventory.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def _shard_stats(path, sample_cap=None):
    """Compute per-shard text statistics. sample_cap caps the doc count to keep this fast."""
    n_docs = 0
    n_chars = 0
    n_words = 0
    lens_char = []
    lens_word = []
    letter_ratios = []
    hash_counts = Counter()
    for doc in iter_docs(path):
        if sample_cap is not None and n_docs >= sample_cap:
            break
        L = len(doc)
        W = len(doc.split())
        n_docs += 1
        n_chars += L
        n_words += W
        lens_char.append(L)
        lens_word.append(W)
        letter_ratios.append(letter_ratio(doc))
        prefix = doc[:PREFIX_HASH_LEN]
        h = hashlib.md5(prefix.encode("utf-8", errors="replace")).hexdigest()
        hash_counts[h] += 1
    dups = sum(c - 1 for c in hash_counts.values() if c > 1)
    return {
        "file": os.path.basename(path),
        "n_docs": n_docs,
        "n_chars": n_chars,
        "n_words": n_words,
        "char_len_p50": percentile(lens_char, 50),
        "char_len_p95": percentile(lens_char, 95),
        "char_len_p99": percentile(lens_char, 99),
        "word_len_p50": percentile(lens_word, 50),
        "letter_ratio_mean": statistics.mean(letter_ratios) if letter_ratios else 0.0,
        "prefix_duplicates": dups,
        "prefix_dup_rate": dups / max(n_docs, 1),
    }


def cmd_stats(args):
    paths = list_parquets(args.dir)
    if args.split == "train":
        paths = paths[:-1]
    elif args.split == "val":
        paths = paths[-1:]
    if args.max_shards:
        paths = paths[: args.max_shards]
    rows = []
    for p in paths:
        with stopwatch(f"stats {os.path.basename(p)}"):
            rows.append(_shard_stats(p, sample_cap=args.sample_cap))
    agg = {
        "n_docs": sum(r["n_docs"] for r in rows),
        "n_chars": sum(r["n_chars"] for r in rows),
        "n_words": sum(r["n_words"] for r in rows),
        "n_shards": len(rows),
        "letter_ratio_mean": statistics.mean(r["letter_ratio_mean"] for r in rows) if rows else 0.0,
        "prefix_dup_rate": (
            sum(r["prefix_duplicates"] for r in rows) / max(sum(r["n_docs"] for r in rows), 1)
        ),
    }
    out = {"dir": args.dir, "split": args.split, "aggregate": agg, "per_shard": rows}
    name = f"stats_{args.split}_{os.path.basename(args.dir)}.json"
    path = os.path.join(REPORT_DIR, name)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    # CSV mirror for paper-grade plots later
    import csv
    csv_path = os.path.join(REPORT_DIR, name.replace(".json", ".csv"))
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    return out


def _doc_passes(doc):
    """Return (ok, reason)."""
    if not doc:
        return False, "empty"
    L = len(doc)
    if L < MIN_CHARS:
        return False, "too_short"
    if L > MAX_CHARS:
        return False, "too_long"
    awl = avg_word_len(doc)
    if awl < MIN_AVG_WORD_LEN:
        return False, "awl_low"
    if awl > MAX_AVG_WORD_LEN:
        return False, "awl_high"
    if letter_ratio(doc) < MIN_LETTER_RATIO:
        return False, "letter_ratio_low"
    return True, "ok"


# Code-mode filters. Code rejects prose heuristics (lots of {};=, single-char
# tokens, etc.), so we keep only length sanity + dedupe. MIN/MAX flipped
# slightly to allow short snippets and long files.
CODE_MIN_CHARS = 100
CODE_MAX_CHARS = 400_000


def _code_doc_passes(doc):
    """Code-aware doc filter — drops only obvious garbage."""
    if not doc:
        return False, "empty"
    L = len(doc)
    if L < CODE_MIN_CHARS:
        return False, "too_short"
    if L > CODE_MAX_CHARS:
        return False, "too_long"
    return True, "ok"


def _clean_one_shard(task):
    """Single-shard cleaning worker — used by both serial and parallel paths.

    With shared_hashes=None, dedup is intra-shard only (safe for parallelism).
    """
    in_path, out_dir, code_mode, rows_per_group = task
    passes_fn = _code_doc_passes if code_mode else _doc_passes
    local_counter = Counter()
    kept_docs = []
    local_hashes = set()
    for doc in iter_docs(in_path):
        ok, reason = passes_fn(doc)
        local_counter[reason] += 1
        if not ok:
            continue
        h = hashlib.md5(doc[:PREFIX_HASH_LEN].encode("utf-8", errors="replace")).hexdigest()
        if h in local_hashes:
            local_counter["dup_prefix"] += 1
            continue
        local_hashes.add(h)
        kept_docs.append(doc)
    out_name = os.path.basename(in_path)
    out_path = os.path.join(out_dir, out_name)
    tbl = pa.table({"text": pa.array(kept_docs, type=pa.large_string())})
    pq.write_table(tbl, out_path, compression="zstd", row_group_size=rows_per_group)
    return {
        "file": out_name,
        "kept": len(kept_docs),
        "counter": dict(local_counter),
    }


def cmd_clean(args):
    paths = list_parquets(args.in_dir)
    if not paths:
        print(f"no shards in {args.in_dir}", file=sys.stderr)
        sys.exit(1)
    if args.max_shards:
        paths = paths[: args.max_shards]
    os.makedirs(args.out_dir, exist_ok=True)

    passes_fn = _code_doc_passes if args.code else _doc_passes
    mode_label = "code" if args.code else "prose"

    global_counter = Counter()
    schema = pa.schema([("text", pa.large_string())])
    rg_target = args.rows_per_group

    summary_per_shard = []

    if args.workers and args.workers > 1:
        # Parallel path: intra-shard dedup only (each worker has its own hash set).
        # Good for prose at small dup rates; for code prefer --workers 1 to keep
        # global dedup.
        tasks = [(p, args.out_dir, args.code, rg_target) for p in paths]
        print(f"[parallel clean[{mode_label}] workers={args.workers} shards={len(tasks)}] "
              "(intra-shard dedup only)", flush=True)
        t0 = time.time()
        with Pool(processes=args.workers) as pool:
            for res in pool.imap_unordered(_clean_one_shard, tasks):
                global_counter.update(res["counter"])
                summary_per_shard.append({
                    "file": res["file"],
                    "in_docs": sum(res["counter"].values()),
                    "kept": res["kept"],
                    "rejected": res["counter"],
                })
                done = len(summary_per_shard)
                if done % 25 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    eta = elapsed * (len(tasks) - done) / max(done, 1)
                    print(f"  [{done:>4}/{len(tasks)}] {res['file']} kept={res['kept']:,}  "
                          f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    else:
        seen_hashes = set()
        for p in paths:
            kept_docs = []
            local_counter = Counter()
            with stopwatch(f"clean[{mode_label}] {os.path.basename(p)}"):
                for doc in iter_docs(p):
                    ok, reason = passes_fn(doc)
                    local_counter[reason] += 1
                    if not ok:
                        continue
                    h = hashlib.md5(doc[:PREFIX_HASH_LEN].encode("utf-8", errors="replace")).hexdigest()
                    if h in seen_hashes:
                        local_counter["dup_prefix"] += 1
                        continue
                    seen_hashes.add(h)
                    kept_docs.append(doc)
                out_name = os.path.basename(p)
                out_path = os.path.join(args.out_dir, out_name)
                tbl = pa.table({"text": pa.array(kept_docs, type=pa.large_string())})
                pq.write_table(
                    tbl, out_path,
                    compression="zstd",
                    row_group_size=rg_target,
                )
            global_counter.update(local_counter)
            summary_per_shard.append({
                "file": out_name,
                "in_docs": sum(local_counter.values()),
                "kept": local_counter["ok"] - local_counter.get("dup_prefix", 0),
                "rejected": dict(local_counter),
            })
            print(f"  -> {out_path} kept={local_counter['ok']} (dup_prefix={local_counter['dup_prefix']})")

    if args.code:
        filters_used = {
            "mode": "code",
            "CODE_MIN_CHARS": CODE_MIN_CHARS,
            "CODE_MAX_CHARS": CODE_MAX_CHARS,
            "PREFIX_HASH_LEN": PREFIX_HASH_LEN,
        }
        summary_name = "clean_summary_code.json"
    else:
        filters_used = {
            "mode": "prose",
            "MIN_CHARS": MIN_CHARS, "MAX_CHARS": MAX_CHARS,
            "MIN_AVG_WORD_LEN": MIN_AVG_WORD_LEN, "MAX_AVG_WORD_LEN": MAX_AVG_WORD_LEN,
            "MIN_LETTER_RATIO": MIN_LETTER_RATIO,
            "PREFIX_HASH_LEN": PREFIX_HASH_LEN,
        }
        summary_name = "clean_summary.json"
    summary = {
        "in_dir": args.in_dir,
        "out_dir": args.out_dir,
        "filters": filters_used,
        "totals": dict(global_counter),
        "per_shard": summary_per_shard,
    }
    path = os.path.join(REPORT_DIR, summary_name)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["totals"], indent=2))
    print(f"\nwrote {path}")


def cmd_validate(args):
    """Re-run stats on cleaned dir and compare to raw."""
    raw_stats_path = os.path.join(REPORT_DIR, f"stats_train_{os.path.basename(args.raw_dir)}.json")
    if not os.path.exists(raw_stats_path):
        print(f"missing raw stats at {raw_stats_path}; run `stats --dir {args.raw_dir}` first",
              file=sys.stderr)
        sys.exit(1)
    raw = json.load(open(raw_stats_path))

    # Run stats on cleaned dir
    args2 = argparse.Namespace(dir=args.clean_dir, split="train",
                               max_shards=args.max_shards, sample_cap=None)
    cleaned = cmd_stats(args2)

    delta = {
        "raw_docs": raw["aggregate"]["n_docs"],
        "clean_docs": cleaned["aggregate"]["n_docs"],
        "kept_fraction": cleaned["aggregate"]["n_docs"] / max(raw["aggregate"]["n_docs"], 1),
        "raw_chars": raw["aggregate"]["n_chars"],
        "clean_chars": cleaned["aggregate"]["n_chars"],
        "char_kept_fraction": cleaned["aggregate"]["n_chars"] / max(raw["aggregate"]["n_chars"], 1),
        "letter_ratio_raw": raw["aggregate"]["letter_ratio_mean"],
        "letter_ratio_clean": cleaned["aggregate"]["letter_ratio_mean"],
        "prefix_dup_rate_raw": raw["aggregate"]["prefix_dup_rate"],
        "prefix_dup_rate_clean": cleaned["aggregate"]["prefix_dup_rate"],
    }
    path = os.path.join(REPORT_DIR, "validate_summary.json")
    with open(path, "w") as f:
        json.dump(delta, f, indent=2)
    print(json.dumps(delta, indent=2))
    print(f"\nwrote {path}")

    # Optional: histograms — only if matplotlib is installed
    try:
        _plot_doc_length_histogram(args.raw_dir, args.clean_dir, args.max_shards)
    except Exception as e:
        print(f"(plot skipped: {e})")


def _plot_doc_length_histogram(raw_dir, clean_dir, max_shards):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def collect(d):
        lens = []
        for p in list_parquets(d)[: max_shards or None]:
            for doc in iter_docs(p):
                lens.append(len(doc))
        return lens

    raw = collect(raw_dir)
    clean = collect(clean_dir)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    bins = [10 ** x for x in [i / 4 for i in range(0, 25)]]
    ax.hist(raw, bins=bins, alpha=0.45, label=f"raw (n={len(raw):,})")
    ax.hist(clean, bins=bins, alpha=0.65, label=f"clean (n={len(clean):,})")
    ax.set_xscale("log")
    ax.set_xlabel("document length (chars, log scale)")
    ax.set_ylabel("count")
    ax.set_title("ClimbMix shard — doc length distribution, raw vs cleaned")
    ax.legend()
    ax.grid(True, alpha=0.3)
    png = os.path.join(REPORT_DIR, "doc_length_hist.png")
    pdf = os.path.join(REPORT_DIR, "doc_length_hist.pdf")
    fig.tight_layout()
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f"wrote {png} and {pdf}")


# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Scaling-law data pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inventory")
    pi.add_argument("--dir", default=DEFAULT_RAW_DIR)
    pi.set_defaults(func=cmd_inventory)

    ps = sub.add_parser("stats")
    ps.add_argument("--dir", default=DEFAULT_RAW_DIR)
    ps.add_argument("--split", choices=["all", "train", "val"], default="all")
    ps.add_argument("--max-shards", type=int, default=0,
                    help="only inspect this many shards (0 = all)")
    ps.add_argument("--sample-cap", type=int, default=0,
                    help="cap docs per shard for fast stats (0 = all)")
    ps.set_defaults(func=lambda a: cmd_stats(_norm(a)))

    pc = sub.add_parser("clean")
    pc.add_argument("--in", dest="in_dir", default=DEFAULT_RAW_DIR)
    pc.add_argument("--out", dest="out_dir", default=DEFAULT_OUT_DIR)
    pc.add_argument("--max-shards", type=int, default=0)
    pc.add_argument("--rows-per-group", type=int, default=1000)
    pc.add_argument("--code", action="store_true",
                    help="apply code-aware filters (skip prose heuristics)")
    pc.add_argument("--workers", type=int, default=1,
                    help="parallel workers (>1 disables cross-shard dedup; "
                         "good for prose, prefer 1 for code)")
    pc.set_defaults(func=lambda a: cmd_clean(_norm(a)))

    pv = sub.add_parser("validate")
    pv.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    pv.add_argument("--clean-dir", default=DEFAULT_OUT_DIR)
    pv.add_argument("--max-shards", type=int, default=0)
    pv.set_defaults(func=cmd_validate)

    args = p.parse_args()
    args.func(args)


def _norm(a):
    # turn 0 → None for "no cap"
    if hasattr(a, "max_shards") and a.max_shards == 0:
        a.max_shards = None
    if hasattr(a, "sample_cap") and a.sample_cap == 0:
        a.sample_cap = None
    return a


if __name__ == "__main__":
    main()
