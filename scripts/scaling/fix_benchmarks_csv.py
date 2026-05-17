"""
Post-process scaling_v100/benchmarks/benchmarks.csv: backfill val_bpb /
train_bpb columns from the per-tag eval_*.log files (the original
shell-script regex was wrong, only CORE was captured).

Safe to re-run; idempotent.
"""
import os
import re
import csv
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nanochat.common import get_base_dir  # noqa: E402

BASE_DIR = get_base_dir()
DIR = os.path.join(BASE_DIR, "scaling_v100", "benchmarks")
csv_path = os.path.join(DIR, "benchmarks.csv")

rows = list(csv.DictReader(open(csv_path)))
for r in rows:
    log = os.path.join(DIR, f'eval_{r["model_tag"]}.log')
    if not os.path.exists(log):
        continue
    t = open(log).read()
    v = re.search(r"val bpb:\s+([\d.]+)", t)
    tr = re.search(r"train bpb:\s+([\d.]+)", t)
    if v:
        r["val_bpb"] = v.group(1)
    if tr:
        r["train_bpb"] = tr.group(1)

with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"fixed {len(rows)} rows in {csv_path}")
