"""
Stricter classification v2:
  - code: needs BOTH (multiple unambiguous code tokens) AND (high non-prose structure)
  - math: needs LaTeX commands count OR enclosed $...$ math expressions
  - structured: list-heavy
  - prose: default
"""
import os, re, sys, json, argparse, random
from collections import Counter, defaultdict
import pyarrow.parquet as pq

# Unambiguous code tokens — multi-char operators / keywords that don't appear in prose
CODE_HARD = re.compile(
    r"(?:\bdef\s+\w+\(|\bclass\s+\w+\s*[:({]|"
    r"\bfunction\s+\w+\(|"
    r"\b(?:import|from)\s+[\w\.]+|"
    r"#include\s*<|"
    r"\bpublic\s+(?:static\s+)?(?:void|int|String|class)|"
    r"\bprivate\s+(?:static\s+)?(?:void|int|String|class)|"
    r"\b(?:var|let|const)\s+\w+\s*=|"
    r"return\s+\w+;|"
    r"=>\s*[\{\(]|"
    r"==[=]?|!=[=]?|"
    r"&&|\|\||"
    r"<\w+>|</\w+>|"  # HTML/XML tags
    r"```|"
    r"\$\w+\s*=|"     # bash/PHP
    r"println!?\(|"
    r"console\.log\(|"
    r"std::|"
    r"->\s*\w+\s*\{"
    r")"
)

# Lines that start with code-ish indentation patterns
CODE_LINE = re.compile(r"^\s{2,}[\w\$/\#].*[\{\};\(\)=]")
SHEBANG = re.compile(r"^#!")

MATH_LATEX = re.compile(
    r"\\(?:frac|sum|int|prod|sqrt|alpha|beta|gamma|delta|theta|"
    r"lambda|sigma|mu|partial|nabla|infty|leq|geq|neq|cdot|"
    r"begin\{|end\{|mathbb|mathrm|mathcal|left|right)"
)
MATH_INLINE = re.compile(r"\$[^\$\n]{2,80}\$")
MATH_HEAVY_EQUATION_KW = re.compile(
    r"\b(?:theorem|lemma|corollary|proof|equation|derivative|integral|"
    r"matrix|eigenvalue|polynomial|inequality)\b", re.I)


def list_parquets(d):
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.endswith(".parquet") and not f.endswith(".tmp") and f != "shard_99999.parquet"
    )


def iter_random_docs(path, n):
    pf = pq.ParquetFile(path)
    n_rg = pf.num_row_groups
    rg_indices = list(range(n_rg))
    random.shuffle(rg_indices)
    out = []
    for ri in rg_indices:
        if len(out) >= n:
            break
        rg = pf.read_row_group(ri, columns=["text"]).column("text").to_pylist()
        random.shuffle(rg)
        out.extend(rg[: max(1, n - len(out))])
    return out[:n]


def classify(doc):
    L = len(doc)
    if L < 50:
        return "other_short"

    lines = doc.split("\n")
    n_lines = max(len(lines), 1)

    # --- MATH first (strongest signal) ---
    latex_cmds = len(MATH_LATEX.findall(doc))
    inline_math = len(MATH_INLINE.findall(doc))
    math_kw = len(MATH_HEAVY_EQUATION_KW.findall(doc))

    # Real math has multiple LaTeX/inline cues
    math_score = latex_cmds + inline_math + (math_kw // 2)
    if math_score >= 3:
        return "math"
    if latex_cmds >= 1 and inline_math >= 2:
        return "math"

    # --- CODE: need multiple hard tokens AND code-like structure ---
    code_hard_hits = len(CODE_HARD.findall(doc))
    indented_lines = sum(1 for ln in lines if CODE_LINE.match(ln))
    indented_ratio = indented_lines / n_lines
    shebang = bool(SHEBANG.match(lines[0])) if lines else False

    # Strong signal: explicit code markers
    if code_hard_hits >= 5 and indented_ratio > 0.10:
        return "code"
    # Stronger signal: heavy indentation (real source code formatting)
    if indented_ratio >= 0.30 and code_hard_hits >= 2:
        return "code"
    if shebang:
        return "code"
    # Markdown-fenced code blocks
    if doc.count("```") >= 2 and code_hard_hits >= 3:
        return "code_in_md"
    # HTML/XML heavy
    tag_hits = len(re.findall(r"<[a-zA-Z][^>]{0,40}>", doc))
    if tag_hits >= 10 and tag_hits / max(L, 1) * 1000 > 5:
        return "markup"

    # --- STRUCTURED (list-heavy) ---
    bullet_re = re.compile(r"^[ \t]*([\-\*\+]|\d+[\.\)]|>)\s")
    bullet_lines = sum(1 for ln in lines if bullet_re.match(ln))
    list_ratio = bullet_lines / n_lines
    if list_ratio >= 0.5 and n_lines >= 5:
        return "structured"

    # --- PROSE default ---
    return "prose"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--per-shard", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    paths = list_parquets(args.dir)
    print(f"sampling from {len(paths)} shards, {args.per_shard} docs each")

    counts = Counter()
    char_counts = Counter()
    examples = defaultdict(list)
    sampled = 0
    per_shard_counts = []

    for i, p in enumerate(paths):
        docs = iter_random_docs(p, args.per_shard)
        sc = Counter()
        for d in docs:
            label = classify(d)
            counts[label] += 1
            sc[label] += 1
            char_counts[label] += len(d)
            sampled += 1
            if len(examples[label]) < 4:
                examples[label].append(d[:400])
        per_shard_counts.append(sc)
        if (i + 1) % 20 == 0 or i == len(paths) - 1:
            print(f"  ..{i+1}/{len(paths)} shards, total sampled: {sampled}")

    total = sum(counts.values())
    total_chars = sum(char_counts.values())

    print("\n=== category breakdown (by docs) ===")
    for cat in sorted(counts.keys(), key=lambda c: -counts[c]):
        pct = 100 * counts[cat] / total
        print(f"  {cat:18s} {counts[cat]:6,d}  ({pct:5.2f}%)")
    print(f"\n  total docs sampled: {total:,}")

    print("\n=== category breakdown (by chars / proxy for tokens) ===")
    for cat in sorted(char_counts.keys(), key=lambda c: -char_counts[c]):
        pct = 100 * char_counts[cat] / total_chars
        avg_len = char_counts[cat] / counts[cat] if counts[cat] else 0
        print(f"  {cat:18s} {char_counts[cat]:12,d} chars  ({pct:5.2f}%)  avg_doc={avg_len:.0f}")

    print("\n=== per-shard stability ===")
    for cat in counts.keys():
        ratios = []
        for sc in per_shard_counts:
            tot = sum(sc.values())
            if tot > 0:
                ratios.append(sc.get(cat, 0) / tot)
        if ratios:
            mean = sum(ratios) / len(ratios)
            var = sum((r - mean) ** 2 for r in ratios) / len(ratios)
            std = var ** 0.5
            print(f"  {cat:18s}: per-shard mean {100*mean:5.2f}% ± {100*std:4.2f}%")

    out_path = "/tmp/classify_summary_v2.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_shards": len(paths),
            "per_shard_n": args.per_shard,
            "total_sampled": total,
            "counts": dict(counts),
            "char_counts": dict(char_counts),
            "examples": dict(examples),
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
