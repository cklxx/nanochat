"""
v1 vs v2 emergence comparison: pull results.csv (vocab=32K sweep) and
emerge.csv (vocab=8K progressive sweep), plot them on the same axes, and
write paper-quality figures + a fit_summary covering both regimes.

Outputs to $NANOCHAT_BASE_DIR/scaling_v100_emerge/figures/.
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nanochat.common import get_base_dir  # noqa: E402

BASE_DIR = get_base_dir()
V1_DIR = os.path.join(BASE_DIR, "scaling_v100")
V2_DIR = os.path.join(BASE_DIR, "scaling_v100_emerge")
FIG_DIR = os.path.join(V2_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=220)
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    plt.close(fig)


def load_v1():
    """v1: vocab=32K sweep results + CORE benchmarks."""
    res = pd.read_csv(os.path.join(V1_DIR, "results.csv"))
    bench = pd.read_csv(os.path.join(V1_DIR, "benchmarks", "benchmarks.csv"))
    bench["core_metric"] = pd.to_numeric(bench["core_metric"], errors="coerce")
    bench["flops_budget"] = pd.to_numeric(bench["flops_budget"], errors="coerce")
    res["flops_budget"] = pd.to_numeric(res["flops_budget"], errors="coerce")
    df = res.merge(
        bench[["flops_budget", "depth", "core_metric"]],
        on=["flops_budget", "depth"],
        how="left",
    )
    df["effective_params"] = df["params_transformer"] + df["params_lm_head"]
    df["vocab"] = "32K"
    return df


def load_v2():
    """v2: vocab=8K progressive emerge sweep (CORE and val_bpb already inline)."""
    df = pd.read_csv(os.path.join(V2_DIR, "emerge.csv"))
    # core_metric and val_bpb already columns
    df["flops_budget"] = pd.to_numeric(df["flops_budget"], errors="coerce")
    df["core_metric"] = pd.to_numeric(df["core_metric"], errors="coerce")
    df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
    # synthesise effective_params from params_total minus embeddings (approximation
    # since emerge.csv doesn't break params down — but we know vocab=8K embedding
    # is ~3 * vocab * model_dim).
    df["model_dim"] = df["depth"] * 64
    df["emb_approx"] = 3 * 8192 * df["model_dim"]
    df["effective_params"] = df["params_total"] - df["emb_approx"]
    df["vocab"] = "8K"
    return df


def power_law_fit(x, y):
    lx, ly = np.log10(x.astype(float)), np.log10(y.astype(float))
    slope, intercept = np.polyfit(lx, ly, 1)
    yhat = 10 ** (slope * lx + intercept)
    ss_res = float(np.sum((y.astype(float) - yhat) ** 2))
    ss_tot = float(np.sum((y.astype(float) - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1-only", action="store_true")
    args = p.parse_args()

    v1 = load_v1()
    v2 = load_v2() if not args.v1_only and os.path.exists(os.path.join(V2_DIR, "emerge.csv")) else None

    # ===== Plot 1: val_bpb vs compute (compute-optimal frontier) =====
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    v1_opt = (v1.sort_values("val_bpb").drop_duplicates("flops_budget")
                 .sort_values("flops_budget"))
    ax.loglog(v1_opt["flops_budget"], v1_opt["val_bpb"], "o-",
              label="v1 vocab=32K (best per budget)", color="C0", markersize=10)
    if v2 is not None and len(v2) >= 2:
        # for v2, every (flops, depth) is one run — use best per flops as frontier
        v2_opt = (v2.sort_values("val_bpb").drop_duplicates("flops_budget")
                     .sort_values("flops_budget"))
        ax.loglog(v2_opt["flops_budget"], v2_opt["val_bpb"], "s-",
                  label="v2 vocab=8K (best per budget)", color="C3", markersize=10)
        # fit both
        fit1 = power_law_fit(v1_opt["flops_budget"].values, v1_opt["val_bpb"].values)
        fit2 = power_law_fit(v2_opt["flops_budget"].values, v2_opt["val_bpb"].values)
        cs = np.geomspace(min(v1_opt["flops_budget"].min(), v2_opt["flops_budget"].min()),
                          max(v1_opt["flops_budget"].max(), v2_opt["flops_budget"].max()), 100)
        ax.plot(cs, 10 ** (fit1["slope"] * np.log10(cs) + fit1["intercept"]),
                "--", color="C0", alpha=0.5,
                label=fr"v1 fit: $L \propto C^{{{fit1['slope']:.3f}}}$ $R^2$={fit1['r2']:.2f}")
        ax.plot(cs, 10 ** (fit2["slope"] * np.log10(cs) + fit2["intercept"]),
                "--", color="C3", alpha=0.5,
                label=fr"v2 fit: $L \propto C^{{{fit2['slope']:.3f}}}$ $R^2$={fit2['r2']:.2f}")
    ax.set_xlabel("training compute $C$ (FLOPs)")
    ax.set_ylabel("best val_bpb at compute $C$")
    ax.set_title("Loss vs compute — v1 (vocab=32K) vs v2 (vocab=8K) on V100")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    _save(fig, "loss_vs_compute_v1v2")

    # ===== Plot 2: CORE vs compute =====
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    v1c = v1.dropna(subset=["core_metric"]).copy()
    for d in sorted(v1c["depth"].unique()):
        sub = v1c[v1c["depth"] == d].sort_values("flops_budget")
        ax.semilogx(sub["flops_budget"], sub["core_metric"], "o-", alpha=0.5,
                    label=f"v1 32K d={d}", markersize=7)
    if v2 is not None:
        v2c = v2.dropna(subset=["core_metric"]).copy()
        for d in sorted(v2c["depth"].unique()):
            sub = v2c[v2c["depth"] == d].sort_values("flops_budget")
            ax.semilogx(sub["flops_budget"], sub["core_metric"], "s-",
                        label=f"v2 8K d={d}", markersize=9, linewidth=2)
    ax.axhline(0, ls=":", color="gray", label="random baseline")
    ax.axhline(0.20, ls="-.", color="orange", alpha=0.8, label="target CORE=0.20")
    ax.axhline(0.2565, ls="--", color="gray", alpha=0.7, label="GPT-2 1.6B = 0.2565")
    ax.set_xlabel("training compute $C$ (FLOPs)")
    ax.set_ylabel("DCLM CORE metric (centered)")
    ax.set_title("Capability emergence vs compute (V100, vocab=32K vs vocab=8K)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "core_vs_compute_v1v2")

    # ===== Plot 3: CORE vs val_bpb (does capability track loss?) =====
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    ax.scatter(v1["val_bpb"], v1["core_metric"], c="C0", marker="o",
               label="v1 vocab=32K", s=80, edgecolor="black", alpha=0.7)
    if v2 is not None:
        ax.scatter(v2["val_bpb"], v2["core_metric"], c="C3", marker="s",
                   label="v2 vocab=8K", s=80, edgecolor="black", alpha=0.7)
    ax.axhline(0, ls=":", color="gray")
    ax.axhline(0.20, ls="-.", color="orange", alpha=0.6, label="CORE=0.20")
    ax.set_xlabel("validation bits-per-byte (val_bpb)")
    ax.set_ylabel("DCLM CORE metric (centered)")
    ax.set_title("CORE tracks val_bpb across both vocab regimes")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, "core_vs_bpb")

    # ===== Plot 4: token:param ratio in the emerge runs =====
    if v2 is not None and len(v2):
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
        ratio_v2 = v2["tokens_trained"] / v2["effective_params"]
        ax.scatter(v2["flops_budget"], ratio_v2, c=v2["core_metric"], s=200,
                   cmap="viridis", edgecolor="black")
        for _, row in v2.iterrows():
            ax.annotate(f"d={int(row['depth'])}",
                        (row["flops_budget"], row["tokens_trained"]/row["effective_params"]),
                        xytext=(8, 8), textcoords="offset points", fontsize=9)
        ax.axhline(20, ls=":", color="gray", label="Chinchilla 20:1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("compute $C$ (FLOPs)")
        ax.set_ylabel("tokens : effective params")
        ax.set_title("v2 emerge runs — token:param ratio (color = CORE)")
        cbar = plt.colorbar(ax.collections[0], label="CORE")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        _save(fig, "token_param_v2")

    # ===== fit summary =====
    fit_summary = {
        "v1_loss_fit": power_law_fit(v1_opt["flops_budget"].values, v1_opt["val_bpb"].values),
        "v2_loss_fit": (power_law_fit(v2_opt["flops_budget"].values, v2_opt["val_bpb"].values)
                        if v2 is not None and len(v2) >= 2 else None),
        "v1_best_core": float(v1["core_metric"].max()),
        "v2_best_core": float(v2["core_metric"].max()) if v2 is not None else None,
        "v1_best_run": v1.loc[v1["core_metric"].idxmax(), ["flops_budget", "depth", "val_bpb", "core_metric"]].to_dict()
                       if v1["core_metric"].notna().any() else None,
        "v2_best_run": (v2.loc[v2["core_metric"].idxmax(), ["flops_budget", "depth", "val_bpb", "core_metric"]].to_dict()
                        if v2 is not None and v2["core_metric"].notna().any() else None),
    }

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    path = os.path.join(V2_DIR, "fit_summary_v1v2.json")
    with open(path, "w") as f:
        json.dump(_clean(fit_summary), f, indent=2, default=str)
    print(f"wrote figures to {FIG_DIR}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
