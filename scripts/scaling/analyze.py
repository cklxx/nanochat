"""
Analysis & paper-quality plots for the minimum-scale scaling law sweep.

Reads $NANOCHAT_BASE_DIR/scaling_v100/results.csv (produced by
runs/scaling/v100_min.sh) and writes:

  scaling_v100/figures/
    isoflop.{png,pdf}                IsoFLOP curves (val_bpb vs N at fixed C)
    loss_vs_compute.{png,pdf}        L(C) on the compute-optimal frontier
    optimal_N_vs_C.{png,pdf}         N*(C) power law
    optimal_D_vs_C.{png,pdf}         D*(C) power law
    fit_summary.json                 fit parameters & R² for each plot

The methodology follows the Chinchilla-style IsoFLOP analysis (Hoffmann 2022)
and the nanochat dev/scaling_analysis notebook: per-budget quadratic-in-log
fit to find the optimum N*, then power-law fit of N* and D* against C.

This script never modifies the CSV — it is safe to re-run.
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
RESULTS_DIR = os.path.join(BASE_DIR, "scaling_v100")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def effective_params(row):
    # "Kaplan-style" matmul-only parameter count — the cleanest signal at small scale
    # because embedding tables dominate when depth is tiny.
    return row["params_transformer"] + row["params_lm_head"]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=220)
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    plt.close(fig)


def isoflop_fit(df_b):
    """Quadratic-in-log fit to find the optimum N at this compute budget."""
    x = np.log10(df_b["effective_params"].values)
    y = df_b["val_bpb"].values
    a, b, c = np.polyfit(x, y, 2)
    # min at x* = -b/(2a) if a>0
    if a <= 0:
        return None
    x_star = -b / (2 * a)
    y_star = a * x_star ** 2 + b * x_star + c
    return {"a": a, "b": b, "c": c, "log10_N_star": x_star, "N_star": 10 ** x_star, "bpb_star": y_star}


def power_law_fit(x, y):
    lx, ly = np.log10(x), np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    yhat = 10 ** (slope * lx + intercept)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": slope, "intercept": intercept, "r2": r2, "yhat": yhat}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.path.join(RESULTS_DIR, "results.csv"))
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"missing {args.csv} — run runs/scaling/v100_min.sh first", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    df = df[df["val_bpb"].notna() & (df["val_bpb"] > 0)].copy()
    df["effective_params"] = df.apply(effective_params, axis=1)
    df["flops_budget"] = df["flops_budget"].astype(float)

    # ===== IsoFLOP plot =====
    flops_budgets = sorted(df["flops_budget"].unique())
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(flops_budgets)))

    iso_fits = []
    for c, col in zip(flops_budgets, colors):
        sub = df[df["flops_budget"] == c].sort_values("effective_params")
        if len(sub) < 3:
            ax.plot(sub["effective_params"], sub["val_bpb"], "o-", color=col,
                    label=f"C={c:.0e} (n={len(sub)})", markersize=7)
            continue
        ax.plot(sub["effective_params"], sub["val_bpb"], "o", color=col,
                label=f"C={c:.0e}", markersize=8)
        fit = isoflop_fit(sub)
        if fit:
            xs = np.linspace(np.log10(sub["effective_params"].min()) - 0.1,
                             np.log10(sub["effective_params"].max()) + 0.1, 200)
            ys = fit["a"] * xs ** 2 + fit["b"] * xs + fit["c"]
            ax.plot(10 ** xs, ys, "--", color=col, alpha=0.6)
            ax.plot([fit["N_star"]], [fit["bpb_star"]], "*", color=col, markersize=14,
                    markeredgecolor="black", markeredgewidth=0.8)
            iso_fits.append({"flops": c, **{k: v for k, v in fit.items() if k != "a"}, "a": fit["a"]})

    ax.set_xscale("log")
    ax.set_xlabel("effective parameters $N$ (transformer + lm_head)")
    ax.set_ylabel("validation bits-per-byte (val_bpb)")
    ax.set_title("IsoFLOP curves on V100 — minimum-scale validation of Chinchilla-style scaling")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _save(fig, "isoflop")

    # ===== Loss vs compute (compute-optimal frontier) =====
    # Prefer the IsoFLOP fit's bpb* when available; otherwise the per-budget
    # empirical min val_bpb (a conservative proxy when the fit isn't unimodal).
    opt_rows = []
    for c in flops_budgets:
        sub = df[df["flops_budget"] == c].sort_values("effective_params")
        if len(sub) >= 3:
            fit = isoflop_fit(sub)
            if fit:
                n_star = fit["N_star"]
                bpb_star = fit["bpb_star"]
                # take the actual run closest to N_star (for tokens/depth)
                idx = (np.log10(sub["effective_params"]) - np.log10(n_star)).abs().idxmin()
                row = sub.loc[idx].copy()
                row["val_bpb"] = bpb_star
                row["effective_params"] = n_star
                opt_rows.append(row)
                continue
        # fallback: empirical min within this budget
        opt_rows.append(sub.loc[sub["val_bpb"].idxmin()])
    opt = pd.DataFrame(opt_rows).sort_values("flops_budget")
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
    ax.loglog(opt["flops_budget"], opt["val_bpb"], "o-", markersize=9)
    # power-law fit: val_bpb - irreducible ≈ A * C^(-alpha). Fit on raw bpb first.
    loss_fit = power_law_fit(opt["flops_budget"].values, opt["val_bpb"].values)
    cs = np.geomspace(opt["flops_budget"].min(), opt["flops_budget"].max(), 100)
    ax.plot(cs, 10 ** (loss_fit["slope"] * np.log10(cs) + loss_fit["intercept"]),
            "--", color="C3",
            label=fr"fit: $L \propto C^{{{loss_fit['slope']:.3f}}}$, $R^2$={loss_fit['r2']:.3f}")
    ax.set_xlabel("training compute $C$ (FLOPs)")
    ax.set_ylabel("best val_bpb at compute $C$")
    ax.set_title("Loss vs compute — V100 minimum-scale frontier")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, "loss_vs_compute")

    # ===== Optimal N(C) and D(C) =====
    fit_summary = {"isoflop_fits": iso_fits, "loss_vs_compute": loss_fit | {"yhat": None}}

    # Use the IsoFLOP-derived optima (with empirical fallback) for N*, D*
    opt_rows = opt
    if len(opt_rows) >= 2:
        n_fit = power_law_fit(opt_rows["flops_budget"].values,
                              opt_rows["effective_params"].values)
        d_fit = power_law_fit(opt_rows["flops_budget"].values,
                              opt_rows["tokens_trained"].values)

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
        ax.loglog(opt_rows["flops_budget"], opt_rows["effective_params"], "o", markersize=10)
        cs = np.geomspace(opt_rows["flops_budget"].min(),
                          opt_rows["flops_budget"].max(), 100)
        ax.plot(cs, 10 ** (n_fit["slope"] * np.log10(cs) + n_fit["intercept"]),
                "--", color="C2",
                label=fr"$N^* \propto C^{{{n_fit['slope']:.3f}}}$, $R^2$={n_fit['r2']:.3f}")
        ax.set_xlabel("compute $C$ (FLOPs)")
        ax.set_ylabel("optimal parameters $N^*$")
        ax.set_title("Compute-optimal parameter scaling")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        _save(fig, "optimal_N_vs_C")

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
        ax.loglog(opt_rows["flops_budget"], opt_rows["tokens_trained"], "o", markersize=10)
        ax.plot(cs, 10 ** (d_fit["slope"] * np.log10(cs) + d_fit["intercept"]),
                "--", color="C1",
                label=fr"$D^* \propto C^{{{d_fit['slope']:.3f}}}$, $R^2$={d_fit['r2']:.3f}")
        ax.set_xlabel("compute $C$ (FLOPs)")
        ax.set_ylabel("optimal tokens $D^*$")
        ax.set_title("Compute-optimal data scaling")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        _save(fig, "optimal_D_vs_C")

        fit_summary["N_vs_C"] = {k: v for k, v in n_fit.items() if k != "yhat"}
        fit_summary["D_vs_C"] = {k: v for k, v in d_fit.items() if k != "yhat"}

        # ===== Token:param ratio at the compute-optimal frontier =====
        # Chinchilla predicted ~20; nanochat empirically tunes target ratio.
        ratio = opt_rows["tokens_trained"].values / opt_rows["effective_params"].values
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
        ax.semilogx(opt_rows["flops_budget"], ratio, "o-", markersize=10)
        ax.axhline(20, ls=":", color="gray", label="Chinchilla (D/N=20)")
        ax.set_xlabel("compute $C$ (FLOPs)")
        ax.set_ylabel(r"compute-optimal $D^* / N^*$")
        ax.set_title("Token:param ratio along the compute-optimal frontier")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        _save(fig, "token_param_ratio")
        fit_summary["token_param_ratio"] = list(map(float, ratio))

    # JSON-safe
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if v is not None and not isinstance(v, np.ndarray)}
        if isinstance(o, (list, tuple)):
            return [_clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    out_path = os.path.join(RESULTS_DIR, "fit_summary.json")
    with open(out_path, "w") as f:
        json.dump(_clean(fit_summary), f, indent=2)
    print(f"wrote figures to {FIG_DIR}")
    print(f"wrote fits to {out_path}")

    # ===== Optional: benchmark (CORE) plots if benchmarks.csv exists =====
    bench_csv = os.path.join(RESULTS_DIR, "benchmarks", "benchmarks.csv")
    if os.path.exists(bench_csv):
        bdf = pd.read_csv(bench_csv)
        bdf["core_metric"] = pd.to_numeric(bdf["core_metric"], errors="coerce")
        bdf = bdf.dropna(subset=["core_metric"]).copy()
        bdf["flops_budget"] = bdf["flops_budget"].astype(float)
        # join in effective_params from main df
        bdf = bdf.merge(
            df[["flops_budget", "depth", "effective_params", "params_total", "tokens_trained"]],
            on=["flops_budget", "depth"], how="left")

        # CORE vs compute, one trace per depth
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        for d in sorted(bdf["depth"].unique()):
            sub = bdf[bdf["depth"] == d].sort_values("flops_budget")
            ax.semilogx(sub["flops_budget"], sub["core_metric"], "o-",
                        label=f"depth={d}", markersize=8)
        ax.axhline(0, ls=":", color="gray", label="random baseline (centered)")
        ax.axhline(0.2565, ls="--", color="gray", alpha=0.7, label="GPT-2 (1.6B)")
        ax.set_xlabel("training compute $C$ (FLOPs)")
        ax.set_ylabel("DCLM CORE metric (centered)")
        ax.set_title("Capability emergence vs compute (V100 minimum-scale runs)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        _save(fig, "core_vs_compute")

        # CORE vs params, one trace per budget
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        for c in sorted(bdf["flops_budget"].unique()):
            sub = bdf[bdf["flops_budget"] == c].sort_values("effective_params")
            ax.semilogx(sub["effective_params"], sub["core_metric"], "o-",
                        label=f"C={c:.0e}", markersize=8)
        ax.axhline(0, ls=":", color="gray")
        ax.set_xlabel("effective parameters $N$")
        ax.set_ylabel("DCLM CORE metric (centered)")
        ax.set_title("CORE vs $N$ at fixed compute")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        _save(fig, "core_vs_params")

        fit_summary["benchmark_summary"] = {
            "n_evaluated": int(len(bdf)),
            "best_core": float(bdf["core_metric"].max()),
            "best_core_row": _clean(bdf.loc[bdf["core_metric"].idxmax()].to_dict()),
            "mean_core": float(bdf["core_metric"].mean()),
        }
        with open(out_path, "w") as f:
            json.dump(_clean(fit_summary), f, indent=2)
        print(f"wrote benchmark plots to {FIG_DIR} and refreshed {out_path}")


if __name__ == "__main__":
    main()
