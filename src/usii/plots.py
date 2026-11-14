"""Figures for the sweep results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .threshold import BUDGETS, PRIMARY_BUDGET, PRIMARY_RULE, accurate_mask

FAMILY_COLORS = {"Decision Tree": "#4778a8", "Random Forest": "#e28e2c", "Gradient Boosting": "#59a14f"}


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)


def plot_usii_vs_accuracy(cfg, table, out):
    datasets, families = list(cfg["dataset"].unique()), list(cfg["family"].unique())
    fig, axes = plt.subplots(len(datasets), len(families), figsize=(4.6 * len(families), 3.3 * len(datasets)), squeeze=False)
    prim = table[(table["rule"] == PRIMARY_RULE) & (table["budget"] == PRIMARY_BUDGET)]
    for i, ds in enumerate(datasets):
        for j, fam in enumerate(families):
            ax, sub = axes[i, j], cfg[(cfg["dataset"] == ds) & (cfg["family"] == fam)]
            acc = accurate_mask(sub, PRIMARY_RULE)
            within = sub["sim_cost"] <= PRIMARY_BUDGET
            color = FAMILY_COLORS[fam]
            ax.errorbar(sub["usii"], sub["acc_mean"], yerr=sub["acc_se"], fmt="none", ecolor="#bbbbbb", lw=.8, zorder=1)
            ax.scatter(sub.loc[within, "usii"], sub.loc[within, "acc_mean"], c=color, s=28, zorder=2, label=f"SC ≤ {PRIMARY_BUDGET}")
            ax.scatter(sub.loc[~within, "usii"], sub.loc[~within, "acc_mean"], facecolors="none", edgecolors=color, s=28, zorder=2, label=f"SC > {PRIMARY_BUDGET}")
            floor = sub.loc[acc, "acc_mean"].min()
            ax.axhline(floor, color="grey", ls=":", lw=1)
            row = prim[(prim["dataset"] == ds) & (prim["family"] == fam)].iloc[0]
            if not np.isnan(row["usii_threshold"]):
                ax.axvline(row["usii_threshold"], color="crimson", ls="--", lw=1)
            ax.scatter([row["knee_usii"]], [row["knee_accuracy"]], marker="*", s=170, c="gold", edgecolors="k", zorder=3, label="knee")
            ax.set_title(f"{ds} — {fam}", fontsize=9)
            if j == 0: ax.set_ylabel("CV accuracy")
            if i == len(datasets) - 1: ax.set_xlabel("USII")
            if i == 0 and j == 0: ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("USII vs accuracy per configuration (dotted: accuracy floor, red dashed: USII cut-off for SC ≤ 9, star: knee)", fontsize=10)
    _save(fig, out, "usii_vs_accuracy.png")


def plot_accuracy_vs_cost(cfg, out):
    datasets = list(cfg["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.6), sharey=False, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = cfg[cfg["dataset"] == ds]
        for fam, g in d.groupby("family", sort=False):
            ax.scatter(g["sim_cost"], g["acc_mean"], s=14, alpha=.45, color=FAMILY_COLORS[fam])
            env = g.sort_values("sim_cost").assign(best=lambda x: x["acc_mean"].cummax())
            ax.step(env["sim_cost"], env["best"], where="post", color=FAMILY_COLORS[fam], lw=1.8, label=fam)
        for b in (7, 9):
            ax.axvline(b, color="grey", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_title(ds, fontsize=10); ax.set_xlabel("simulation cost (conditions per prediction)")
    axes[0, 0].set_ylabel("CV accuracy")
    fig.suptitle("Accuracy vs simulation cost (lines: best accuracy reachable at or below a cost; dotted: SC = 7, 9)", fontsize=10)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    _save(fig, out, "accuracy_vs_simulation_cost.png")


def plot_usii_vs_cost(cfg, out):
    datasets = list(cfg["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = cfg[cfg["dataset"] == ds]
        for fam, g in d.groupby("family", sort=False):
            ax.scatter(g["sim_cost"], g["usii"], s=16, color=FAMILY_COLORS[fam], label=fam, alpha=.8)
        for b in BUDGETS:
            ax.axvline(b, color="grey", ls=":", lw=.8)
        rho = d[["sim_cost", "usii"]].corr(method="spearman").iloc[0, 1]
        ax.set_xscale("log"); ax.set_title(f"{ds}  (Spearman ρ = {rho:.2f})", fontsize=9)
        ax.set_xlabel("simulation cost")
    axes[0, 0].set_ylabel("USII"); axes[0, 0].legend(fontsize=7)
    _save(fig, out, "usii_vs_simulation_cost.png")


def plot_threshold_summary(table, out):
    t = table[(table["rule"] == PRIMARY_RULE) & (table["budget"] == PRIMARY_BUDGET) & (table["dataset"] != "ALL")]
    datasets, families = list(t["dataset"].unique()), list(t["family"].unique())
    x, w = np.arange(len(datasets)), 0.26
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
    for k, fam in enumerate(families):
        g = t[t["family"] == fam].set_index("dataset").loc[datasets]
        a1.bar(x + (k - 1) * w, g["min_sim_cost_accurate"], w, color=FAMILY_COLORS[fam], label=fam)
        a2.bar(x + (k - 1) * w, g["knee_usii"], w, color=FAMILY_COLORS[fam], label=fam)
        a2.scatter(x + (k - 1) * w, g["usii_threshold"], marker="_", s=260, c="crimson", zorder=3,
                   label="USII cut-off for SC ≤ 9" if k == 0 else None)
    for b in (7, 9):
        a1.axhline(b, color="grey", ls=":", lw=1)
    a1.set_yscale("log"); a1.set_xticks(x, datasets, rotation=20)
    a1.set_ylabel("lowest simulation cost among accurate-enough models")
    a1.set_title("Cost needed to stay accurate (dotted: SC = 7 and 9)", fontsize=10); a1.legend(fontsize=8)
    a2.set_xticks(x, datasets, rotation=20); a2.set_ylabel("USII")
    a2.set_title("USII of the most interpretable accurate-enough model (knee)", fontsize=10)
    a2.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, out, "threshold_summary.png")


def plot_weight_sensitivity(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in ((axes[0], "spearman_p05", "5th-percentile Spearman ρ vs baseline ranking"),
                           (axes[1], "knee_same_fraction", "Share of weightings that keep the same knee model")):
        pv = summary.pivot(index="dataset", columns="family", values=col)[list(FAMILY_COLORS)]
        im = ax.imshow(pv.to_numpy(), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(pv.shape[1]), pv.columns, rotation=20); ax.set_yticks(range(pv.shape[0]), pv.index)
        for (r, c), v in np.ndenumerate(pv.to_numpy()):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", color="white" if v < .6 else "black", fontsize=9)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Robustness to 200 random weightings (Dirichlet(1,1,1,1))", fontsize=11)
    _save(fig, out, "weight_sensitivity.png")


def make_all(cfg, table, wsum, out):
    out = Path(out)
    plot_usii_vs_accuracy(cfg, table, out)
    plot_accuracy_vs_cost(cfg, out)
    plot_usii_vs_cost(cfg, out)
    plot_threshold_summary(table, out)
    plot_weight_sensitivity(wsum, out)
