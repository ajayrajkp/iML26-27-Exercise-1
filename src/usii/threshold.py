"""Threshold-of-interpretability analysis.

"Interpretable" is operationalised in two independent ways, and the threshold is where
they can be satisfied together:

* Cognitive budget: simulation cost SC <= B, i.e. a person has to evaluate at most B
  split conditions to reproduce one prediction by hand. B is swept over BUDGETS;
  B = 9 (upper end of the "7 +/- 2" working-memory heuristic) is the primary value.
* Accuracy: accuracy within a tolerance of the best configuration of the same family on
  the same dataset. Rules: the one-standard-error rule (primary) and fixed tolerances.

Two questions are answered for every (dataset, family) and for each family pooled over
datasets ("ALL"):
1. Feasibility: does any configuration satisfy both criteria?
2. USII threshold: what USII value best separates models within the cognitive budget from
   those outside it (Youden-optimal cut-off), and how well does USII rank them (AUC)?
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

BUDGETS = (5, 7, 9, 15, 25)
PRIMARY_BUDGET = 9
PRIMARY_RULE = "1se"
RULES = ("1se", "tol_0.01", "tol_0.02", "tol_0.05")


def accurate_mask(df, rule):
    """Boolean mask of configurations that are 'accurate enough' within one (dataset, family) group."""
    best = df["acc_mean"].idxmax()
    top, se = df.loc[best, "acc_mean"], df.loc[best, "acc_se"]
    margin = se if rule == "1se" else float(rule.split("_")[1])
    return df["acc_mean"] >= top - margin - 1e-12


def usii_cutoff(usii, positive):
    """AUC and Youden-optimal USII cut-off for separating positives (within budget) from negatives."""
    positive = np.asarray(positive, dtype=bool)
    if positive.all() or (~positive).all():
        return np.nan, np.nan
    auc = roc_auc_score(positive, usii)
    fpr, tpr, thr = roc_curve(positive, usii)
    j = tpr - fpr
    # roc_curve lists thresholds in decreasing order, so the first maximiser is the highest cut-off.
    best = int(np.flatnonzero(j >= j.max() - 1e-12)[0])
    return float(auc), float(thr[best])


def _knee(sub, accurate):
    """Most interpretable (highest USII) configuration among the accurate-enough ones."""
    cand = sub[accurate].sort_values(["usii", "acc_mean"], ascending=False)
    return cand.iloc[0]


def analyze_group(sub, rule, budget):
    acc = accurate_mask(sub, rule)
    within = sub["sim_cost"] <= budget
    knee = _knee(sub, acc)
    both = acc & within
    auc, tau = usii_cutoff(sub["usii"].to_numpy(), within.to_numpy())
    best_both = sub[both].sort_values(["acc_mean", "usii"], ascending=False)
    return {
        "n_configs": len(sub), "n_accurate": int(acc.sum()), "n_within_budget": int(within.sum()),
        "n_both": int(both.sum()), "feasible": bool(both.any()),
        "best_acc": float(sub["acc_mean"].max()),
        "min_sim_cost_accurate": float(sub.loc[acc, "sim_cost"].min()),
        "knee_config": knee["config_id"], "knee_usii": float(knee["usii"]),
        "knee_sim_cost": float(knee["sim_cost"]), "knee_accuracy": float(knee["acc_mean"]),
        "best_feasible_config": best_both.iloc[0]["config_id"] if both.any() else "",
        "best_feasible_accuracy": float(best_both.iloc[0]["acc_mean"]) if both.any() else np.nan,
        "best_feasible_usii": float(best_both.iloc[0]["usii"]) if both.any() else np.nan,
        "auc_usii_vs_budget": auc, "usii_threshold": tau,
    }


def threshold_table(config_df, budgets=BUDGETS, rules=RULES):
    """Full grid of threshold results over (dataset or ALL) x family x rule x budget."""
    rows = []
    for family, fam_df in config_df.groupby("family", sort=False):
        for ds, sub in list(fam_df.groupby("dataset", sort=False)) + [("ALL", None)]:
            for rule in rules:
                for b in budgets:
                    if ds == "ALL":
                        # Pool datasets: accuracy rule applies per dataset, USII cut-off is pooled.
                        masks, parts = [], []
                        for _, d in fam_df.groupby("dataset", sort=False):
                            masks.append(accurate_mask(d, rule)); parts.append(d)
                        pooled = pd.concat(parts)
                        acc = pd.concat(masks)
                        within = pooled["sim_cost"] <= b
                        auc, tau = usii_cutoff(pooled["usii"].to_numpy(), within.to_numpy())
                        per_ds = [bool((m & (d["sim_cost"] <= b)).any()) for m, d in zip(masks, parts)]
                        rows.append({"dataset": ds, "family": family, "rule": rule, "budget": b,
                                     "n_configs": len(pooled), "n_accurate": int(acc.sum()),
                                     "n_within_budget": int(within.sum()), "n_both": int((acc & within).sum()),
                                     "feasible": all(per_ds), "auc_usii_vs_budget": auc, "usii_threshold": tau})
                    else:
                        rows.append({"dataset": ds, "family": family, "rule": rule, "budget": b,
                                     **analyze_group(sub, rule, b)})
    return pd.DataFrame(rows)


def pareto_flags(df, x="usii", y="acc_mean"):
    """True for configurations not dominated in (higher x, higher y)."""
    xs, ys = df[x].to_numpy(), df[y].to_numpy()
    flags = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        dom = (xs >= xs[i]) & (ys >= ys[i]) & ((xs > xs[i]) | (ys > ys[i]))
        flags[i] = not dom.any()
    return pd.Series(flags, index=df.index)


def add_pareto(config_df):
    """Flag Pareto-optimal configurations over (USII, accuracy), across and within families."""
    out = config_df.copy()
    out["pareto_overall"] = False
    out["pareto_in_family"] = False
    for _, idx in out.groupby("dataset", sort=False).groups.items():
        out.loc[idx, "pareto_overall"] = pareto_flags(out.loc[idx]).to_numpy()
    for _, idx in out.groupby(["dataset", "family"], sort=False).groups.items():
        out.loc[idx, "pareto_in_family"] = pareto_flags(out.loc[idx]).to_numpy()
    return out


def primary_summary(table):
    """Compact headline table: primary rule and budget, per dataset and family."""
    t = table[(table["rule"] == PRIMARY_RULE) & (table["budget"] == PRIMARY_BUDGET)]
    return t.drop(columns=["rule", "budget"]).reset_index(drop=True)
