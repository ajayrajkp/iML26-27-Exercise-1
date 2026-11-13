"""Robustness of the USII-based conclusions to the weights and normalisation constants.

Everything is recomputed from the stored per-fold structural statistics, so no model is
refitted. For each alternative setting we report, per (dataset, family):
  * spearman: rank agreement of USII with the baseline across configurations
  * auc: how well USII ranks within-budget models (cf. threshold.py)
  * knee_same: whether the 'most interpretable accurate-enough configuration' is unchanged
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metric import COMPONENTS, MAX_DEPTH, MAX_TREES, WEIGHTS, components_from_stats, usii_from_components
from .threshold import PRIMARY_BUDGET, PRIMARY_RULE, accurate_mask, usii_cutoff


def named_weight_schemes():
    schemes = {"baseline": dict(WEIGHTS), "equal": {k: 0.25 for k in COMPONENTS}}
    for dpc in (0.15, 0.45):  # remaining weight shared in proportion to the baseline
        rest = 1.0 - dpc
        base_rest = 1.0 - WEIGHTS["DPC"]
        schemes[f"dpc_{dpc:.2f}"] = {k: (dpc if k == "DPC" else WEIGHTS[k] * rest / base_rest) for k in COMPONENTS}
    for drop in COMPONENTS:  # ablation: remove one component, renormalise
        rest = 1.0 - WEIGHTS[drop]
        schemes[f"drop_{drop}"] = {k: 0.0 if k == drop else WEIGHTS[k] / rest for k in COMPONENTS}
    return schemes


def dirichlet_schemes(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return {f"dirichlet_{i:03d}": dict(zip(COMPONENTS, w)) for i, w in enumerate(rng.dirichlet(np.ones(4), size=n))}


def config_usii(fold_df, weights=None, max_depth=MAX_DEPTH, max_trees=MAX_TREES):
    """Mean USII per (dataset, config) for given weights/constants, from per-fold raw statistics."""
    c = components_from_stats(fold_df["mean_path_depth"], fold_df["mean_internal_nodes"],
                             fold_df["n_used_features"], fold_df["n_features"], fold_df["n_trees"],
                             max_depth, max_trees)
    tmp = fold_df[["dataset", "family", "config_id"]].copy()
    tmp["usii"] = usii_from_components(c, weights)
    return tmp.groupby(["dataset", "config_id"], sort=False)["usii"].mean()


def _compare(config_df, new_usii, label, kind):
    rows = []
    cfg = config_df.set_index(["dataset", "config_id"]).copy()
    cfg["usii_new"] = new_usii.reindex(cfg.index)
    for (ds, fam), sub in cfg.groupby(["dataset", "family"], sort=False):
        acc = accurate_mask(sub.reset_index(), PRIMARY_RULE).to_numpy()
        base_knee = sub.index[acc][np.argmax(sub["usii"].to_numpy()[acc])]
        new_knee = sub.index[acc][np.argmax(sub["usii_new"].to_numpy()[acc])]
        rho = spearmanr(sub["usii"], sub["usii_new"]).statistic if sub["usii_new"].nunique() > 1 else np.nan
        auc, tau = usii_cutoff(sub["usii_new"].to_numpy(), (sub["sim_cost"] <= PRIMARY_BUDGET).to_numpy())
        rows.append({"setting_type": kind, "setting": label, "dataset": ds, "family": fam,
                     "spearman": rho, "auc": auc, "usii_threshold": tau,
                     "knee_same": bool(base_knee == new_knee), "knee_config": new_knee[1]})
    return rows


def weight_sensitivity(fold_df, config_df):
    named, rows = named_weight_schemes(), []
    for label, w in named.items():
        rows += _compare(config_df, config_usii(fold_df, w), label, "named")
    named_df = pd.DataFrame(rows)
    rows = []
    for label, w in dirichlet_schemes().items():
        rows += _compare(config_df, config_usii(fold_df, w), label, "dirichlet")
    dd = pd.DataFrame(rows)
    summary = dd.groupby(["dataset", "family"], sort=False).agg(
        n_weightings=("setting", "size"), spearman_mean=("spearman", "mean"), spearman_min=("spearman", "min"),
        spearman_p05=("spearman", lambda s: s.quantile(0.05)), auc_mean=("auc", "mean"), auc_min=("auc", "min"),
        knee_same_fraction=("knee_same", "mean")).reset_index()
    return named_df, summary


def constants_sensitivity(fold_df, config_df):
    rows = []
    for md in (6, 8, 10):
        for mt in (100, 300, 1000):
            if (md, mt) == (MAX_DEPTH, MAX_TREES):
                continue
            rows += _compare(config_df, config_usii(fold_df, None, md, mt), f"max_depth={md},max_trees={mt}", "constants")
    return pd.DataFrame(rows)
