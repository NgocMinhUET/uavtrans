"""Statistical testing and confidence interval utilities for UDST evaluation.

Provides:
  - Bootstrap confidence intervals (95% by default)
  - Paired Wilcoxon signed-rank test between two method columns
  - BD-Rate (Bjontegaard-Delta Rate) between two rate-accuracy curves
  - Summary table builder with CI columns
"""
from __future__ import annotations

from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    stat: str = "mean"
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for a statistic.

    Args:
        values:  1-D sequence of per-image measurements.
        n_boot:  Number of bootstrap resamples.
        ci:      Coverage level, e.g. 0.95 for 95% CI.
        seed:    Random seed for reproducibility.
        stat:    "mean" or "median".

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    fn = np.mean if stat == "mean" else np.median
    point = fn(arr)

    boots = np.array([fn(rng.choice(arr, size=len(arr), replace=True))
                      for _ in range(n_boot)])
    alpha = 1.0 - ci
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Paired Wilcoxon test
# ─────────────────────────────────────────────────────────────────────────────

def paired_wilcoxon(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided"
) -> dict:
    """
    Paired Wilcoxon signed-rank test: H0 is that medians are equal.

    Args:
        a, b:        Paired per-image measurements (same order).
        alternative: "two-sided" | "greater" | "less"
                     "greater" means H1: a > b.

    Returns:
        dict with keys: statistic, p_value, significant_005, effect_size_r
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = sp_stats.wilcoxon(a - b, alternative=alternative,
                                    zero_method="wilcox")

    n = len(a)
    # Rank-biserial correlation as effect size
    r = stat / (n * (n + 1) / 2)

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant_005": bool(p < 0.05),
        "significant_001": bool(p < 0.01),
        "effect_size_r": float(r),
        "n": int(n),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BD-Rate
# ─────────────────────────────────────────────────────────────────────────────

def bd_rate(
    rate_ref: Sequence[float],
    metric_ref: Sequence[float],
    rate_test: Sequence[float],
    metric_test: Sequence[float],
    min_points: int = 4
) -> float:
    """
    Bjontegaard-Delta Rate: percentage bit-saving of test vs reference
    at the same quality level. Negative = test uses fewer bits.

    Both sequences must be sorted by rate (ascending).
    Computes average over the overlapping metric range.

    Args:
        rate_ref, metric_ref:   Reference curve (bytes, dice/iou).
        rate_test, metric_test: Test curve.
        min_points:             Minimum points needed; returns NaN if fewer.

    Returns:
        BD-Rate in % (negative = better for test).
    """
    rate_ref   = np.log(np.clip(rate_ref,   1, None))
    rate_test  = np.log(np.clip(rate_test,  1, None))
    metric_ref  = np.asarray(metric_ref,  dtype=np.float64)
    metric_test = np.asarray(metric_test, dtype=np.float64)

    if len(rate_ref) < min_points or len(rate_test) < min_points:
        return float("nan")

    min_m = max(metric_ref.min(), metric_test.min())
    max_m = min(metric_ref.max(), metric_test.max())
    if min_m >= max_m:
        return float("nan")

    # Fit cubic polynomials metric → log(rate) for interpolation
    try:
        poly_ref  = np.polyfit(metric_ref,  rate_ref,  deg=3)
        poly_test = np.polyfit(metric_test, rate_test, deg=3)
    except np.linalg.LinAlgError:
        return float("nan")

    ms = np.linspace(min_m, max_m, 100)
    log_rate_ref  = np.polyval(poly_ref,  ms)
    log_rate_test = np.polyval(poly_test, ms)

    avg_diff = np.trapz(log_rate_test - log_rate_ref, ms) / (max_m - min_m)
    return float((np.exp(avg_diff) - 1.0) * 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_table(
    df: pd.DataFrame,
    method_col: str = "method",
    metrics: tuple = ("dice", "iou", "precision", "recall"),
    byte_col: str = "payload_bytes",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a publication-ready summary table with mean, 95% CI, and std
    for each method.

    Returns:
        DataFrame with columns: method, N, {metric}_mean, {metric}_ci_lo,
        {metric}_ci_hi, {metric}_std, payload_kb_mean, payload_kb_ci_lo,
        payload_kb_ci_hi.
    """
    rows = []
    for method, grp in df.groupby(method_col):
        row: dict = {"method": method, "N": len(grp)}

        kb_vals = grp[byte_col].values / 1024.0
        pt, lo, hi = bootstrap_ci(kb_vals, n_boot=n_boot, ci=ci, seed=seed)
        row["payload_kb_mean"] = round(pt, 3)
        row["payload_kb_ci_lo"] = round(lo, 3)
        row["payload_kb_ci_hi"] = round(hi, 3)

        for m in metrics:
            if m not in grp.columns:
                continue
            vals = grp[m].dropna().values
            pt, lo, hi = bootstrap_ci(vals, n_boot=n_boot, ci=ci, seed=seed)
            row[f"{m}_mean"]  = round(pt, 4)
            row[f"{m}_ci_lo"] = round(lo, 4)
            row[f"{m}_ci_hi"] = round(hi, 4)
            row[f"{m}_std"]   = round(float(np.std(vals)), 4)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("payload_kb_mean").reset_index(drop=True)


def compare_methods(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str = "dice",
    method_col: str = "method",
    id_col: str = "event_id",
) -> dict:
    """
    Paired statistical comparison of two methods on per-image measurements.

    Pairs are matched by event_id (must be the same images for both methods).

    Returns:
        Dict with mean_diff, wilcoxon result, interpretation string.
    """
    a_df = df[df[method_col] == method_a].set_index(id_col)[metric]
    b_df = df[df[method_col] == method_b].set_index(id_col)[metric]
    common = a_df.index.intersection(b_df.index)

    if len(common) < 5:
        return {"error": "Too few paired samples", "n": len(common)}

    a = a_df.loc[common].values
    b = b_df.loc[common].values

    wtest = paired_wilcoxon(a, b, alternative="two-sided")
    mean_diff = float(np.mean(a - b))

    if wtest["p_value"] < 0.05:
        direction = f"{method_a} > {method_b}" if mean_diff > 0 else f"{method_b} > {method_a}"
        interpretation = f"Significant (p={wtest['p_value']:.4f}): {direction}, |r|={wtest['effect_size_r']:.3f}"
    else:
        interpretation = f"Not significant (p={wtest['p_value']:.4f}), mean diff={mean_diff:+.4f}"

    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "n_pairs": len(common),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": mean_diff,
        **wtest,
        "interpretation": interpretation,
    }


def run_all_comparisons(
    df: pd.DataFrame,
    reference_method: str,
    other_methods: list[str],
    metric: str = "dice",
    method_col: str = "method",
    id_col: str = "event_id",
) -> pd.DataFrame:
    """Compare reference_method against each of other_methods. Returns a table."""
    rows = []
    for m in other_methods:
        if m == reference_method:
            continue
        result = compare_methods(df, reference_method, m, metric, method_col, id_col)
        rows.append(result)
    return pd.DataFrame(rows)
