"""Volatility clustering test #1: autocorrelation of r, |r| and r^2, plus Ljung-Box.

If volatility clusters, returns r are ~uncorrelated but |r| (and r^2) are positively
autocorrelated: a big move today makes a big move tomorrow more likely.

Run on the two sets of series that move often enough for this to mean anything:
  * AWS spot listings (price changes on ~75% of days, in small steps)
  * spot matched-listing (chained) indexes per GPU type, as in analysis.py

Gaps: returns sit on a calendar-day grid, so a return exists only when both days have a price,
and autocorrelations use only pairs where both lagged values exist.
Null baseline: shuffle each series' returns only among days with the same weekday inside the same
30-day block (keeps gaps, the return distribution, weekly schedules and slow regime changes;
destroys short-horizon timing) and recompute. Ljung-Box is reported both with textbook p-values
and against this null (per-series 95th percentile of shuffled Q(10)).
Data-hygiene fixes from audit.py (duplicate paths, suspect days, Nebius split) come via common.py.
Outputs go to ./out.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from common import (BLUE, GRID, MUTED, ORANGE, OUT, aws_spot_active, ar_filter, load_returns,
                    shuffle_cols_blocked, spot_chained_indexes)

LAGS = 20
LB_LAGS = 10       # Ljung-Box Q(10)
N_SHUFFLE = 200
rng = np.random.default_rng(0)

R_all, meta, days = load_returns()



# ------------------------------------------------------------------ stats
def acf(X, lags=LAGS):
    """Per-column autocorrelation at lags 1..L using only pairs where both values exist.
    X: (T, N) array with NaNs. Returns (L, N) array and (N,) count of non-NaN obs."""
    X = X - np.nanmean(X, axis=0)
    out = np.full((lags, X.shape[1]), np.nan)
    for k in range(1, lags + 1):
        a, b = X[k:], X[:-k]
        m = ~np.isnan(a) & ~np.isnan(b)
        a, b = np.where(m, a, 0), np.where(m, b, 0)
        den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k - 1] = np.where(m.sum(0) >= 20, (a * b).sum(0) / den, np.nan)
    return out, (~np.isnan(X)).sum(0)


def ljung_box(rho, n, h=LB_LAGS):
    """Q(h) = n(n+2) sum_k rho_k^2/(n-k) and its chi2(h) p-value, per column."""
    k = np.arange(1, h + 1)[:, None]
    q = n * (n + 2) * np.nansum(rho[:h] ** 2 / (n - k), axis=0)
    return q, chi2.sf(q, h)


def analyse(R, label):
    """ACF of r, |r|, r^2 per series + Ljung-Box + shuffle baseline for the median |r| ACF."""
    X = R.to_numpy()
    res = {}
    for name, Z in [("r", X), ("abs_r", np.abs(X)), ("r2", X ** 2)]:
        rho, n = acf(Z)
        q, p = ljung_box(rho, n)
        res[name] = dict(rho=rho, n=n, q=q, p=p)
    # Null: shuffle within weekday x 30-day blocks (keeps weekly schedules and regime shifts).
    null_med, null_q = [], []
    for _ in range(N_SHUFFLE):
        rho0, n0 = acf(np.abs(shuffle_cols_blocked(X, rng, R.index)))
        null_med.append(np.nanmedian(rho0, axis=1))
        null_q.append(ljung_box(rho0, n0)[0])
    null_med = np.array(null_med)
    q_crit = np.nanpercentile(np.array(null_q), 95, axis=0)  # per-series calibrated Q(10) threshold
    per_series = pd.DataFrame({
        "n_returns": res["r"]["n"],
        "chg_freq": (np.abs(np.nan_to_num(X)) > 1e-9).sum(0) / res["r"]["n"],
        "acf1_r": res["r"]["rho"][0], "acf1_abs_r": res["abs_r"]["rho"][0],
        "acf1_r2": res["r2"]["rho"][0],
        "lb10_p_r": res["r"]["p"], "lb10_p_abs_r": res["abs_r"]["p"], "lb10_p_r2": res["r2"]["p"],
        "lb10_abs_r_beats_null": res["abs_r"]["q"] > q_crit,
    }, index=R.columns)
    curve = pd.DataFrame({
        "lag": np.arange(1, LAGS + 1),
        "median_acf_r": np.nanmedian(res["r"]["rho"], axis=1),
        "median_acf_abs_r": np.nanmedian(res["abs_r"]["rho"], axis=1),
        "median_acf_r2": np.nanmedian(res["r2"]["rho"], axis=1),
        "null_abs_r_p05": np.percentile(null_med, 5, axis=0),
        "null_abs_r_p95": np.percentile(null_med, 95, axis=0),
    })
    summary = pd.Series({
        "series": X.shape[1],
        "share_LB10_sig_r": (per_series.lb10_p_r < 0.05).mean(),
        "share_LB10_sig_abs_r": (per_series.lb10_p_abs_r < 0.05).mean(),
        "share_LB10_sig_r2": (per_series.lb10_p_r2 < 0.05).mean(),
        "share_abs_r_beats_blocked_null": per_series.lb10_abs_r_beats_null.mean(),
        "median_acf1_r": per_series.acf1_r.median(),
        "median_acf1_abs_r": per_series.acf1_abs_r.median(),
        "null_median_acf1_abs_r_p95": curve.null_abs_r_p95.iloc[0],
    }, name=label)
    return per_series, curve, summary



# ------------------------------------------------------------------ A. AWS spot listings
R_aws = aws_spot_active(R_all, meta)  # >= 100 returns and moves on >= 30% of days
aws_series, aws_curve, aws_sum = analyse(R_aws, "AWS spot listings")
E_aws, aws_p = ar_filter(R_aws)
awsf_series, awsf_curve, awsf_sum = analyse(E_aws, "AWS spot listings, AR-filtered")
aws_series.join(awsf_series.add_prefix("filt_")).join(aws_p).join(meta).to_csv(
    os.path.join(OUT, "clustering_aws_spot_series.csv"))
aws_curve.round(4).to_csv(os.path.join(OUT, "clustering_aws_spot_acf.csv"), index=False)
awsf_curve.round(4).to_csv(os.path.join(OUT, "clustering_aws_spot_acf_filtered.csv"), index=False)

# ------------------------------------------------------------------ B. spot chained indexes
# Same construction as analysis.py: daily mean of same-listing returns, >= 3 listings, no Vast.ai;
# indexes that (almost) never move are dropped.
ch = spot_chained_indexes(R_all, meta)
idx_series, idx_curve, idx_sum = analyse(ch, "spot chained indexes")
E_idx, idx_p = ar_filter(ch)
idxf_series, idxf_curve, idxf_sum = analyse(E_idx, "spot chained indexes, AR-filtered")
idx_series.join(idxf_series.add_prefix("filt_")).join(idx_p).round(4).sort_values(
    "filt_acf1_abs_r", ascending=False).to_csv(os.path.join(OUT, "clustering_spot_index_series.csv"))
idx_curve.round(4).to_csv(os.path.join(OUT, "clustering_spot_index_acf.csv"), index=False)
idxf_curve.round(4).to_csv(os.path.join(OUT, "clustering_spot_index_acf_filtered.csv"), index=False)

summary = pd.concat([aws_sum, awsf_sum, idx_sum, idxf_sum], axis=1).T
summary.round(4).to_csv(os.path.join(OUT, "clustering_summary.csv"))

# ------------------------------------------------------------------ chart
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, curve, title in [(axes[0], aws_curve, f"AWS spot listings, raw (n={len(aws_series)})"),
                         (axes[1], awsf_curve, "AWS spot listings, after AR filter"),
                         (axes[2], idxf_curve, f"Spot GPU indexes, after AR filter (n={len(idx_series)})")]:
    ax.fill_between(curve.lag, curve.null_abs_r_p05, curve.null_abs_r_p95, color=GRID, label="|r| if no clustering (shuffled, 5-95%)")
    ax.plot(curve.lag, curve.median_acf_abs_r, color=ORANGE, lw=2, marker="o", ms=4, label="|r|  (size of move)")
    ax.plot(curve.lag, curve.median_acf_r, color=BLUE, lw=2, marker="o", ms=4, label="r  (direction of move)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("lag (days)")
    ax.set_xticks([1, 5, 10, 15, 20])
axes[0].set_ylabel("median autocorrelation across series")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Volatility clustering: do big moves follow big moves?", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "09_volatility_clustering_acf.png"), dpi=150)
plt.close(fig)

pd.set_option("display.width", 200, "display.max_columns", 20)
print("== summary ==\n", summary.round(3))
print("\n== AWS spot: median ACF by lag ==\n", aws_curve.head(10).round(3).to_string(index=False))
print("\n== spot indexes: median ACF by lag ==\n", idx_curve.head(10).round(3).to_string(index=False))
print("\n== spot indexes, per GPU ==\n", idx_series.round(3).sort_values("acf1_abs_r", ascending=False).to_string())
print("\n== AWS spot: chosen AR order ==\n", aws_p.value_counts().sort_index().to_string())
print("\n== AWS spot after AR filter: median ACF by lag ==\n", awsf_curve.head(10).round(3).to_string(index=False))
print("\n== spot indexes after AR filter, per GPU ==\n",
      idxf_series.join(idx_p).round(3).sort_values("acf1_abs_r", ascending=False).to_string())
