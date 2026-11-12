"""Volatility clustering test #1: autocorrelation of r, |r| and r^2, plus Ljung-Box.

If volatility clusters, returns r are ~uncorrelated but |r| (and r^2) are positively
autocorrelated: a big move today makes a big move tomorrow more likely.

Run on the two sets of series that move often enough for this to mean anything:
  * AWS spot listings (price changes on ~75% of days, in small steps)
  * spot matched-listing (chained) indexes per GPU type, as in analysis.py

Gaps: returns sit on a calendar-day grid, so a return exists only when both days have a price,
and autocorrelations use only pairs where both lagged values exist.
Null baseline: shuffle each series' returns over its own observed days (keeps the gaps and the
return distribution, destroys any time ordering) and recompute.
Outputs go to ./out.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
LAGS = 20
LB_LAGS = 10       # Ljung-Box Q(10)
N_SHUFFLE = 200
rng = np.random.default_rng(0)

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

# ------------------------------------------------------------------ load (same cleaning as analysis.py)
df = pd.read_parquet(os.path.join(HERE, "all.parquet"))
df = df[(df.quality == "ok") & (df.gpu_count > 0) & (df.price_per_hour > 0)
        & (df.gpu_type != "Unknown") & (df.provider != "datacrunch")].copy()
df["day"] = df.timestamp.dt.tz_convert(None).dt.normalize()
# Log returns are scale-free, so $/instance vs $/GPU (and the fractional-GPU fix) doesn't matter here.
key = ["provider", "instance_type", "region", "is_spot", "gpu_type", "gpu_count", "vcpus", "ram_gb"]
daily = (df.sort_values("timestamp").groupby(key + ["day"], dropna=False).price_per_hour.last()
         .reset_index())
daily["sid"] = daily.groupby(key, dropna=False).ngroup()
days = pd.date_range(daily.day.min(), daily.day.max(), freq="D")
logp = np.log(daily.pivot(index="day", columns="sid", values="price_per_hour").reindex(days))
R_all = logp.diff()  # NaN unless both consecutive calendar days have a price
meta = daily.drop_duplicates("sid").set_index("sid")[key]


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


def shuffle_cols(X):
    """Permute each column's observed values among its own observed positions."""
    Y = X.copy()
    for j in range(X.shape[1]):
        m = ~np.isnan(X[:, j])
        Y[m, j] = rng.permutation(X[m, j])
    return Y


def analyse(R, label):
    """ACF of r, |r|, r^2 per series + Ljung-Box + shuffle baseline for the median |r| ACF."""
    X = R.to_numpy()
    res = {}
    for name, Z in [("r", X), ("abs_r", np.abs(X)), ("r2", X ** 2)]:
        rho, n = acf(Z)
        q, p = ljung_box(rho, n)
        res[name] = dict(rho=rho, n=n, q=q, p=p)
    null_med = np.array([np.nanmedian(acf(np.abs(shuffle_cols(X)))[0], axis=1) for _ in range(N_SHUFFLE)])
    per_series = pd.DataFrame({
        "n_returns": res["r"]["n"],
        "chg_freq": (np.abs(np.nan_to_num(X)) > 1e-9).sum(0) / res["r"]["n"],
        "acf1_r": res["r"]["rho"][0], "acf1_abs_r": res["abs_r"]["rho"][0],
        "acf1_r2": res["r2"]["rho"][0],
        "lb10_p_r": res["r"]["p"], "lb10_p_abs_r": res["abs_r"]["p"], "lb10_p_r2": res["r2"]["p"],
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
        "median_acf1_r": per_series.acf1_r.median(),
        "median_acf1_abs_r": per_series.acf1_abs_r.median(),
        "null_median_acf1_abs_r_p95": curve.null_abs_r_p95.iloc[0],
    }, name=label)
    return per_series, curve, summary


def ar_filter(R, pmax=5):
    """Remove the predictable (trend/momentum) part of each series before testing |r|.

    Per column: fit AR(p) with intercept by OLS for p = 0..pmax on rows where r_t and all pmax
    lags exist (a common sample, so BIC compares like with like), pick p by BIC, return the
    residuals e_t (NaN elsewhere) and the chosen p. Clustering is then tested on |e|, which no
    longer inherits autocorrelation from r itself."""
    X = R.to_numpy()
    T, N = X.shape
    E = np.full_like(X, np.nan)
    chosen = np.zeros(N, dtype=int)
    for j in range(N):
        x = X[:, j]
        L = np.column_stack([np.r_[np.full(k, np.nan), x[:-k]] for k in range(1, pmax + 1)])
        rows = ~np.isnan(x) & ~np.isnan(L).any(axis=1)
        n = rows.sum()
        if n < 30:
            continue
        y, Lr = x[rows], L[rows]
        best = None
        for p in range(pmax + 1):
            A = np.column_stack([np.ones(n), Lr[:, :p]])
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ beta
            rss = max((resid ** 2).sum(), 1e-300)
            bic = n * np.log(rss / n) + (p + 1) * np.log(n)
            if best is None or bic < best[0]:
                best = (bic, p, resid)
        E[rows, j] = best[2]
        chosen[j] = best[1]
    return pd.DataFrame(E, index=R.index, columns=R.columns), pd.Series(chosen, index=R.columns, name="ar_p")


# ------------------------------------------------------------------ A. AWS spot listings
aws_ids = meta.index[(meta.provider == "aws") & meta.is_spot]
R_aws = R_all[aws_ids]
n_ret = R_aws.notna().sum()
moves = (R_aws.abs() > 1e-9).sum() / n_ret
R_aws = R_aws.loc[:, (n_ret >= 100) & (moves >= 0.3)]  # enough data and enough movement
aws_series, aws_curve, aws_sum = analyse(R_aws, "AWS spot listings")
E_aws, aws_p = ar_filter(R_aws)
awsf_series, awsf_curve, awsf_sum = analyse(E_aws, "AWS spot listings, AR-filtered")
aws_series.join(awsf_series.add_prefix("filt_")).join(aws_p).join(meta).to_csv(
    os.path.join(OUT, "clustering_aws_spot_series.csv"))
aws_curve.round(4).to_csv(os.path.join(OUT, "clustering_aws_spot_acf.csv"), index=False)
awsf_curve.round(4).to_csv(os.path.join(OUT, "clustering_aws_spot_acf_filtered.csv"), index=False)

# ------------------------------------------------------------------ B. spot chained indexes
# Same construction as analysis.py: daily mean of same-listing returns, >= 3 listings, no Vast.ai.
long = R_all.stack().rename("r").reset_index().rename(columns={"level_0": "day"})
long = long.join(meta, on="sid")
long = long[(long.provider != "vastai") & long.is_spot]
ch = long.groupby(["gpu_type", "day"]).r.agg(["mean", "size"])
ch = ch[ch["size"] >= 3]["mean"].unstack("gpu_type").reindex(days)
ch = ch.loc[:, ch.notna().sum() >= 150]
ch = ch.loc[:, (ch.abs() > 1e-9).sum() / ch.notna().sum() >= 0.1]  # drop indexes that (almost) never move
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
