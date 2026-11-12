"""Shared helpers for the clustering scripts: chart style, cleaned daily return grid, AR filter,
fast within-series shuffle."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CACHE = os.path.join(HERE, "data", "daily_returns_cache.pkl")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

KEY = ["provider", "instance_type", "region", "is_spot", "gpu_type", "gpu_count", "vcpus", "ram_gb"]
EPS = 1e-9  # |log return| above this counts as a price change


def load_returns():
    """Daily log-return grid for every listing (same cleaning as analysis.py).

    Returns (R, meta, days):
      R    DataFrame (calendar day x series id) of log returns; NaN unless both that day and the
           previous calendar day have a price, so gaps never get bridged.
      meta DataFrame of KEY columns per series id.
    Log returns are scale-free, so $/instance vs $/GPU (and the fractional-GPU fix) doesn't matter.
    """
    if os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    df = pd.read_parquet(os.path.join(HERE, "all.parquet"))
    df = df[(df.quality == "ok") & (df.gpu_count > 0) & (df.price_per_hour > 0)
            & (df.gpu_type != "Unknown") & (df.provider != "datacrunch")].copy()
    df["day"] = df.timestamp.dt.tz_convert(None).dt.normalize()
    daily = (df.sort_values("timestamp").groupby(KEY + ["day"], dropna=False).price_per_hour.last()
             .reset_index())
    daily["sid"] = daily.groupby(KEY, dropna=False).ngroup()
    days = pd.date_range(daily.day.min(), daily.day.max(), freq="D")
    logp = np.log(daily.pivot(index="day", columns="sid", values="price_per_hour").reindex(days))
    R = logp.diff()
    meta = daily.drop_duplicates("sid").set_index("sid")[KEY]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pd.to_pickle((R, meta, days), CACHE)
    return R, meta, days


def shuffle_cols(X, rng):
    """Permute each column's observed values among its own observed positions (NaNs stay put).
    Keeps the gaps and each series' value distribution; destroys time ordering."""
    X = np.asarray(X, dtype=float)
    obs_first = np.argsort(np.isnan(X), axis=0, kind="stable")           # observed positions, in order
    rand_first = np.argsort(np.where(np.isnan(X), np.inf, rng.random(X.shape)), axis=0)  # observed, shuffled
    Y = np.full_like(X, np.nan)
    np.put_along_axis(Y, obs_first, np.take_along_axis(X, rand_first, axis=0), axis=0)
    return Y


def ar_filter(R, pmax=5):
    """Remove the predictable (trend/momentum) part of each series.

    Per column: fit AR(p) with intercept by OLS for p = 0..pmax on rows where r_t and all pmax
    lags exist (a common sample, so BIC compares like with like), pick p by BIC, return the
    residuals e_t (NaN elsewhere) and the chosen p."""
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


def aws_spot_active(R, meta, min_obs=100, min_move=0.3):
    """AWS spot listings with enough data and enough movement (used by clustering + GARCH)."""
    ids = meta.index[(meta.provider == "aws") & meta.is_spot]
    Ra = R[ids]
    n = Ra.notna().sum()
    return Ra.loc[:, (n >= min_obs) & ((Ra.abs() > EPS).sum() / n >= min_move)]


def spot_chained_indexes(R, meta, min_days=150, min_move=0.1):
    """Per-GPU spot matched-listing index returns (mean same-listing return, >=3 listings, no Vast.ai)."""
    long = R.stack().rename("r").reset_index()
    long.columns = ["day", "sid", "r"]
    long = long.join(meta, on="sid")
    long = long[(long.provider != "vastai") & long.is_spot]
    ch = long.groupby(["gpu_type", "day"]).r.agg(["mean", "size"])
    ch = ch[ch["size"] >= 3]["mean"].unstack("gpu_type").reindex(R.index)
    ch = ch.loc[:, ch.notna().sum() >= min_days]
    return ch.loc[:, (ch.abs() > EPS).sum() / ch.notna().sum() >= min_move]
