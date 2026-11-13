"""Shared helpers: chart style, raw loading with daily sampling, data-hygiene fixes, cleaned daily
return grid, AR filter, fast within-series shuffle."""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "all.parquet")
CACHE = os.path.join(HERE, "data", "daily_returns_cache.pkl")

# --- daily sampling ----------------------------------------------------------------------------
# The scraper runs 1-3 times a day at varying times (09:00 and 21:00 on a schedule, plus runs at
# ~10:00-15:00). Taking the day's *last* snapshot made a "daily" return span anything from ~10h to
# ~40h, and AWS spot changes far more often over the long ones. Each listing-day instead uses the
# snapshot closest to 09:00 UTC (Jan-Mar snapshots were all ~09:20), so returns are ~24h apart.
SAMPLE_HOUR_UTC = 9.0

# --- data-hygiene settings (see audit.py) ------------------------------------------------------
# Stale feed: windows where a provider's prices are frozen copies of the last good scrape. Each is
# (providers, first frozen day, last frozen day); providers=None means every provider except
# STALE_EXEMPT. The day after a window is its catch-up day: the return then spans the whole window.
# Frozen days are treated as missing; the catch-up return is kept for price trends (it is where
# prices really went) but dropped for volatility/clustering (it is weeks of moves, not one day).
#  * 2026-05-08..05-24: right after the dataset's two-month gap, no listing at any provider except
#    Vast.ai changed on any snapshot (AWS spot normally changes on ~40% of snapshots); on 05-25 97%
#    of AWS, 82% of GCP, 86% of Verda spot and every RunPod spot listing changed at once.
#  * 2026-06-01: the day's only snapshot (14:54) is an exact copy of 05-31 21:00, Vast.ai aside.
#  * 2026-06-25..07-22: every AWS spot listing made its last change on 06-24 and nothing moved until
#    07-23, when 97% changed at once; GCP skipped all its Monday batches in the same window and
#    caught up on Thursday 07-23. Other providers kept updating.
STALE_EXEMPT = ("vastai",)
STALE_WINDOWS = [
    (None, pd.Timestamp("2026-05-08"), pd.Timestamp("2026-05-24")),
    (None, pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-01")),
    (("aws", "gcp"), pd.Timestamp("2026-06-25"), pd.Timestamp("2026-07-22")),
]
# GPUVOL_KEEP_SUSPECT=1 skips the stale-feed fixes (raw data, for a sensitivity comparison);
# outputs then go to out/with_suspect_days/.
KEEP_SUSPECT = os.environ.get("GPUVOL_KEEP_SUSPECT") == "1"
# Nebius spot switched from near-static to daily dynamic pricing on this day; the two regimes are
# split into separate series so a regime switch can't pass as clustering.
NEBIUS_REGIME_START = pd.Timestamp("2026-10-08")
POST_OFFSET = 10_000_000  # series-id offset for the post-switch Nebius series

OUT = os.path.join(HERE, "out", "with_suspect_days") if KEEP_SUSPECT else os.path.join(HERE, "out")
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

TIER = {  # rough generation / segment buckets
    **dict.fromkeys(["B200", "B300", "GB200", "GB300"], "Blackwell DC"),
    **dict.fromkeys(["H100", "H200", "H100NVL", "H200NVL", "GH200", "H800"], "Hopper DC"),
    **dict.fromkeys(["A100", "A800", "A10", "A10G", "A30", "A40", "A16"], "Ampere DC"),
    **dict.fromkeys(["L4", "L40", "L40S"], "Ada DC"),
    **dict.fromkeys(["V100", "P100", "P4", "T4", "K80", "M60"], "Legacy DC"),
    **dict.fromkeys(["RTX4090", "RTX5090", "RTX3090", "RTX4080", "RTX5080", "A6000",
                     "RTX6000Ada", "RTXPRO6000", "RTXPRO6000S", "RTXPRO4500", "A5000", "A4000"], "Workstation/consumer"),
}
TIER_OTHER = "Other (TPU, misc)"


# ------------------------------------------------------------------ raw data
def _snapshot_files():
    return glob.glob(os.path.join(HERE, "data", "prices", "*", "*.parquet"))


def _ensure_src():
    """(Re)build all.parquet from data/prices/ when it is missing or older than any snapshot file."""
    files = _snapshot_files()
    if files and (not os.path.exists(SRC) or max(map(os.path.getmtime, files)) > os.path.getmtime(SRC)):
        pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).to_parquet(SRC)


def load_raw(columns=None):
    """All snapshot rows with the basic quality filters, plus a naive-UTC `day` column.
    all.parquet is (re)built from data/prices/ when missing or older than any snapshot file."""
    _ensure_src()
    if columns is not None:
        need = ["quality", "gpu_count", "price_per_hour", "gpu_type", "provider", "timestamp"]
        columns = list(dict.fromkeys(list(columns) + need))
    df = pd.read_parquet(SRC, columns=columns)
    df = df[(df.quality == "ok") & (df.gpu_count > 0) & (df.price_per_hour > 0) & (df.gpu_type != "Unknown")
            # DataCrunch rebranded to Verda; its May-Jul rows duplicate Verda's catalog.
            & (df.provider != "datacrunch")].copy()
    df["day"] = df.timestamp.dt.tz_convert(None).dt.normalize()
    return df


def pick_daily(df, keys):
    """One row per (keys, day): the snapshot closest to SAMPLE_HOUR_UTC (ties -> earlier)."""
    hours = (df.timestamp.dt.tz_convert(None) - df.day).dt.total_seconds() / 3600
    d = df.assign(_dist=(hours - SAMPLE_HOUR_UTC).abs(), _ts=df.timestamp)
    d = d.sort_values(["_dist", "_ts"], kind="stable").drop_duplicates(keys + ["day"])
    return d.drop(columns=["_dist", "_ts"])


def stale_rows(provider, day):
    """Boolean mask for rows inside a stale-feed window (provider/day Series)."""
    mask = pd.Series(False, index=provider.index)
    if KEEP_SUSPECT:
        return mask
    for provs, start, end in STALE_WINDOWS:
        p = ~provider.isin(STALE_EXEMPT) if provs is None else provider.isin(provs)
        mask |= p & day.between(start, end)
    return mask


def stale_spans():
    """(providers, start, end) for charts; providers=None means all but STALE_EXEMPT."""
    return [] if KEEP_SUSPECT else STALE_WINDOWS


# ------------------------------------------------------------------ return grid
def load_returns(clean=True, keep_catchup=False):
    """Daily log-return grid for every listing (same sampling and cleaning as analysis.py).

    Returns (R, meta, days):
      R    DataFrame (calendar day x series id) of log returns; NaN unless both that day and the
           previous calendar day have a price, so gaps never get bridged.
      meta DataFrame of KEY columns per series id.
    Log returns are scale-free, so $/instance vs $/GPU (and the fractional-GPU fix) doesn't matter.
    clean=True applies the audit fixes: blank the stale-feed windows, split Nebius at its regime
    switch. keep_catchup=True keeps the catch-up return after each stale window (for trends).
    """
    R, meta, days = _load_raw_returns()
    if clean:
        R = blank_stale(R, meta, keep_catchup)
        R, meta = split_nebius(R, meta)
    return R, meta, days


def blank_stale(R, meta, keep_catchup=False):
    """NaN out returns on frozen days (and, unless keep_catchup, the catch-up day after)."""
    R = R.copy()
    if KEEP_SUSPECT:
        return R
    for provs, start, end in STALE_WINDOWS:
        p = ~meta.provider.isin(STALE_EXEMPT) if provs is None else meta.provider.isin(provs)
        cols = meta.index[p]
        R.loc[start:end, cols] = np.nan
        if not keep_catchup:
            R.loc[end + pd.Timedelta(days=1):end + pd.Timedelta(days=1), cols] = np.nan
    return R


def split_nebius(R, meta):
    """Nebius before/after its 2026-10-08 switch become separate series ('nebius_static' and
    'nebius_dynamic'). The return on the switch day itself (old regime -> new) is dropped."""
    nb = meta.index[meta.provider == "nebius"]
    before = R.index < NEBIUS_REGIME_START
    post = R[nb].copy()
    post.loc[R.index <= NEBIUS_REGIME_START] = np.nan
    post.columns = post.columns + POST_OFFSET
    R = R.copy()
    R.loc[~before, nb] = np.nan
    meta = meta.copy()
    meta_post = meta.loc[nb].copy()
    meta_post.index = meta_post.index + POST_OFFSET
    meta.loc[nb, "provider"] = "nebius_static"
    meta_post["provider"] = "nebius_dynamic"
    return pd.concat([R, post], axis=1), pd.concat([meta, meta_post])


def dedupe_paths(R):
    """Keep one column per identical return path (same changes on the same days, same gaps).
    GCP and Verda set one price for many regions/node sizes; counting each copy as a separate
    series inflates 'share of series significant' and listing-weighted averages.
    Only listings that changed at least once are collapsed: listings that never move all share the
    same all-zero path, but a fixed price is a real property of each product, not a copied decision."""
    if R.shape[1] < 2:
        return R
    moves = (R.abs() > EPS).any().to_numpy()
    sig = R.round(6).fillna(9.0).T.apply(lambda row: hash(row.to_numpy().tobytes()), axis=1)
    return R.loc[:, ~(sig.duplicated().to_numpy() & moves)]


def _load_raw_returns():
    _ensure_src()
    sig = (os.path.getmtime(SRC), os.path.getsize(SRC), SAMPLE_HOUR_UTC)
    if os.path.exists(CACHE):
        cached = pd.read_pickle(CACHE)
        if len(cached) == 4 and cached[3] == sig:
            return cached[:3]
    df = load_raw(columns=KEY + ["price_per_hour", "timestamp", "quality"])
    daily = pick_daily(df, KEY)[KEY + ["day", "price_per_hour"]]
    daily["sid"] = daily.groupby(KEY, dropna=False).ngroup()
    days = pd.date_range(daily.day.min(), daily.day.max(), freq="D")
    logp = np.log(daily.pivot(index="day", columns="sid", values="price_per_hour").reindex(days))
    R = logp.diff()
    meta = daily.drop_duplicates("sid").set_index("sid")[KEY]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pd.to_pickle((R, meta, days, sig), CACHE)
    return R, meta, days


# ------------------------------------------------------------------ stats helpers
def shuffle_cols(X, rng):
    """Permute each column's observed values among its own observed positions (NaNs stay put).
    Keeps the gaps and each series' value distribution; destroys time ordering."""
    X = np.asarray(X, dtype=float)
    obs_first = np.argsort(np.isnan(X), axis=0, kind="stable")           # observed positions, in order
    rand_first = np.argsort(np.where(np.isnan(X), np.inf, rng.random(X.shape)), axis=0)  # observed, shuffled
    Y = np.full_like(X, np.nan)
    np.put_along_axis(Y, obs_first, np.take_along_axis(X, rand_first, axis=0), axis=0)
    return Y


def shuffle_cols_blocked(X, rng, index, block_days=30):
    """Stricter null: shuffle each column only among days with the same weekday inside the same
    30-day block. Keeps weekly schedules (GCP Monday batches, Verda's quiet weekends) and slow
    regime changes in the null, so only short-horizon timing within a month counts as clustering."""
    X = np.asarray(X, dtype=float)
    Y = np.full_like(X, np.nan)
    idx = pd.DatetimeIndex(index)
    grp = idx.dayofweek.to_numpy() + 7 * ((idx - idx[0]).days.to_numpy() // block_days)
    for g in np.unique(grp):
        rows = np.where(grp == g)[0]
        Y[rows] = shuffle_cols(X[rows], rng)
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
    return dedupe_paths(Ra.loc[:, (n >= min_obs) & ((Ra.abs() > EPS).sum() / n >= min_move)])


def spot_chained_indexes(R, meta, min_days=150, min_move=0.1):
    """Per-GPU spot matched-listing index returns (mean same-listing return, >=3 listings, no Vast.ai).
    Duplicate price paths are collapsed first so one region-wide price isn't counted many times."""
    ids = meta.index[meta.is_spot & (meta.provider != "vastai")]
    R = dedupe_paths(R[ids])
    long = R.stack().rename("r").reset_index()
    long.columns = ["day", "sid", "r"]
    long = long.join(meta, on="sid")
    ch = long.groupby(["gpu_type", "day"]).r.agg(["mean", "size"])
    ch = ch[ch["size"] >= 3]["mean"].unstack("gpu_type").reindex(R.index)
    ch = ch.loc[:, ch.notna().sum() >= min_days]
    return ch.loc[:, (ch.abs() > EPS).sum() / ch.notna().sum() >= min_move]
