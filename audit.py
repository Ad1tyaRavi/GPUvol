"""Data-hygiene audit: mechanical artifacts that could masquerade as volatility or clustering.

Each check is read-only and writes a table to out/audit_*.csv. Nothing here changes the analyses;
it tells us which ones need a fix.

 A. Coverage gaps     - days a provider is missing, or only partly scraped (listing count collapses)
 B. Snapshot timing   - the "daily" price is the snapshot closest to 09:00 UTC; how far apart are
                        those picks, and does change frequency still depend on the interval?
 C. Intraday changes  - on days with 2+ snapshots, how often does a price differ within the day?
 D. Day-of-week       - scheduled repricing (e.g. GCP Mondays); chi-square test per provider/market
 E. Shared shock days - many providers changing at once: catalog churn (scraper) or uniform
                        repricing (one multiplier applied to everything)?
 F. Flicker/rounding  - tiny changes (<0.1%) and A->B->A reversals within a week
 G. Duplicate series  - listings with identical return paths (same price set for many regions);
                        effective number of independent series
 H. Spot pegged       - spot price a fixed fraction of on-demand, so spot carries no extra info
 I. Extreme jumps     - moves > 3x (or < 1/3x) and whether they reverse within 3 days
 J. Frozen days       - days on which AWS spot (which normally changes on most days) shows no change
                        at all: a stale feed, not a quiet market. Compare with common.STALE_WINDOWS.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from common import EPS, KEY, OUT, SAMPLE_HOUR_UTC, STALE_WINDOWS, load_raw, load_returns

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)

raw = load_raw()
raw["market"] = np.where(raw.is_spot, "spot", "on-demand")
R, meta, days = load_returns(clean=False)  # audit the raw data, before fixes
meta = meta.assign(market=np.where(meta.is_spot, "spot", "on-demand"))
CHG = (R.abs() > EPS).astype(float).where(R.notna())
report = {}


def save(name, df):
    df.to_csv(os.path.join(OUT, f"audit_{name}.csv"))
    return df


# ------------------------------------------------------------------ A. coverage
snap_days = raw.day.drop_duplicates().sort_values()
cnt = raw.groupby(["provider", "day"]).size().unstack("provider").reindex(snap_days)
med = cnt.median()
cov = pd.DataFrame({
    "days_present": cnt.notna().sum(),
    "days_missing": cnt.isna().sum(),
    "first_day": cnt.apply(lambda s: s.first_valid_index()).dt.date,
    "partial_days(<70% of median rows)": (cnt < 0.7 * med).sum(),
})
report["A"] = save("coverage", cov.sort_values("days_missing", ascending=False))

# ------------------------------------------------------------------ B. snapshot timing
snaps = raw.timestamp.dt.tz_convert(None).drop_duplicates().to_frame("ts")
snaps["day"] = snaps.ts.dt.normalize()
snaps["dist"] = ((snaps.ts - snaps.day).dt.total_seconds() / 3600 - SAMPLE_HOUR_UTC).abs()
last = snaps.sort_values(["dist", "ts"]).drop_duplicates("day").set_index("day").ts.sort_index()  # picked snapshot
interval_h = (last - last.shift()).dt.total_seconds() / 3600
interval_h = interval_h[(last.index.to_series().diff().dt.days == 1)]
timing = pd.DataFrame({"last_snapshot_hour_utc": last.dt.hour, "hours_since_prev_last": interval_h})
bucket = pd.cut(timing.hours_since_prev_last, [0, 18, 30, 60], labels=["<18h", "18-30h", ">30h"])
rows = []
for (prov, mk), ids in meta.groupby(["provider", "market"]).groups.items():
    c = CHG[ids]
    rate = (c.sum(1) / c.notna().sum(1)).where(c.notna().sum(1) > 0)
    if rate.notna().sum() < 50 or rate.mean() < 0.01:
        continue
    g = rate.groupby(bucket.reindex(rate.index)).mean()
    rows.append({"group": f"{prov} {mk}", **{f"chg_rate_{k}": v for k, v in g.items()}})
report["B_hours"] = timing.last_snapshot_hour_utc.value_counts().sort_index()
report["B_interval"] = timing.hours_since_prev_last.describe(percentiles=[.05, .25, .5, .75, .95]).round(1)
report["B"] = save("snapshot_interval_vs_change_rate", pd.DataFrame(rows).set_index("group").round(3))

# ------------------------------------------------------------------ C. intraday changes
multi = raw.groupby("day").timestamp.nunique()
multi_days = multi[multi >= 2].index
sub = raw[raw.day.isin(multi_days)]
intra = (sub.groupby(KEY + ["day", "market"], dropna=False).price_per_hour.nunique() > 1).rename("changed").reset_index()
report["C"] = save("intraday_changes", intra.groupby(["provider", "market"]).changed.agg(["mean", "size"])
                   .rename(columns={"mean": "share_listing_days_with_intraday_change", "size": "listing_days"})
                   .query("share_listing_days_with_intraday_change > 0").round(4)
                   .sort_values("share_listing_days_with_intraday_change", ascending=False))

# ------------------------------------------------------------------ D. day-of-week
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
rows = []
for (prov, mk), ids in meta.groupby(["provider", "market"]).groups.items():
    c = CHG[ids]
    ch, ob = c.sum(1), c.notna().sum(1)
    if ch.sum() < 30:
        continue
    by = pd.DataFrame({"chg": ch, "obs": ob}).groupby(c.index.dayofweek).sum()
    table = np.array([by.chg, by.obs - by.chg])
    p = chi2_contingency(table)[1] if (table > 0).all() else 0.0
    rate = (by.chg / by.obs).reindex(range(7))
    rows.append({"group": f"{prov} {mk}", **dict(zip(DOW, rate.round(3))), "max/min": rate.max() / max(rate.min(), 1e-9),
                 "chi2_p": p, "changes": int(ch.sum())})
report["D"] = save("day_of_week", pd.DataFrame(rows).set_index("group").sort_values("chi2_p"))

# ------------------------------------------------------------------ E. shared shock days
act = {}
for prov, ids in meta.groupby("provider").groups.items():
    obs = CHG[ids].notna().sum(1)
    act[prov] = (CHG[ids].sum(1) / obs).where(obs >= 5)
act = pd.DataFrame(act)
unusual = act.gt(act.quantile(0.95)) & act.gt(0)
shock_days = unusual.index[unusual.sum(1) >= 2]
rows = []
for d in shock_days:
    prev = d - pd.Timedelta(days=1)
    for prov in act.columns[unusual.loc[d]]:
        today = set(map(tuple, raw[(raw.provider == prov) & (raw.day == d)][KEY].astype(str).values))
        yday = set(map(tuple, raw[(raw.provider == prov) & (raw.day == prev)][KEY].astype(str).values))
        ids = meta.index[meta.provider == prov]
        r = R.loc[d, ids]
        moved = r[r.abs() > EPS]
        rows.append({"day": d.date(), "provider": prov, "share_changed": round(act.loc[d, prov], 3),
                     "listings_new": len(today - yday), "listings_gone": len(yday - today),
                     "listings_prev": len(yday),
                     "median_move_%": round(100 * np.expm1(moved.median()), 2) if len(moved) else np.nan,
                     "spread_of_moves_%": round(100 * moved.std(), 2) if len(moved) > 1 else np.nan,
                     "share_up": round((moved > 0).mean(), 2) if len(moved) else np.nan})
report["E"] = save("shock_days", pd.DataFrame(rows).set_index(["day", "provider"]))

# ------------------------------------------------------------------ F. flicker / rounding
rows = []
logp = R.cumsum()  # log price up to a per-series constant (gaps restart nothing; only differences used)
for (prov, mk), ids in meta.groupby(["provider", "market"]).groups.items():
    r = R[ids]
    moved = r.abs() > EPS
    n = int(moved.sum().sum())
    if n < 30:
        continue
    tiny = int((moved & (r.abs() < 1e-3)).sum().sum())
    # A->B->A: a change followed, within 7 days, by changes that sum back to exactly zero
    rev = 0
    X = r.to_numpy()
    for j in range(X.shape[1]):
        x = X[:, j]
        idx = np.where(np.abs(np.nan_to_num(x)) > EPS)[0]
        for i in idx:
            window = x[i + 1:i + 8]
            if np.isnan(window).any():
                window = window[:np.argmax(np.isnan(window))]
            cs = np.cumsum(np.nan_to_num(window))
            if len(cs) and (np.abs(cs + x[i]) < 1e-9).any() and (np.abs(window) > EPS).any():
                rev += 1
    rows.append({"group": f"{prov} {mk}", "changes": n, "share_tiny_<0.1%": tiny / n,
                 "share_reversed_within_7d": rev / n})
report["F"] = save("flicker", pd.DataFrame(rows).set_index("group").round(3).sort_values("share_reversed_within_7d", ascending=False))

# ------------------------------------------------------------------ G. duplicate series
rows = []
for (prov, mk), ids in meta.groupby(["provider", "market"]).groups.items():
    r = R[ids]
    r = r.loc[:, r.notna().sum() >= 100]
    active = r.loc[:, (r.abs() > EPS).sum() >= 5]
    if active.shape[1] < 2:
        continue
    sig = active.round(6).fillna(999).T.apply(lambda row: hash(tuple(row)), axis=1)
    rows.append({"group": f"{prov} {mk}", "active_listings": active.shape[1],
                 "unique_return_paths": sig.nunique(),
                 "largest_identical_group": int(sig.value_counts().iloc[0])})
dup = pd.DataFrame(rows).set_index("group")
dup["duplication_factor"] = (dup.active_listings / dup.unique_return_paths).round(2)
report["G"] = save("duplicate_series", dup.sort_values("duplication_factor", ascending=False))

# ------------------------------------------------------------------ H. spot pegged to on-demand
pair_key = [k for k in KEY if k != "is_spot"]
d = raw.groupby(pair_key + ["day", "is_spot"], dropna=False).price_per_hour.last().unstack("is_spot").dropna()
d.columns = ["od", "spot"]
d["ratio"] = (d.spot / d.od).round(4)
pr = d.groupby(pair_key, dropna=False).ratio.agg(["nunique", "size", "median"]).reset_index()
pr = pr[pr["size"] >= 30]
report["H"] = save("spot_pegging", pr.groupby("provider").agg(
    pairs=("nunique", "size"), share_constant_ratio=("nunique", lambda x: (x == 1).mean()),
    median_ratio=("median", "median")).round(3).sort_values("share_constant_ratio", ascending=False))

# ------------------------------------------------------------------ I. extreme jumps
big = R.abs() > np.log(3)
rows = []
for sid in big.columns[big.any()]:
    x = R[sid]
    for dday in x.index[big[sid]]:
        nxt = x.loc[dday + pd.Timedelta(days=1): dday + pd.Timedelta(days=3)].dropna()
        rows.append({"provider": meta.loc[sid, "provider"], "market": meta.loc[sid, "market"],
                     "gpu": meta.loc[sid, "gpu_type"], "day": dday.date(), "move_x": round(float(np.exp(x[dday])), 2),
                     "reversed_within_3d": bool((np.abs(nxt.cumsum() + x[dday]) < 0.05).any())})
ext = pd.DataFrame(rows)
report["I"] = save("extreme_jumps", ext.groupby(["provider", "market"]).agg(
    jumps=("move_x", "size"), share_reversed=("reversed_within_3d", "mean"),
    median_move_x=("move_x", "median")).round(2) if len(ext) else ext)

# ------------------------------------------------------------------ J. frozen days
aws = meta.index[(meta.provider == "aws") & meta.is_spot]
aws_rate = CHG[aws].mean(1)
frozen = aws_rate[aws_rate == 0].index
known = pd.DatetimeIndex([d for _, a, b in STALE_WINDOWS for d in pd.date_range(a, b)])
rows = []
for d in frozen:
    rates = {p: CHG[ids].loc[d].mean() for p, ids in meta.groupby("provider").groups.items()
             if CHG[ids].loc[d].notna().sum() >= 5}
    rows.append({"day": d.date(), "in_STALE_WINDOWS": d in known,
                 "providers_with_changes": ", ".join(f"{p} {v:.2f}" for p, v in sorted(rates.items()) if v > 0)})
report["J"] = save("frozen_days", pd.DataFrame(rows).set_index("day") if rows else pd.DataFrame())

for k, v in report.items():
    print(f"\n===== {k} =====\n{v}")
