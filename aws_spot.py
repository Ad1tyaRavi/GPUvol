"""AWS spot, the one market in the data that reprices daily: four results.

 A. Momentum     - does today's move predict the next week? Pooled slope of the next-7-day return
                   on today's return (days with a change), by size of today's move, and out of
                   sample (listings picked and slope fit on data through August, tested on
                   September onward). 95% intervals resample whole 14-day blocks of days.
 B. Local markets- do listings move together? Median pairwise correlation of returns (daily and
                   weekly) for pairs sharing an instance type across regions, sharing a region
                   across chips, or sharing neither. Null: circularly shift every listing by its
                   own random whole number of weeks (keeps each series' autocorrelation and
                   weekday pattern, destroys alignment between listings), 30 times.
 C. Timing vs size - "clustering" split into whether a price changes (quiet days come in runs?)
                   and how big the changes are (corr of log|move| on consecutive change days,
                   raw and AR-filtered, per listing). Both are compared with shuffles of each
                   listing's own values: a plain shuffle and a weekday x 30-day blocked one.
 D. On-demand ceiling - spot never exceeds on-demand for the same instance/region. How often is it
                   at that ceiling, and how much of the quiet time does it explain? (C reruns
                   the timing test without at-ceiling days.)

Same sampling and stale-feed fixes as the other scripts (common.py); universe for A-C is
common.aws_spot_active (>= 100 daily returns, moves on >= 30% of days, identical paths collapsed).
Outputs: out/aws_*.csv.
"""
import os

import numpy as np
import pandas as pd

from common import (EPS, KEY, OUT, ar_filter, aws_spot_active, load_raw, load_returns, pick_daily,
                    shuffle_cols, shuffle_cols_blocked, stale_rows)

rng = np.random.default_rng(4)
R, meta, days = load_returns()
A = aws_spot_active(R, meta)
m = meta.loc[A.columns]
X = A.to_numpy()
print(f"AWS spot listings: {A.shape[1]}")

# ------------------------------------------------------------------ A. momentum
H = 7
SPLIT = "2026-09-01"


def today_vs_next(F):
    """(today's change, next-7-day change, day) on days with a change and all 7 next days observed."""
    Z = F.to_numpy()
    fwd = np.full_like(Z, np.nan)
    for t in range(len(Z) - H):
        w = Z[t + 1:t + 1 + H]
        fwd[t] = np.where(np.isnan(w).any(0), np.nan, w.sum(0))
    d = np.repeat(F.index.values[:, None], Z.shape[1], axis=1).ravel()
    x, y = Z.ravel(), fwd.ravel()
    k = ~np.isnan(x) & ~np.isnan(y) & (np.abs(x) > EPS)
    return x[k], y[k], d[k]


x, y, d = today_vs_next(A)


def fit(xx, yy):
    b = (xx * yy).sum() / (xx * xx).sum()
    return {"slope": b, "r2": 1 - ((yy - b * xx) ** 2).sum() / ((yy - yy.mean()) ** 2).sum(),
            "same_direction": (np.sign(yy) == np.sign(xx)).mean(), "n": len(xx)}


# 95% intervals by resampling whole 14-day blocks of days. Rows are not independent: next-7-day
# windows of nearby days overlap, and listings move together on the same day (B below). Keeping
# every listing-day of a block together covers both; 14 days is twice the forecast window.
BLOCK_DAYS, N_BOOT, MIN_BLOCKS = 14, 1000, 10
boot_rng = np.random.default_rng(14)  # own stream, so B and C below draw the same shuffles as before


def block_ci(xx, yy, dd, b=None):
    """Intervals for the slope and hit rate; with a fixed slope b, also for out-of-sample R^2.
    Every statistic is a ratio of sums, so each resample is a weighted sum of per-block totals."""
    blk = np.unique((dd - dd.min()) // np.timedelta64(BLOCK_DAYS, "D"), return_inverse=True)[1]
    k = blk.max() + 1
    tot = lambda v: np.bincount(blk, v, minlength=k)
    S = {"n": tot(np.ones_like(xx)), "xy": tot(xx * yy), "xx": tot(xx * xx), "y": tot(yy), "yy": tot(yy * yy),
         "hit": tot((np.sign(yy) == np.sign(xx)).astype(float))}
    if b is not None:
        S["sse"] = tot((yy - b * xx) ** 2)
    W = np.stack([np.bincount(boot_rng.integers(0, k, k), minlength=k) for _ in range(N_BOOT)])
    s = {key: W @ v for key, v in S.items()}
    stats = {"slope": s["xy"] / s["xx"], "same_direction": s["hit"] / s["n"]}
    if b is not None:
        stats["r2"] = 1 - s["sse"] / (s["yy"] - s["y"] ** 2 / s["n"])
    out = {}
    for key, v in stats.items():
        # With only a few blocks a resample can't vary much, so the interval comes out far too
        # narrow (Sep-Nov has 4). Report none below MIN_BLOCKS; the draws still happen so the
        # random stream, and every other interval, stays the same.
        lo, hi = np.percentile(v, [2.5, 97.5]) if k >= MIN_BLOCKS else (np.nan, np.nan)
        out[f"{key}_p025"], out[f"{key}_p975"] = lo, hi
    out["blocks"] = k
    return out


# Out of sample: listings chosen, and the slope fitted, on data before September only; tested on
# those same listings from September onward (no hindsight in which listings are tested).
xo, yo, do = today_vs_next(aws_spot_active(R, meta, until=pd.Timestamp(SPLIT) - pd.Timedelta(days=1)))
# Training rows need their whole next-7-day window before the split, so no test-period prices
# leak into the training slope.
train, test = do + np.timedelta64(H, "D") < np.datetime64(SPLIT), do >= np.datetime64(SPLIT)
b_tr = fit(xo[train], yo[train])["slope"]
# "slope" here is the slope refit on the test period itself, so it can be compared with the
# training slope; r2 is out of sample (training slope applied to test rows).
xt, yt = xo[test], yo[test]
oos = {"slope": fit(xt, yt)["slope"], "train_slope": b_tr,
       "r2": 1 - ((yt - b_tr * xt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum(),
       "same_direction": (np.sign(yt) == np.sign(xt)).mean(), "n": int(test.sum())}
tr_ci = block_ci(xo[train], yo[train], do[train])
oos.update(block_ci(xt, yt, do[test], b=b_tr), train_slope_p025=tr_ci["slope_p025"],
           train_slope_p975=tr_ci["slope_p975"])
rows = {"all": {**fit(x, y), **block_ci(x, y, d)}, "out_of_sample_sep_nov": oos}
q = pd.qcut(np.abs(x), 4, labels=["smallest", "2nd", "3rd", "largest"])
for lab in q.categories:
    s = np.asarray(q == lab)
    rows[f"today_{lab}_quarter"] = {**fit(x[s], y[s]), **block_ci(x[s], y[s], d[s]),
                                    "median_abs_today_pct": 100 * np.median(np.abs(x[s]))}
mom = pd.DataFrame(rows).T
mom.round(4).to_csv(os.path.join(OUT, "aws_momentum.csv"))
# binned means for the README chart: 20 quantile bins of today's signed move
bins = pd.qcut(x, 20, duplicates="drop")
pd.DataFrame({"today": x, "next7": y, "bin": bins}).groupby("bin", observed=True)[["today", "next7"]].mean() \
    .round(5).to_csv(os.path.join(OUT, "aws_momentum_bins.csv"), index=False)
ac = {}
Xc = X - np.nanmean(X, 0)
for k in range(1, 57):
    a, b = Xc[k:], Xc[:-k]
    mm = ~np.isnan(a) & ~np.isnan(b)
    a, b = np.where(mm, a, 0), np.where(mm, b, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))
    ac[k] = np.nanmedian(np.where(mm.sum(0) >= 40, r, np.nan))
pd.Series(ac, name="median_acf_r").rename_axis("lag").round(4).to_csv(os.path.join(OUT, "aws_return_acf.csv"))

# ------------------------------------------------------------------ B. local markets
inst, reg, gpu = m.instance_type.to_numpy(), m.region.to_numpy(), m.gpu_type.to_numpy()
same_i, same_r, same_g = (v[:, None] == v[None, :] for v in (inst, reg, gpu))
GROUPS = {"same instance type, different region": same_i & ~same_r,
          "same region, different chip": same_r & ~same_g,
          "different chip, different region": ~same_g & ~same_r}


def group_medians(F, h):
    """Median pairwise corr per group, of h-day summed returns (>= 5 of 7 days observed for h=7)."""
    if h > 1:
        F = F.groupby((F.index - F.index[0]).days // h).sum(min_count=5)
    C = F.corr(min_periods=60 if h == 1 else 12).to_numpy()
    iu = np.triu(np.ones_like(C, bool), 1) & ~np.isnan(C)
    return {g: np.median(C[iu & mk]) for g, mk in GROUPS.items()}, {g: int((iu & mk).sum()) for g, mk in GROUPS.items()}


rows = []
for h in (1, 7):
    act, npairs = group_medians(A, h)
    null = []
    for _ in range(30):
        T = len(A)
        sh = pd.DataFrame({c: np.roll(A[c].to_numpy(), 7 * rng.integers(1, T // 7)) for c in A.columns}, index=A.index)
        null.append(group_medians(sh, h)[0])
    null = pd.DataFrame(null)
    for g in GROUPS:
        rows.append({"horizon_days": h, "pairs": g, "n_pairs": npairs[g], "median_corr": act[g],
                     "null_p025": null[g].quantile(0.025), "null_p975": null[g].quantile(0.975)})
local = pd.DataFrame(rows)
local["beats_null"] = local.median_corr > local.null_p975
local.round(4).to_csv(os.path.join(OUT, "aws_comovement.csv"), index=False)

# ------------------------------------------------------------------ on-demand ceiling (data)
# Spot never goes above on-demand for the same instance/region. Built here because C uses it.
df = load_raw(columns=KEY + ["price_per_hour"])
df = pick_daily(df[(df.provider == "aws") & ~stale_rows(df.provider, df.day)], KEY)
pair = [c for c in KEY if c != "is_spot"]
p = df.pivot_table(index=pair + ["day"], columns="is_spot", values="price_per_hour").dropna()
p.columns = ["od", "spot"]
p["at_ceiling"] = p.spot >= p.od - 1e-9
# day x listing grid aligned with A: True where that listing's spot price sat at on-demand
cg = m[pair].reset_index().merge(p.at_ceiling.reset_index(), on=pair)
sid_col = m.index.name or "index"
CEIL = (cg.pivot(index="day", columns=sid_col, values="at_ceiling").reindex(index=A.index, columns=A.columns)
        .fillna(False).astype(bool).to_numpy())

# ------------------------------------------------------------------ C. timing vs size
I = np.where(np.isnan(X), np.nan, (np.abs(X) > EPS).astype(float))
quiet_runs = []
for j in range(I.shape[1]):
    n = 0
    for v in I[:, j]:
        if v == 0:
            n += 1
        else:
            if n:
                quiet_runs.append(n)
            n = 0
quiet_runs = np.array(quiet_runs)


def size_corr(Z, chg):
    """Per listing: corr of log|move| on consecutive change days (NaN if < 30 such pairs)."""
    out = np.full(Z.shape[1], np.nan)
    for j in range(Z.shape[1]):
        z, c = Z[:, j], chg[:, j]
        u, v = z[1:], z[:-1]
        k = c[1:] & c[:-1] & ~np.isnan(u) & ~np.isnan(v) & (u != 0) & (v != 0)
        if k.sum() >= 30:
            out[j] = np.corrcoef(np.log(np.abs(u[k])), np.log(np.abs(v[k])))[0, 1]
    return out


def transitions(I):
    """Pooled P(change | change yesterday), P(change | quiet yesterday)."""
    a, b = I[1:], I[:-1]
    mm = ~np.isnan(a) & ~np.isnan(b)
    return (a * b)[mm].sum() / b[mm].sum(), (a * (1 - b))[mm].sum() / (1 - b)[mm].sum()


E = ar_filter(A)[0].to_numpy()
chg = np.abs(np.nan_to_num(X)) > EPS
after_chg, after_quiet = transitions(I)
# Null for the timing result: the same blocked shuffle of each listing's change days. Pooled
# transition rates mix listings with different change rates, so even shuffled data shows runs.
null_t = np.array([transitions(shuffle_cols_blocked(I, rng, A.index)) for _ in range(100)])
# Same test without the ceiling: a price stuck at on-demand can't rise, so it sits still for days
# by construction. Drop every return whose day, or the day before, is at the ceiling.
at_cap = CEIL | np.vstack([np.zeros((1, CEIL.shape[1]), bool), CEIL[:-1]])
I_ex = np.where(at_cap, np.nan, I)
after_chg_ex, after_quiet_ex = transitions(I_ex)
null_ex = np.array([transitions(shuffle_cols_blocked(I_ex, rng, A.index)) for _ in range(100)])
timing = {"p_change": np.nanmean(I),
          "p_change_after_change": after_chg,
          "p_change_after_quiet_day": after_quiet,
          "p_change_after_quiet_day_null_median": np.median(null_t[:, 1]),
          "p_change_after_quiet_day_null_p025": np.percentile(null_t[:, 1], 2.5),
          "ex_ceiling_share_returns_dropped": 1 - np.isfinite(I_ex).sum() / np.isfinite(I).sum(),
          "ex_ceiling_p_change": np.nanmean(I_ex),
          "ex_ceiling_p_change_after_change": after_chg_ex,
          "ex_ceiling_p_change_after_quiet_day": after_quiet_ex,
          "ex_ceiling_p_change_after_quiet_day_null_median": np.median(null_ex[:, 1]),
          "ex_ceiling_p_change_after_quiet_day_null_p025": np.percentile(null_ex[:, 1], 2.5),
          "quiet_runs": len(quiet_runs),
          "share_quiet_days_in_runs_ge3": quiet_runs[quiet_runs >= 3].sum() / quiet_runs.sum()}
# Size correlation is measured on change days only. The nulls shuffle only those days' values
# (raw moves, or AR residuals on the same days), so actual and null use the same days.
# Two nulls: a plain shuffle (no clustering at any horizon) and the weekday x 30-day blocked
# shuffle (keeps month-level volatility regimes, so it only tests clustering within a month).
# Per listing, "beats" = its own correlation is above the 95th percentile of its own 100 shuffles
# (5% of listings would beat it by chance).
for name, Z in [("raw", X), ("ar_filtered", E)]:
    Zc = np.where(chg, Z, np.nan)
    nulls = {"plain": [], "blocked": []}
    for _ in range(100):
        for kind, S in [("plain", shuffle_cols(Zc, rng)), ("blocked", shuffle_cols_blocked(Zc, rng, A.index))]:
            nulls[kind].append(size_corr(S, ~np.isnan(S)))
    actual = size_corr(Z, chg)
    timing[f"size_corr_{name}"] = np.nanmedian(actual)
    timing[f"size_corr_{name}_listings"] = np.isfinite(actual).sum()
    for kind, nl in nulls.items():
        nl = np.array(nl)  # (shuffles, listings)
        timing[f"size_corr_{name}_{kind}_null_median"] = np.median(np.nanmedian(nl, axis=1))
        timing[f"size_corr_{name}_{kind}_null_p975"] = np.percentile(np.nanmedian(nl, axis=1), 97.5)
        ok = np.isfinite(actual)
        timing[f"size_corr_{name}_share_beats_{kind}_null"] = (actual[ok] > np.nanpercentile(nl[:, ok], 95, axis=0)).mean()
timing = pd.Series(timing)
timing.round(4).to_csv(os.path.join(OUT, "aws_timing_vs_size.csv"), header=["value"])

# ------------------------------------------------------------------ D. on-demand ceiling
sp = df[df.is_spot].sort_values(KEY + ["day"])
g = sp.groupby(KEY, dropna=False)
sp = sp.assign(consec=sp.day - g.day.shift() == pd.Timedelta(days=1),
               chg=sp.price_per_hour != g.price_per_hour.shift())
sp = sp[sp.consec].merge(p.at_ceiling.reset_index(), on=pair + ["day"], how="inner")
ceil = pd.Series({"max_spot_over_on_demand": (p.spot / p.od).max(),
                  "share_days_at_ceiling": sp.at_ceiling.mean(),
                  "p_change_at_ceiling": sp[sp.at_ceiling].chg.mean(),
                  "p_change_below_ceiling": sp[~sp.at_ceiling].chg.mean(),
                  "share_quiet_days_at_ceiling": sp[~sp.chg].at_ceiling.mean()})
ceil.round(4).to_csv(os.path.join(OUT, "aws_ceiling.csv"), header=["value"])
by_gpu = p.reset_index().groupby("gpu_type").agg(listing_days=("spot", "size"),
                                                  median_spot_over_od=("spot", lambda s: np.nan),
                                                  share_at_ceiling=("at_ceiling", "mean"))
by_gpu["median_spot_over_od"] = (p.spot / p.od).groupby(level="gpu_type").median()
by_gpu.round(3).sort_values("share_at_ceiling", ascending=False).to_csv(os.path.join(OUT, "aws_ceiling_by_gpu.csv"))

pd.set_option("display.width", 200, "display.max_columns", 20)
print("\n== A. momentum (next 7 days vs today) ==\n", mom.round(3))
print("\n   median return ACF:", {k: round(v, 3) for k, v in ac.items() if k in (1, 2, 3, 5, 7, 14, 28)})
print("\n== B. co-movement ==\n", local.round(3).to_string(index=False))
print("\n== C. timing vs size ==\n", timing.round(3).to_string())
print("\n== D. on-demand ceiling ==\n", ceil.round(3).to_string())
print(by_gpu.round(3).sort_values("share_at_ceiling", ascending=False).head(8))
