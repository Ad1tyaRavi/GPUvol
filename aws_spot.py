"""AWS spot, the one market in the data that reprices daily: four results.

 A. Momentum     - does today's move predict the next week? Pooled slope of the next-7-day return
                   on today's return (days with a change), by size of today's move, and out of
                   sample (fit through August, test September onward).
 B. Local markets- do listings move together? Median pairwise correlation of returns (daily and
                   weekly) for pairs sharing an instance type across regions, sharing a region
                   across chips, or sharing neither. Null: circularly shift every listing by its
                   own random whole number of weeks (keeps each series' autocorrelation and
                   weekday pattern, destroys alignment between listings), 30 times.
 C. Timing vs size - "clustering" split into whether a price changes (quiet days come in runs?)
                   and how big the changes are (corr of log|move| on consecutive change days,
                   raw and AR-filtered, vs a weekday x 30-day blocked shuffle).
 D. On-demand ceiling - spot never exceeds on-demand for the same instance/region. How often is
                   it at that ceiling, and how much of the quiet time does the ceiling explain?

Same sampling and stale-feed fixes as the other scripts (common.py); universe for A-C is
common.aws_spot_active (>= 100 daily returns, moves on >= 30% of days, identical paths collapsed).
Outputs: out/aws_*.csv.
"""
import os

import numpy as np
import pandas as pd

from common import (EPS, KEY, OUT, ar_filter, aws_spot_active, load_raw, load_returns, pick_daily,
                    shuffle_cols_blocked, stale_rows)

rng = np.random.default_rng(4)
R, meta, days = load_returns()
A = aws_spot_active(R, meta)
m = meta.loc[A.columns]
X = A.to_numpy()
print(f"AWS spot listings: {A.shape[1]}")

# ------------------------------------------------------------------ A. momentum
H = 7
fwd = np.full_like(X, np.nan)
for t in range(len(X) - H):
    w = X[t + 1:t + 1 + H]
    fwd[t] = np.where(np.isnan(w).any(0), np.nan, w.sum(0))  # next 7 days, only if all observed
day_of = np.repeat(A.index.values[:, None], X.shape[1], axis=1).ravel()
x, y, d = X.ravel(), fwd.ravel(), day_of
keep = ~np.isnan(x) & ~np.isnan(y) & (np.abs(x) > EPS)
x, y, d = x[keep], y[keep], d[keep]


def fit(xx, yy):
    b = (xx * yy).sum() / (xx * xx).sum()
    return {"slope": b, "r2": 1 - ((yy - b * xx) ** 2).sum() / ((yy - yy.mean()) ** 2).sum(),
            "same_direction": (np.sign(yy) == np.sign(xx)).mean(), "n": len(xx)}


train, test = d < np.datetime64("2026-09-01"), d >= np.datetime64("2026-09-01")
b_tr = fit(x[train], y[train])["slope"]
oos = {"slope": b_tr, "r2": 1 - ((y[test] - b_tr * x[test]) ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum(),
       "same_direction": (np.sign(y[test]) == np.sign(x[test])).mean(), "n": int(test.sum())}
rows = {"all": fit(x, y), "out_of_sample_sep_nov": oos}
q = pd.qcut(np.abs(x), 4, labels=["smallest", "2nd", "3rd", "largest"])
for lab in q.categories:
    s = np.asarray(q == lab)
    rows[f"today_{lab}_quarter"] = {**fit(x[s], y[s]), "median_abs_today_pct": 100 * np.median(np.abs(x[s]))}
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

# ------------------------------------------------------------------ C. timing vs size
I = np.where(np.isnan(X), np.nan, (np.abs(X) > EPS).astype(float))
a, b = I[1:], I[:-1]
mm = ~np.isnan(a) & ~np.isnan(b)
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
    out = []
    for j in range(Z.shape[1]):
        z, c = Z[:, j], chg[:, j]
        u, v = z[1:], z[:-1]
        k = c[1:] & c[:-1] & ~np.isnan(u) & ~np.isnan(v) & (u != 0) & (v != 0)
        if k.sum() >= 30:
            out.append(np.corrcoef(np.log(np.abs(u[k])), np.log(np.abs(v[k])))[0, 1])
    return np.median(out)


E = ar_filter(A)[0].to_numpy()
chg = np.abs(np.nan_to_num(X)) > EPS
timing = {"p_change": np.nanmean(I),
          "p_change_after_change": (a * b)[mm].sum() / b[mm].sum(),
          "p_change_after_quiet_day": (a * (1 - b))[mm].sum() / (1 - b)[mm].sum(),
          "quiet_runs": len(quiet_runs),
          "share_quiet_days_in_runs_ge3": quiet_runs[quiet_runs >= 3].sum() / quiet_runs.sum()}
for name, Z in [("raw", X), ("ar_filtered", E)]:
    nl = []
    for _ in range(100):
        S = shuffle_cols_blocked(Z, rng, A.index)
        nl.append(size_corr(S, np.abs(np.nan_to_num(S)) > EPS))
    timing[f"size_corr_{name}"] = size_corr(Z, chg)
    timing[f"size_corr_{name}_null_p975"] = np.percentile(nl, 97.5)
timing = pd.Series(timing)
timing.round(4).to_csv(os.path.join(OUT, "aws_timing_vs_size.csv"), header=["value"])

# ------------------------------------------------------------------ D. on-demand ceiling
df = load_raw(columns=KEY + ["price_per_hour"])
df = pick_daily(df[(df.provider == "aws") & ~stale_rows(df.provider, df.day)], KEY)
pair = [c for c in KEY if c != "is_spot"]
p = df.pivot_table(index=pair + ["day"], columns="is_spot", values="price_per_hour").dropna()
p.columns = ["od", "spot"]
p["at_ceiling"] = p.spot >= p.od - 1e-9
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
