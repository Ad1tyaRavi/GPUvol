"""Clustering test #4: market-wide clustering. Do listings and providers reprice together?

Spot listings only (on-demand barely moves), Vast.ai excluded.

A. Within a provider: per day, activity A_t = share of the provider's listings whose price changed.
   If listings changed independently, A_t would hover tightly around its mean. The implied average
   pairwise correlation between two listings' change indicators,
       rho = (Var(A) * N / (p(1-p)) - 1) / (N - 1)     (N = typical listings per day, p = mean A)
   is ~0 for independent listings and ~1 when the provider reprices everything at once.
   It is also reported after removing weekday means and a 31-day trend from A_t (as B and C do),
   so a weekly schedule or a slow change in activity doesn't count as listings moving together.
B. Across providers: correlation of daily activity, same day and lead-lag (provider X yesterday vs
   provider Y today), plus the same for direction (mean log return of the provider's spot listings).
C. Same chip, different providers: per-GPU spot index per provider; same-day and lead-lag return
   correlations for pairs with enough overlap.
Before B and C, every daily series has its weekday means and its 30-day centred rolling mean
removed, so shared weekly schedules (GCP Mondays, quiet weekends) and slow shared regime changes
don't count as providers moving together. Correlations are Spearman (rank), so one shared shock
day can't drive them.
Null for B and C: circularly shift one series by a random whole number of weeks, 500 times; a
correlation is called significant if |corr| exceeds the 95th percentile of shifted |corr|.
Data-hygiene fixes (duplicate paths collapsed, stale-feed days dropped, Nebius split at 2026-10-08)
come via common.py.
"""
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import EPS, INK, OUT, dedupe_paths, load_returns

N_SHIFT = 500
MIN_OVERLAP = 100
rng = np.random.default_rng(2)

R, meta, days = load_returns()
spot = meta.index[meta.is_spot & (meta.provider != "vastai")]
R = dedupe_paths(R[spot])
meta = meta.loc[R.columns]
chg = (R.abs() > EPS).astype(float).where(R.notna())  # 1 = changed, 0 = didn't, NaN = no data


def deseason(s):
    """Remove weekday means, then the 31-day centred rolling mean."""
    s = s - s.groupby(s.index.dayofweek).transform("mean")
    return s - s.rolling(31, center=True, min_periods=10).mean()


# ------------------------------------------------------------------ A. within-provider sync
act, dirn, rows = {}, {}, []
for prov, ids in meta.groupby("provider").groups.items():
    obs = R[ids].notna().sum(1)
    a = (chg[ids].sum(1) / obs).where(obs >= 5)
    if a.notna().sum() < MIN_OVERLAP or a.std() == 0:
        continue
    act[prov] = a
    dirn[prov] = R[ids].mean(1).where(obs >= 5)
    p, N = a.mean(), obs[obs >= 5].median()
    rho = (a.var() * N / (p * (1 - p)) - 1) / (N - 1)
    rho_ds = ((deseason(a) + p).var() * N / (p * (1 - p)) - 1) / (N - 1)
    big = a > 0.5
    rows.append({"provider": prov, "listings_per_day": N, "mean_activity": p, "implied_pairwise_corr": rho,
                 "implied_pairwise_corr_deseasoned": rho_ds,
                 "share_of_days_most_listings_change": big.mean(),
                 "share_of_changes_on_those_days": (chg[ids].sum(1)[big]).sum() / chg[ids].sum(1).sum()})
    rows[-1]["days_with_any_change"] = int((a > 0).sum())
within = pd.DataFrame(rows).set_index("provider").round(3).sort_values("implied_pairwise_corr", ascending=False)
within.to_csv(os.path.join(OUT, "sync_within_provider.csv"))
act, dirn = pd.DataFrame(act), pd.DataFrame(dirn)

# Shared shock days: >= 2 providers repricing far above their usual level on the same day.
# These may be real market events or scraper/catalog changes; listed so they can be inspected.
unusual = act.gt(act.quantile(0.95)) & act.gt(0)
shock = act[unusual.sum(1) >= 2].round(2)
shock.insert(0, "providers_unusual", unusual.sum(1)[unusual.sum(1) >= 2])
shock.to_csv(os.path.join(OUT, "sync_shared_shock_days.csv"))

# Cross-provider tests need a provider that actually reprices: >= 10 days with any change.
active = within.index[within.days_with_any_change >= 10]


act, dirn = act[active].apply(deseason), dirn[active].apply(deseason)


# ------------------------------------------------------------------ B/C helpers
def corr_lag(x, y, lag):
    """Spearman corr(x_{t-lag}, y_t) on overlapping non-NaN days; returns (corr, n).
    Rank correlation on purpose: Pearson here is driven by one or two shared shock days
    (catch-up days after a stale feed, when everything moves at once)."""
    xs = x.shift(lag)
    m = xs.notna() & y.notna()
    if m.sum() < MIN_OVERLAP or xs[m].nunique() < 3 or y[m].nunique() < 3:
        return np.nan, int(m.sum())
    return float(np.corrcoef(xs[m].rank(), y[m].rank())[0, 1]), int(m.sum())


def shift_null(x, y, lag):
    """95th percentile of |corr| when y is circularly shifted by a random offset."""
    yv = y.to_numpy()
    T = len(yv)
    out = []
    for s in 7 * rng.integers(1, T // 7, N_SHIFT):  # whole weeks, so weekday alignment is kept
        c, _ = corr_lag(x, pd.Series(np.roll(yv, s), index=y.index), lag)
        out.append(abs(c))
    return np.nanpercentile(out, 95)


def pairwise(frame, label):
    rows = []
    for a, b in itertools.permutations(frame.columns, 2):
        for lag in (0, 1):
            if lag == 0 and a > b:
                continue  # same-day corr is symmetric
            c, n = corr_lag(frame[a], frame[b], lag)
            if np.isnan(c):
                continue
            thr = shift_null(frame[a], frame[b], lag)
            rows.append({"series": label, "leader": a, "follower": b, "lag": lag, "corr": c, "n_days": n,
                         "null_abs_p95": thr, "significant": abs(c) > thr})
    return pd.DataFrame(rows)


# B. across providers: activity and direction
cross = pd.concat([pairwise(act, "activity"), pairwise(dirn, "direction")], ignore_index=True)
cross.round(3).to_csv(os.path.join(OUT, "sync_cross_provider.csv"), index=False)

# C. same chip, different providers (per-provider per-GPU spot index of same-listing returns)
chip_rows = []
for gpu in ["H100", "H200", "B200", "B300", "A100", "L40S", "L4", "T4", "V100", "A10G"]:
    ids = meta.index[meta.gpu_type == gpu]
    if len(ids) == 0:
        continue
    per_prov = {}
    for prov, pids in meta.loc[ids].groupby("provider").groups.items():
        s = R[pids].mean(1)
        if s.notna().sum() >= MIN_OVERLAP and (s.abs() > EPS).mean() >= 0.05:
            per_prov[prov] = deseason(s)
    if len(per_prov) >= 2:
        chip_rows.append(pairwise(pd.DataFrame(per_prov), gpu))
chip = pd.concat(chip_rows, ignore_index=True) if chip_rows else pd.DataFrame()
chip.round(3).to_csv(os.path.join(OUT, "sync_same_chip.csv"), index=False)

# ------------------------------------------------------------------ chart: activity correlation heatmaps
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f1ed", "#eb6834"])
provs = list(act.columns)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
for ax, lag, title in [(axes[0], 0, "Same day: corr(activity X, activity Y)"),
                       (axes[1], 1, "Lead-lag: corr(X yesterday, Y today)")]:
    M = pd.DataFrame(np.nan, index=provs, columns=provs)
    S = pd.DataFrame(False, index=provs, columns=provs)
    sub = cross[(cross.series == "activity") & (cross.lag == lag)]
    for _, r in sub.iterrows():
        M.loc[r.leader, r.follower] = r["corr"]
        S.loc[r.leader, r.follower] = r.significant
        if lag == 0:
            M.loc[r.follower, r.leader] = r["corr"]
            S.loc[r.follower, r.leader] = r.significant
    im = ax.imshow(M.to_numpy(dtype=float), cmap=cmap, vmin=-0.6, vmax=0.6)
    for i, j in itertools.product(range(len(provs)), range(len(provs))):
        v = M.iat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}{'*' if S.iat[i, j] else ''}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(provs)), provs, rotation=45, ha="right")
    ax.set_yticks(range(len(provs)), provs)
    ax.set_xlabel("Y (follower, today)" if lag else "")
    ax.set_ylabel("X (leader, yesterday)" if lag else "")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(False)
fig.colorbar(im, ax=axes, shrink=0.8, label="correlation (* = beyond circular-shift null, 95%)")
fig.suptitle("Do providers reprice spot GPUs on the same days?", x=0.01, ha="left", fontsize=12, color=INK)
fig.savefig(os.path.join(OUT, "11_provider_repricing_sync.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_rows", 200)
print("== A. within-provider sync ==\n", within)
print("\n== B. cross-provider (significant only) ==\n",
      cross[cross.significant].sort_values(["series", "lag", "corr"], ascending=[True, True, False]).round(3).to_string(index=False))
print("\n   tested:", len(cross), " significant:", int(cross.significant.sum()))
if len(chip):
    print("\n== C. same chip across providers (significant only) ==\n",
          chip[chip.significant].round(3).to_string(index=False))
    print("\n   tested:", len(chip), " significant:", int(chip.significant.sum()))
