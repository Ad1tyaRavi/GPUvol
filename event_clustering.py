"""Clustering test #3: do price-change *days* come in bursts?

Prices here are flat most days and then jump, so instead of asking whether big moves follow big
moves, treat each price change as an event (I_t = 1 if the listing's price changed on day t) and ask:

    lift(k) = P(change today | change k days ago) / P(change today | no change k days ago)

lift = 1  -> changes arrive independently of the past (memoryless)
lift > 1  -> changes cluster: a change makes another one soon more likely
lift < 1  -> changes repel: after a change the price tends to sit still

Counts are pooled within a group (provider x market, GPU tier, ...) over consecutive-day pairs only.
Null: shuffle each listing's event days over its own observed days (keeps its change rate and gaps,
destroys timing) and recompute the pooled lift, 200 times -> 95% band.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, EPS, GRID, INK, MUTED, ORANGE, OUT, load_returns, shuffle_cols

TIER = {
    **dict.fromkeys(["B200", "B300", "GB200", "GB300"], "Blackwell DC"),
    **dict.fromkeys(["H100", "H200", "H100NVL", "H200NVL", "GH200", "H800"], "Hopper DC"),
    **dict.fromkeys(["A100", "A800", "A10", "A10G", "A30", "A40", "A16"], "Ampere DC"),
    **dict.fromkeys(["L4", "L40", "L40S"], "Ada DC"),
    **dict.fromkeys(["V100", "P100", "P4", "T4", "K80", "M60"], "Legacy DC"),
}
MAX_LAG = 14
N_SHUFFLE = 200
rng = np.random.default_rng(1)

R, meta, days = load_returns()
meta = meta.assign(market=np.where(meta.is_spot, "spot", "on-demand"),
                   tier=meta.gpu_type.map(TIER).fillna("Workstation/other"))

# Universe: listings with enough history and enough events to say anything about timing.
n_obs = R.notna().sum()
n_chg = (R.abs() > EPS).sum()
keep = (meta.provider != "vastai") & (n_obs >= 100) & (n_chg >= 10) & (n_chg / n_obs <= 0.95)
R = R.loc[:, keep[keep].index]
meta = meta.loc[R.columns]
I = np.where(np.isnan(R.to_numpy()), np.nan, (np.abs(R.to_numpy()) > EPS).astype(float))


def pair_counts(I, k):
    """Per-series counts over day pairs (t-k, t) where both are observed."""
    A, B = I[k:], I[:-k]
    both = ~np.isnan(A) & ~np.isnan(B)
    A, B = np.where(both, A, 0), np.where(both, B, 0)
    return pd.DataFrame({"n1": (B * both).sum(0), "n11": (A * B).sum(0),
                         "n0": ((1 - B) * both).sum(0), "n01": (A * (1 - B) * both).sum(0)})


def pooled_lift(counts, groups):
    g = counts.groupby(groups.to_numpy()).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (g.n11 / g.n1) / (g.n01 / g.n0), g


GROUPINGS = {
    "provider_x_market": meta.provider + " " + meta.market,
    "spot_tier": meta.tier.where(meta.is_spot),
    "spot_tier_x_provider": (meta.tier + " | " + meta.provider).where(meta.is_spot),
}

# lag-1 lift per group + shuffle null band
c1 = pair_counts(I, 1)
null = {name: [] for name in GROUPINGS}
for _ in range(N_SHUFFLE):
    c1s = pair_counts(shuffle_cols(I, rng), 1)
    for name, grp in GROUPINGS.items():
        null[name].append(pooled_lift(c1s[grp.notna().to_numpy()], grp.dropna())[0])

tables = {}
for name, grp in GROUPINGS.items():
    mask = grp.notna().to_numpy()
    lift, g = pooled_lift(c1[mask], grp.dropna())
    nl = pd.concat(null[name], axis=1)
    t = pd.DataFrame({
        "listings": grp.dropna().value_counts(),
        "p_change": (g.n11 + g.n01) / (g.n1 + g.n0),
        "p_change_after_change": g.n11 / g.n1,
        "p_change_after_no_change": g.n01 / g.n0,
        "lift_lag1": lift,
        "null_p025": nl.quantile(0.025, axis=1), "null_p975": nl.quantile(0.975, axis=1),
        "events": g.n11 + g.n01,
    })
    t["clusters"] = np.select([t.lift_lag1 > t.null_p975, t.lift_lag1 < t.null_p025], ["yes", "repels"], "no")
    t = t[t.events >= 30].sort_values("lift_lag1", ascending=False)
    tables[name] = t.round(3)
    t.round(4).to_csv(os.path.join(OUT, f"events_lift_{name}.csv"))

# lift vs lag for the groups with the most events (spot)
pm = GROUPINGS["provider_x_market"]
# GCP is left out of the lag curve: it reprices in scheduled Monday batches (lift 0 most lags,
# spikes at 7 and 14), which is a calendar, not bursts.
curve_groups = [g for g in tables["provider_x_market"].index if g.endswith("spot") and not g.startswith("gcp")][:5]
curves = {}
for k in range(1, MAX_LAG + 1):
    lk, _ = pooled_lift(pair_counts(I, k), pm)
    curves[k] = lk
curves = pd.DataFrame(curves).T
curves.index.name = "lag"
curves.round(4).to_csv(os.path.join(OUT, "events_lift_by_lag.csv"))

# ------------------------------------------------------------------ chart
t = tables["provider_x_market"]
t = t.sort_values("lift_lag1")
fig, axes = plt.subplots(1, 2, figsize=(13, 0.34 * len(t) + 1.6), gridspec_kw={"width_ratios": [1, 1.2]})
ax = axes[0]
y = np.arange(len(t))
ax.hlines(y, t.null_p025, t.null_p975, color=MUTED, lw=6, alpha=0.25, label="no-clustering range (shuffled, 95%)")
ax.scatter(t.lift_lag1, y, color=[ORANGE if c == "yes" else BLUE if c == "repels" else MUTED for c in t.clusters],
           s=40, zorder=3)
ax.axvline(1, color=MUTED, lw=0.8)
ax.set_yticks(y, t.index)
ax.set_xscale("log")
ax.set_xlabel("lift = P(change | change yesterday) / P(change | no change yesterday)   (log scale)")
ax.set_title("Lag-1 lift by provider and market", loc="left", fontsize=11)
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, fontsize=8, loc="lower right")

ax = axes[1]
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]
for g, c in zip(curve_groups, colors):
    ax.plot(curves.index, curves[g], color=c, lw=2, marker="o", ms=4, label=g)
    ax.text(MAX_LAG + 0.2, curves[g].iloc[-1], g, fontsize=8, color=MUTED, va="center")
ax.axhline(1, color=MUTED, lw=0.8)
ax.set_yscale("log")
ax.set_xlabel("lag k (days)")
ax.set_ylabel("lift at lag k (log scale)")
ax.set_xlim(0.5, MAX_LAG + 4)
ax.set_title("How long does the burst last? (spot; GCP omitted: Monday batch schedule)", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=len(curve_groups))
fig.suptitle("Do price changes come in bursts?", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "10_price_change_bursts.png"), dpi=150)
plt.close(fig)

pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_rows", 100)
print(f"listings: {I.shape[1]:,}")
for name, t in tables.items():
    print(f"\n== {name} ==\n{t}")
print("\n== lift by lag (spot groups) ==\n", curves[curve_groups].round(2))
