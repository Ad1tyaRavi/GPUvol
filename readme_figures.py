"""Figures for the README, written to docs/img/.

Run after analysis.py and aws_spot.py (it reads their CSVs from out/), plus the raw snapshots
for the per-provider price paths.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (BLUE, GRID, HERE, INK, MUTED, NEBIUS_REGIME_START, ORANGE, STALE_EXEMPT, load_raw,
                    pick_daily, stale_rows, stale_spans)

STALE_COLOR = "#ece6dc"  # stale-feed windows, drawn as missing data


def shade_stale(ax, prov=None, label=True):
    """Shade the stale-feed windows that apply to `prov` (all of them when prov is None)."""
    for k, (provs, a, b) in enumerate(stale_spans()):
        if prov is None or (prov not in STALE_EXEMPT if provs is None else prov in provs):
            ax.axvspan(a, b + pd.Timedelta(days=1), color=STALE_COLOR, lw=0,
                       label="stale feed" if label and k == 0 else None)

IMG = os.path.join(HERE, "docs", "img")
OUTD = os.path.join(HERE, "out")
os.makedirs(IMG, exist_ok=True)
GPU_COLORS = {"H100": "#2a78d6", "H200": "#eb6834", "B200": "#1baf7a", "L40S": "#e87ba4",
              "A100": "#6250d6", "T4": "#52514e"}


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 1. four pricing styles
# One typical listing per provider, $/GPU-hr, snapshot closest to 09:00 UTC, stale days dropped.
PICKS = [  # provider, instance_type, region, is_spot, title
    ("aws", "p5.48xlarge", "eu-north-1", True, "AWS spot H100: small moves almost every day"),
    ("gcp", "a3-edgegpu-8g", "europe-west3-a", True, "GCP spot H100: steps, in weekly batches"),
    ("verda", "1H100.80S.30V", "FIN-01", False, "Verda on-demand H100: list price that moves"),
    ("nebius", "gpu-h200-sxm 1gpu-16vcpu-200gb", "eu-north1", True, "Nebius spot H200: static, then dynamic"),
]
raw = load_raw(columns=["provider", "instance_type", "region", "is_spot", "gpu_type", "gpu_count",
                        "price_per_hour", "timestamp", "quality"])
raw = raw[~stale_rows(raw.provider, raw.day)]

# Stretches of 3+ days with no snapshots at all (e.g. 2026-03-10 to 2026-05-07): shaded, line broken.
all_days = pd.date_range(raw.day.min(), raw.day.max(), freq="D")
missing = all_days.difference(raw.day.unique())
runs = pd.Series(missing).groupby((pd.Series(missing).diff().dt.days != 1).cumsum()).agg(["min", "max", "count"])
gaps = runs[runs["count"] >= 3]

fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), sharex=True)
for i, (ax, (prov, inst, reg, spot, title)) in enumerate(zip(axes.flat, PICKS)):
    s = raw[(raw.provider == prov) & (raw.instance_type == inst) & (raw.region == reg) & (raw.is_spot == spot)]
    s = pick_daily(s, []).set_index("day").sort_index().rename(columns={"price_per_hour": "p", "gpu_count": "n"})
    price = s.p / s.n
    full = price.reindex(pd.date_range(price.index.min(), price.index.max(), freq="D")).ffill(limit=2)
    for _, g in gaps.iterrows():
        ax.axvspan(g["min"], g["max"], color=GRID, lw=0, label="no data" if i == 0 and _ == gaps.index[0] else None)
    shade_stale(ax, prov, label=i == 0)
    ax.plot(full.index, full.values, drawstyle="steps-post", color=BLUE, lw=1.6)
    chg = price[price.diff().abs() > 1e-9]
    sub = f"price changed on {len(chg)} of {len(price)} days"
    if prov == "gcp":
        sub += f", {np.isin(chg.index.dayofweek, [0, 1]).mean():.0%} of them on a Monday or Tuesday"
    if prov == "nebius":
        ax.axvline(NEBIUS_REGIME_START, color=ORANGE, lw=1, ls="--")
        ax.text(NEBIUS_REGIME_START, 0.97, "8 Oct ", transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=8, color=MUTED)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=16)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    ax.set_ylabel("$ per GPU-hour")
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
fig.autofmt_xdate()
fig.suptitle("Four providers, four pricing machines", x=0.01, ha="left", fontsize=12.5, color=INK)
fig.tight_layout()
save(fig, "pricing_styles.png")

# ------------------------------------------------------------------ 2. spot price index by chip
# Spot only: the on-demand index is a flat line, which the README says in words.
chain = pd.read_csv(os.path.join(OUTD, "chained_index_daily.csv"), parse_dates=["day"])
chain = chain[chain.market == "spot"]
fig, ax = plt.subplots(figsize=(9, 4.4))
for _, g in gaps.iterrows():
    ax.axvspan(g["min"], g["max"], color=GRID, lw=0, label="no data" if _ == gaps.index[0] else None)
shade_stale(ax)
ends = {}
for gpu, c in GPU_COLORS.items():
    s = chain[chain.gpu_type == gpu]
    if len(s) < 60:
        continue
    ax.plot(s.day, s.level, color=c, lw=1.8, label=gpu)
    ends[gpu] = (s.day.iloc[-1], s.level.iloc[-1])
prev = None  # end labels, nudged apart so they don't overlap
for gpu, (d, v) in sorted(ends.items(), key=lambda kv: kv[1][1]):
    y = v if prev is None else max(v, prev + 0.045)
    ax.text(d, y, f" {gpu} {v - 1:+.0%}", fontsize=8, color=MUTED, va="center")
    prev = y
ax.axhline(1, color=MUTED, lw=0.8)
ax.set_ylabel("spot price index (Jan 2026 = 1.0)")
ax.set_xlim(right=ax.get_xlim()[1] + 30)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
fig.autofmt_xdate()
ax.set_title("Spot prices by chip: most up, in steps", loc="left", fontsize=12, pad=16)
ax.text(0, 1.01, "moves made while unobserved (no data, stale feed) count on the day prices are seen again",
        transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
save(fig, "spot_index.png")

# ------------------------------------------------------------------ 3. AWS spot momentum
bins = pd.read_csv(os.path.join(OUTD, "aws_momentum_bins.csv"))
mom = pd.read_csv(os.path.join(OUTD, "aws_momentum.csv"), index_col=0)
acf = pd.read_csv(os.path.join(OUTD, "aws_return_acf.csv"), index_col="lag").median_acf_r
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
lim = 100 * np.abs(bins.today).max() * 1.05
xs = np.linspace(-lim, lim, 2)
ax.plot(xs, xs * mom.loc["all", "slope"], color=MUTED, lw=1, ls="--",
        label=f"next week = {mom.loc['all', 'slope']:.2f} x today")
ax.scatter(100 * bins.today, 100 * bins.next7, color=ORANGE, s=34, zorder=3, label="average, 20 bins of today's move")
ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel("today's price change (%)")
ax.set_ylabel("change over the next 7 days (%)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Today's move predicts the next week", loc="left", fontsize=11)
ax = axes[1]
ax.bar(acf.index, acf.values, color=[ORANGE if v > 0 else BLUE for v in acf.values], width=0.8)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("lag (days)")
ax.set_ylabel("median autocorrelation of daily changes")
ax.set_xticks([1, 7, 14, 21, 28, 35, 42, 49, 56])
ax.grid(axis="x", visible=False)
ax.set_title("Momentum for ~5 days, then a slight pullback", loc="left", fontsize=11)
fig.suptitle(f"AWS spot prices keep going: the next week continues today's direction "
             f"{mom.loc['all', 'same_direction']:.0%} of the time "
             f"({mom.loc['out_of_sample_sep_nov', 'same_direction']:.0%} out of sample)",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout()
save(fig, "aws_momentum.png")

# ------------------------------------------------------------------ 3b. AWS spot local markets
co = pd.read_csv(os.path.join(OUTD, "aws_comovement.csv"))
order = ["different chip, different region", "same instance type, different region", "same region, different chip"]
fig, ax = plt.subplots(figsize=(9, 3.4))
y = np.arange(len(order))
for off, h, c, lab in [(0.18, 1, BLUE, "daily changes"), (-0.18, 7, ORANGE, "weekly changes")]:
    t = co[co.horizon_days == h].set_index("pairs").loc[order]
    ax.barh(y + off, t.median_corr, height=0.34, color=c, label=lab)
    for yi, (v, n) in enumerate(zip(t.median_corr, t.n_pairs)):
        ax.text(max(v, 0), yi + off, f" {v:.3f}", va="center", fontsize=8, color=MUTED)
    ax.scatter(t.null_p975, y + off, marker="|", s=120, color=INK, zorder=3,
               label="chance (95th pct, listings shifted in time)" if h == 1 else None)
pairs = co[co.horizon_days == 1].set_index("pairs").loc[order].n_pairs
ax.set_yticks(y, [f"{o}\n({n:,} pairs)" for o, n in zip(order, pairs)])
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlim(-0.02, 0.16)
ax.set_xlabel("median correlation between two AWS spot listings")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(axis="y", visible=False)
fig.suptitle("AWS spot is hundreds of local markets: listings barely move together, and region matters more than chip",
             x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout()
save(fig, "aws_local_markets.png")

print("wrote", sorted(os.listdir(IMG)))
