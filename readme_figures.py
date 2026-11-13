"""Figures for the README, written to docs/img/.

Run after analysis.py and event_clustering.py (it reads their CSVs from out/), plus the raw
snapshots for the per-provider price paths.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (BLUE, GRID, HERE, INK, MUTED, NEBIUS_REGIME_START, ORANGE, STALE_END, STALE_PROVIDERS,
                    STALE_START, SUSPECT_DAYS, stale_rows)

STALE_COLOR = "#ece6dc"  # AWS/GCP stale-feed window, drawn as missing data

IMG = os.path.join(HERE, "docs", "img")
OUTD = os.path.join(HERE, "out")
os.makedirs(IMG, exist_ok=True)
GPU_COLORS = {"H100": "#2a78d6", "H200": "#eb6834", "B200": "#1baf7a", "L40S": "#e87ba4",
              "A100": "#6250d6", "T4": "#52514e"}


def save(fig, name):
    fig.savefig(os.path.join(IMG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 1. four pricing styles
# One typical listing per provider, $/GPU-hr, last snapshot each day, suspect days dropped.
PICKS = [  # provider, instance_type, region, is_spot, title
    ("aws", "p5.48xlarge", "eu-north-1", True, "AWS spot H100: small moves almost every day"),
    ("gcp", "a3-edgegpu-8g", "europe-west3-a", True, "GCP spot H100: steps, in weekly batches"),
    ("verda", "1H100.80S.30V", "FIN-01", False, "Verda on-demand H100: list price that moves"),
    ("nebius", "gpu-h200-sxm 1gpu-16vcpu-200gb", "eu-north1", True, "Nebius spot H200: static, then dynamic"),
]
raw = pd.read_parquet(os.path.join(HERE, "all.parquet"),
                      columns=["provider", "instance_type", "region", "is_spot", "gpu_count", "price_per_hour",
                               "timestamp", "quality"])
raw = raw[(raw.quality == "ok") & (raw.gpu_count > 0) & (raw.price_per_hour > 0)]
raw["day"] = raw.timestamp.dt.tz_convert(None).dt.normalize()
raw = raw[~raw.day.isin(SUSPECT_DAYS) & ~stale_rows(raw.provider, raw.day)]

# Stretches of 3+ days with no snapshots at all (e.g. 2026-03-10 to 2026-05-07): shaded, line broken.
all_days = pd.date_range(raw.day.min(), raw.day.max(), freq="D")
missing = all_days.difference(raw.day.unique())
runs = pd.Series(missing).groupby((pd.Series(missing).diff().dt.days != 1).cumsum()).agg(["min", "max", "count"])
gaps = runs[runs["count"] >= 3]

fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), sharex=True)
for i, (ax, (prov, inst, reg, spot, title)) in enumerate(zip(axes.flat, PICKS)):
    s = raw[(raw.provider == prov) & (raw.instance_type == inst) & (raw.region == reg) & (raw.is_spot == spot)]
    s = s.sort_values("timestamp").groupby("day").agg(p=("price_per_hour", "last"), n=("gpu_count", "first"))
    price = s.p / s.n
    full = price.reindex(pd.date_range(price.index.min(), price.index.max(), freq="D")).ffill(limit=2)
    for _, g in gaps.iterrows():
        ax.axvspan(g["min"], g["max"], color=GRID, lw=0, label="no data" if i == 0 and _ == gaps.index[0] else None)
    if prov in STALE_PROVIDERS:
        ax.axvspan(STALE_START, STALE_END, color=STALE_COLOR, lw=0, label="stale feed" if i == 0 else None)
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
ax.axvspan(STALE_START, STALE_END, color=STALE_COLOR, lw=0, label="AWS/GCP feed stale")
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
ax.set_title("Spot prices by chip: most up, in steps, mostly from August", loc="left", fontsize=12)
save(fig, "spot_index.png")

# ------------------------------------------------------------------ 3. AWS bursts decay
lift = pd.read_csv(os.path.join(OUTD, "events_lift_by_lag.csv"), index_col="lag")["aws spot"]
null = pd.read_csv(os.path.join(OUTD, "events_lift_by_lag_null.csv"))
null = null[null.group == "aws spot"].set_index("lag")
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.fill_between(null.index, null.null_p025, null.null_p975, color="#a9a8a2", edgecolor="#a9a8a2", lw=2,
                label="chance (shuffled within weekday & month)")
ax.plot(lift.index, lift.values, color=ORANGE, lw=2.2, marker="o", ms=5, label="actual")
ax.set_xlabel("days since a price change")
ax.set_ylabel("how much likelier another change is")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%gx"))
ax.set_ylim(0, None)
ax.set_xticks(range(1, len(lift) + 1))
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="x", visible=False)
ax.set_title("AWS spot: one price change makes more changes likely for about two weeks", loc="left",
             fontsize=12, pad=16)
ax.text(0, 1.02, "chance sits at 2-3x, not 1x, because some listings simply change more often than others",
        transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
save(fig, "aws_bursts.png")

# ------------------------------------------------------------------ 4. price-change raster
# Rows: up to 40 spot listings per provider (identical price paths collapsed, spread across the
# provider's range of change rates). Columns: days. Dark = price changed that day.
import matplotlib.dates as mdates

from common import EPS, dedupe_paths, load_returns

R, meta, days = load_returns(clean=False)  # Nebius kept whole so its switch is visible
bad = [d + pd.Timedelta(days=k) for d in SUSPECT_DAYS for k in (0, 1)]
R.loc[R.index.intersection(bad)] = np.nan  # same suspect days as the analysis
R.loc[STALE_START:STALE_END, meta.index[meta.provider.isin(STALE_PROVIDERS)]] = np.nan  # stale feed
ROWS = 40
blocks, labels = [], []
for prov in ["aws", "gcp", "verda", "nebius", "azure"]:
    ids = meta.index[(meta.provider == prov) & meta.is_spot]
    X = R[ids]
    X = dedupe_paths(X.loc[:, X.notna().sum() >= 150])
    rate = ((X.abs() > EPS).sum() / X.notna().sum()).sort_values(ascending=False)
    pick = rate.index[np.unique(np.linspace(0, len(rate) - 1, min(ROWS, len(rate))).round().astype(int))]
    M = np.where(X[pick].isna(), np.nan, (X[pick].abs() > EPS).astype(float)).T
    blocks.append(M)
    labels.append((prov, len(pick)))
sep = np.full((3, len(R.index)), -1.0)  # spacer rows between providers
mat = np.vstack([b for blk in blocks for b in (blk, sep)][:-1])
rgb = np.ones(mat.shape + (3,))
hexrgb = lambda h: [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
rgb[mat == 0] = hexrgb("#f2f1ed")       # observed, no change
rgb[mat == 1] = hexrgb("#2a78d6")       # price changed
rgb[np.isnan(mat)] = hexrgb("#fcfcfb")  # no data
rgb[mat == -1] = hexrgb("#fcfcfb")      # spacer
x0, x1 = mdates.date2num(R.index[0]), mdates.date2num(R.index[-1] + pd.Timedelta(days=1))
fig, ax = plt.subplots(figsize=(12, 6.2))
ax.imshow(rgb, aspect="auto", interpolation="nearest", extent=[x0, x1, mat.shape[0], 0])
ticks, row = [], 0
for prov, n in labels:
    ticks.append(row + n / 2)
    row += n + 3
ax.set_yticks(ticks, [p.upper() if p in ("aws", "gcp") else p.capitalize() for p, _ in labels])
ax.tick_params(axis="y", length=0)
ax.xaxis_date()
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
fig.autofmt_xdate()
ax.set_title("Providers reprice in their own batches, not together", loc="left", fontsize=12, pad=16)
ax.text(0, 1.01, f"blue = price changed that day; one row per spot listing (up to {ROWS} per provider), "
        "one column per day; white = no data", transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
save(fig, "repricing_raster.png")

print("wrote", sorted(os.listdir(IMG)))
