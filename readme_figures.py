"""Figures for the README, written to docs/img/.

Run after analysis.py and event_clustering.py (it reads their CSVs from out/), plus the raw
snapshots for the per-provider price paths.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, GRID, HERE, INK, MUTED, ORANGE, SUSPECT_DAYS, NEBIUS_REGIME_START

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
raw = raw[~raw.day.isin(SUSPECT_DAYS)]

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

# ------------------------------------------------------------------ 2. spot vs on-demand index
chain = pd.read_csv(os.path.join(OUTD, "chained_index_daily.csv"), parse_dates=["day"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for ax, m in zip(axes, ["on-demand", "spot"]):
    for _, g in gaps.iterrows():
        ax.axvspan(g["min"], g["max"], color=GRID, lw=0)
    ends = {}
    for gpu, c in GPU_COLORS.items():
        s = chain[(chain.gpu_type == gpu) & (chain.market == m)]
        if len(s) < 60:
            continue
        ax.plot(s.day, s.level, color=c, lw=1.8, label=gpu)
        ends[gpu] = (s.day.iloc[-1], s.level.iloc[-1])
    if m == "spot":  # end labels, nudged apart so they don't overlap
        prev = None
        for gpu, (d, v) in sorted(ends.items(), key=lambda kv: kv[1][1]):
            y = v if prev is None else max(v, prev + 0.045)
            ax.text(d, y, " " + gpu, fontsize=8, color=MUTED, va="center")
            prev = y
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_title(f"{m.capitalize()}", loc="left", fontsize=11)
axes[0].set_ylabel("price index (Jan 2026 = 1.0)")
axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
axes[1].set_xlim(right=axes[1].get_xlim()[1] + 15)
fig.autofmt_xdate()
fig.suptitle("Spot prices climbed through 2026; on-demand barely moved", x=0.01, ha="left", fontsize=12.5)
fig.tight_layout()
save(fig, "spot_vs_ondemand_index.png")

# ------------------------------------------------------------------ 3. how often each provider reprices
t = pd.read_csv(os.path.join(OUTD, "listing_by_provider_x_market.csv"))
t = t[~t.provider.isin(["vastai", "nebius_dynamic"])]  # Vast.ai: sampling noise (excluded everywhere)
t["provider"] = t.provider.replace({"nebius_static": "nebius*", "lambdalabs": "lambda"})
pm = t.pivot(index="provider", columns="market", values="mean_daily_change_freq").fillna(0)
pm = pm.loc[pm.max(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(pm) + 1.3))
y = np.arange(len(pm))
ax.barh(y - 0.2, pm["on-demand"], height=0.36, color=BLUE, label="on-demand")
ax.barh(y + 0.2, pm["spot"], height=0.36, color=ORANGE, label="spot")
for yi, (od, sp) in enumerate(zip(pm["on-demand"], pm["spot"])):
    if max(od, sp) < 0.0005:
        ax.text(0.005, yi, "never changes", va="center", fontsize=8, color=MUTED)
ax.set_yticks(y, pm.index)
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("share of days a listing's price changes")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="y", visible=False)
ax.set_title("How often each provider changes its prices", loc="left", fontsize=12)
ax.text(0, -0.13 - 0.3 / len(pm), "*Nebius before its 8 Oct 2026 switch to dynamic spot pricing",
        transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
save(fig, "change_frequency.png")

# ------------------------------------------------------------------ 4. AWS bursts decay
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
ax.set_title("AWS spot: one price change makes more changes likely for about two weeks", loc="left", fontsize=12)
save(fig, "aws_bursts.png")

print("wrote", sorted(os.listdir(IMG)))
