"""GPU rental price volatility across provider / GPU / spot / region axes.

Two levels of analysis:
  1. Listing level: a "series" is one SKU (provider, instance_type, region, spot, config).
     Prices are sticky, so volatility = how often a price changes x how big the jump is.
  2. Index level: per (provider, gpu_type, spot) daily median $/GPU-hr. This handles
     marketplaces (Vast.ai) whose listing IDs churn every day.
Same data-hygiene fixes as the clustering scripts (see audit.py / common.py): Nebius split at its
2026-10-20 regime switch, suspect days dropped, identical listing return paths collapsed.
Outputs go to ./out (CSV tables + PNG charts), or out/with_suspect_days/ if GPUVOL_KEEP_SUSPECT=1.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (BLUE, HERE, INK, KEEP_SUSPECT, MUTED, NEBIUS_REGIME_START, ORANGE, OUT,
                    SUSPECT_DAYS, dedupe_paths)

MIN_OBS = 30  # min daily returns for a series to count

# ---------------------------------------------------------------- load + clean
src = os.path.join(HERE, "all.parquet")
if not os.path.exists(src):
    import glob
    pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(HERE, "data/prices/*/*.parquet"))],
              ignore_index=True).to_parquet(src)
df = pd.read_parquet(src)
df = df[(df.quality == "ok") & (df.gpu_count > 0) & (df.price_per_hour > 0) & (df.gpu_type != "Unknown")].copy()
# DataCrunch rebranded to Verda; its May-Jul rows duplicate Verda's catalog, so drop them.
df = df[df.provider != "datacrunch"]
# Fractional (vGPU) slices are labeled gpu_count=1 with gpu_memory_gb = the slice's VRAM,
# e.g. Vultr "vcg-a100-1c-6g-4vram" is 4 GB of an 80 GB A100. Convert to effective GPUs using
# the full card's memory, only for provider/chip pairs known to sell slices (A100 40 vs 80 GB
# elsewhere is real hardware, not a slice).
FULL_MEM = {("vultr", "A100"): 80, ("vultr", "A40"): 48, ("vultr", "A16"): 16,
            ("azure", "A10"): 24, ("gcp", "RTXPRO6000"): 96}
full = pd.Series([FULL_MEM.get(k, np.nan) for k in zip(df.provider, df.gpu_type)], index=df.index)
df["gpu_fraction"] = (df.gpu_memory_gb / full).clip(upper=1).fillna(1.0)
df["is_fractional"] = df.gpu_fraction < 0.99
df["eff_gpus"] = df.gpu_count * df.gpu_fraction
df["ppg"] = df.price_per_hour / df.eff_gpus  # $ per (full) GPU-hour
df["day"] = df.timestamp.dt.tz_convert(None).dt.normalize()
df["market"] = np.where(df.is_spot, "spot", "on-demand")
df["region_group"] = df.region_group.fillna("Unknown")
# Nebius: static spot pricing until 2026-10-20, daily dynamic after. Treat as two providers.
nb = df.provider == "nebius"
df.loc[nb, "provider"] = np.where(df.loc[nb, "day"] < NEBIUS_REGIME_START, "nebius_pre_oct20", "nebius_post_oct20")
# Suspect days (likely scraper/catalog changes): dropping the rows also drops the next day's return,
# because returns require consecutive days.
if not KEEP_SUSPECT:
    df = df[~df.day.isin(SUSPECT_DAYS)]

TIER = {  # rough generation / segment buckets
    **dict.fromkeys(["B200", "B300", "GB200", "GB300"], "Blackwell DC"),
    **dict.fromkeys(["H100", "H200", "H100NVL", "GH200", "H800"], "Hopper DC"),
    **dict.fromkeys(["A100", "A800", "A10", "A10G", "A30", "A40", "A16"], "Ampere DC"),
    **dict.fromkeys(["L4", "L40", "L40S"], "Ada DC"),
    **dict.fromkeys(["V100", "P100", "P4", "T4", "K80", "M60"], "Legacy DC"),
    **dict.fromkeys(["RTX4090", "RTX5090", "RTX3090", "RTX4080", "RTX5080", "A6000",
                     "RTX6000Ada", "RTXPRO6000", "RTXPRO6000S", "RTXPRO4500", "A5000", "A4000"], "Workstation/consumer"),
}
df["tier"] = df.gpu_type.map(TIER).fillna("Other (TPU, misc)")
HYPERSCALERS = {"aws", "gcp", "azure", "oci"}
df["provider_class"] = np.select(
    [df.provider.isin(HYPERSCALERS), df.provider.isin(["vastai", "runpod", "lium"])],
    ["hyperscaler", "marketplace"], "neocloud")

# ------------------------------------------------------------ listing level
key = ["provider", "instance_type", "region", "is_spot", "gpu_type", "gpu_count", "eff_gpus", "vcpus", "ram_gb"]
daily = (df.sort_values("timestamp").groupby(key + ["day"], dropna=False)
         .agg(ppg=("ppg", "last"), market=("market", "first"), tier=("tier", "first"),
              region_group=("region_group", "first"), provider_class=("provider_class", "first"))
         .reset_index())
daily = daily.sort_values(key + ["day"])
g = daily.groupby(key, dropna=False, sort=False)
daily["gap"] = g.day.diff().dt.days
daily["r"] = np.log(daily.ppg / g.ppg.shift())
r = daily[daily.gap == 1].dropna(subset=["r"])  # consecutive-day returns only

# Collapse listings with identical return paths (GCP/Verda set one price for many regions/sizes),
# so one price decision isn't counted many times in the axis tables or the matched index.
r["sid"] = r.groupby(key, dropna=False).ngroup()
keep_sids = dedupe_paths(r.pivot(index="day", columns="sid", values="r")).columns
r = r[r.sid.isin(keep_sids)]

ser = (r.groupby(key, dropna=False)
       .agg(n=("r", "size"), vol=("r", "std"), chg_freq=("r", lambda x: (x.abs() > 1e-9).mean()),
            jump=("r", lambda x: x[x.abs() > 1e-9].abs().median()),
            market=("market", "first"), tier=("tier", "first"), region_group=("region_group", "first"),
            provider_class=("provider_class", "first"))
       .reset_index())
ser = ser[ser.n >= MIN_OBS]
ser["ann_vol"] = ser.vol * np.sqrt(365)
ser.to_csv(os.path.join(OUT, "series_vol.csv"), index=False)


def by(axis, frame=ser):
    t = (frame.groupby(axis)
         .agg(series=("vol", "size"), median_ann_vol=("ann_vol", "median"),
              p90_ann_vol=("ann_vol", lambda x: x.quantile(0.9)),
              pct_series_ever_change=("chg_freq", lambda x: (x > 0).mean()),
              mean_daily_change_freq=("chg_freq", "mean"), median_jump=("jump", "median"))
         .sort_values("mean_daily_change_freq", ascending=False))
    return t.round(4)


tables = {ax: by(ax) for ax in ["provider", "market", "tier", "region_group", "provider_class"]}
tables["provider_x_market"] = by(["provider", "market"])

# GPU count per instance (1, 2, 4, 8, 16+ ...); odd counts are rare so bucket them.
COUNT_ORDER = ["fractional", "1", "2", "4", "8", "16+", "other"]
cnt = lambda c: "fractional" if c < 0.99 else (str(int(c)) if c in (1, 2, 4, 8) else ("16+" if c >= 16 else "other"))
ser["gpu_count_bucket"] = pd.Categorical(ser.eff_gpus.map(cnt), COUNT_ORDER, ordered=True)
tables["gpu_count"] = by("gpu_count_bucket").sort_index()
tables["gpu_count_x_market"] = by(["gpu_count_bucket", "market"]).sort_index()
# Same chip, different node size: does tier mix explain the count effect?
tables["gpu_count_x_tier"] = by(["tier", "gpu_count_bucket"]).sort_index()
for name, t in tables.items():
    t.to_csv(os.path.join(OUT, f"listing_by_{name}.csv"))

# -------------------------------------------------------------- index level
# Vast.ai is excluded: the scrape sees ~1-4 offers per GPU per day and offer IDs live ~1 day,
# so a daily median there measures which offers got sampled, not the market price.
idx = (df[df.provider != "vastai"].groupby(["provider", "gpu_type", "market", "day"]).ppg.median()
       .rename("med").reset_index().sort_values(["provider", "gpu_type", "market", "day"]))
gi = idx.groupby(["provider", "gpu_type", "market"])
idx["gap"] = gi.day.diff().dt.days
idx["r"] = np.log(idx.med / gi.med.shift())
ir = idx[idx.gap == 1].dropna(subset=["r"])
ivol = (ir.groupby(["provider", "gpu_type", "market"]).r
        .agg(n="size", vol="std", chg_freq=lambda x: (x.abs() > 1e-9).mean()).reset_index())
ivol = ivol[ivol.n >= MIN_OBS]
ivol["ann_vol"] = ivol.vol * np.sqrt(365)
ivol.sort_values("ann_vol", ascending=False).round(4).to_csv(os.path.join(OUT, "index_vol.csv"), index=False)

# Cross-provider dispersion for flagship GPUs: same chip, how far apart are prices?
FLAG = ["H100", "H200", "A100", "B200", "L40S", "RTX4090", "T4"]
xp = (df[df.gpu_type.isin(FLAG)].groupby(["gpu_type", "market", "day", "provider"]).ppg.median()
      .groupby(["gpu_type", "market", "day"]).agg(["min", "median", "max", "count"]).reset_index())
xp = xp[xp["count"] >= 3]
xp["max_min_ratio"] = xp["max"] / xp["min"]
disp = xp.groupby(["gpu_type", "market"]).agg(days=("day", "size"), median_spread_ratio=("max_min_ratio", "median"),
                                               providers=("count", "median")).round(2)
disp.to_csv(os.path.join(OUT, "cross_provider_dispersion.csv"))

# Per-GPU price premium by node size: same provider, chip, region, market and day,
# $/GPU-hr at count n divided by $/GPU-hr at count 1. >1 = bigger nodes cost more per GPU.
pg = (df[df.provider != "vastai"].groupby(["provider", "gpu_type", "region", "market", "day", "eff_gpus"])
      .ppg.median().unstack("eff_gpus"))
pg = pg[pg[1].notna()] if 1 in pg else pg.iloc[0:0]
prem = (pg.drop(columns=1).div(pg[1], axis=0).stack().rename("ratio").reset_index()
        .rename(columns={"level_6": "eff_gpus"}))
prem["gpu_count_bucket"] = pd.Categorical(prem.eff_gpus.map(cnt), COUNT_ORDER, ordered=True)
premium = (prem.groupby(["gpu_count_bucket", "market"], observed=True).ratio
           .agg(pairs="size", median_ratio="median", p10=lambda x: x.quantile(0.1), p90=lambda x: x.quantile(0.9))
           .round(3))
premium_by_provider = (prem.groupby(["provider", "gpu_count_bucket"], observed=True).ratio
                       .agg(pairs="size", median_ratio="median").round(3))
premium.to_csv(os.path.join(OUT, "gpu_count_per_gpu_premium.csv"))
premium_by_provider.to_csv(os.path.join(OUT, "gpu_count_per_gpu_premium_by_provider.csv"))

# H100 cross-provider price paths
h = df[(df.gpu_type == "H100") & (df.provider != "vastai")].groupby(["provider", "market", "day"]).ppg.median().reset_index()

# Drift: monthly median $/GPU-hr per flagship GPU (provider medians, then median across providers)
df["month"] = df.day.dt.to_period("M")
drift = (df[df.gpu_type.isin(FLAG) & (df.provider != "vastai")]
         .groupby(["gpu_type", "market", "month", "provider"]).ppg.median()
         .groupby(["gpu_type", "market", "month"]).median().unstack("month").round(2))
drift["first_to_last"] = (drift.iloc[:, -1] / drift.iloc[:, 0] - 1).round(3)
drift.to_csv(os.path.join(OUT, "flagship_monthly_drift.csv"))

# Chained (matched-listing) index: each day's return = equal-weight mean of same-listing log
# changes across listings seen on both days. Immune to listings entering/leaving the sample.
rr = r[r.provider != "vastai"]
chain = rr.groupby(["gpu_type", "market", "day"]).r.agg(["mean", "size"]).reset_index()
chain = chain[chain["size"] >= 3]
chain_sum = (chain.groupby(["gpu_type", "market"])
             .agg(days=("mean", "size"), cum_change=("mean", lambda x: np.expm1(x.sum())),
                  daily_vol=("mean", "std"), listings_per_day=("size", "median")))
chain_sum["ann_vol"] = chain_sum.daily_vol * np.sqrt(365)
chain_sum = chain_sum[chain_sum.days >= 60].round(3)
chain_sum.to_csv(os.path.join(OUT, "chained_index.csv"))
chain["level"] = chain.groupby(["gpu_type", "market"])["mean"].cumsum().pipe(np.exp)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, m in zip(axes, ["on-demand", "spot"]):
    for gpu, c in zip(["H100", "H200", "B200", "A100", "L40S", "T4"],
                      ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#52514e"]):
        s = chain[(chain.gpu_type == gpu) & (chain.market == m)]
        if len(s) < 60:
            continue
        ax.plot(s.day, s.level, lw=1.8, color=c, label=gpu)
        ax.text(s.day.iloc[-1], s.level.iloc[-1], " " + gpu, fontsize=8, color=MUTED, va="center")
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_title(f"Matched-listing price index, {m} (Jan 2026 = 1.0)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "07_chained_index.png"), dpi=150); plt.close(fig)

# ------------------------------------------------------------------- charts
def hbar(t, col, title, fname, fmt="{:.0%}", xlabel=""):
    t = t.sort_values(col)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.2))
    ax.barh([str(i) for i in t.index], t[col], color=BLUE, height=0.6)
    for y, v in enumerate(t[col]):
        ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8, color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


hbar(tables["provider"], "mean_daily_change_freq", "Share of days a listing's price changes, by provider",
     "01_change_freq_by_provider.png", xlabel="mean daily probability of a price change")
hbar(tables["tier"], "mean_daily_change_freq", "Share of days a listing's price changes, by GPU tier",
     "02_change_freq_by_tier.png", xlabel="mean daily probability of a price change")
hbar(tables["region_group"], "mean_daily_change_freq", "Share of days a listing's price changes, by region",
     "03_change_freq_by_region.png", xlabel="mean daily probability of a price change")

# spot vs on-demand by provider (two series -> legend + direct labels)
pm = tables["provider_x_market"]["mean_daily_change_freq"].unstack().fillna(0)
pm = pm.loc[pm.max(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(7, 0.42 * len(pm) + 1.2))
y = np.arange(len(pm))
for off, col, c in [(-0.2, "on-demand", BLUE), (0.2, "spot", ORANGE)]:
    if col in pm:
        ax.barh(y + off, pm[col], height=0.36, color=c, label=col)
ax.set_yticks(y, pm.index)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Price change frequency: spot vs on-demand, by provider", loc="left", fontsize=11)
ax.set_xlabel("mean daily probability of a price change")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "04_spot_vs_ondemand.png"), dpi=150); plt.close(fig)

# change frequency by GPU count, spot vs on-demand
cm = tables["gpu_count_x_market"]["mean_daily_change_freq"].unstack().reindex(COUNT_ORDER).dropna(how="all").fillna(0)
fig, ax = plt.subplots(figsize=(7, 3.6))
x = np.arange(len(cm))
for off, col, c in [(-0.2, "on-demand", BLUE), (0.2, "spot", ORANGE)]:
    if col in cm:
        ax.bar(x + off, cm[col], width=0.36, color=c, label=col)
ax.set_xticks(x, [f"{i} GPU" for i in cm.index])
ax.legend(frameon=False)
ax.set_title("Price change frequency by GPUs per instance", loc="left", fontsize=11)
ax.set_ylabel("mean daily probability of a price change")
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "08_change_freq_by_gpu_count.png"), dpi=150); plt.close(fig)

# H100 price paths, small multiples per market
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, m in zip(axes, ["on-demand", "spot"]):
    sub = h[h.market == m]
    top = sub.groupby("provider").size().sort_values(ascending=False).index[:8]
    for p in top:
        s = sub[sub.provider == p]
        ax.plot(s.day, s.ppg, lw=1.6, label=p)
        ax.text(s.day.iloc[-1], s.ppg.iloc[-1], " " + p, fontsize=7, color=MUTED, va="center")
    ax.set_title(f"H100 median $/GPU-hr by provider ({m})", loc="left", fontsize=11)
    ax.set_ylabel("$ per GPU-hour")
    ax.legend(frameon=False, fontsize=7, ncol=2)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "05_h100_paths.png"), dpi=150); plt.close(fig)

# index-level annualized vol (top 20 provider/gpu/market indexes)
top = ivol.sort_values("ann_vol", ascending=False).head(20).set_index(
    ivol.sort_values("ann_vol", ascending=False).head(20).apply(lambda x: f"{x.provider} {x.gpu_type} {x.market}", axis=1))
hbar(top, "ann_vol", "Most volatile price indexes (median $/GPU-hr, annualized vol)",
     "06_top_index_vol.png", fmt="{:.0%}", xlabel="annualized volatility of daily log changes")

# ------------------------------------------------------------------ summary
pd.set_option("display.width", 200, "display.max_columns", 20)
print(f"listings with >= {MIN_OBS} daily returns: {len(ser):,}   returns: {len(r):,}")
for name in ["provider_class", "market", "tier", "provider", "region_group"]:
    print(f"\n== by {name} ==\n{tables[name]}")
print("\n== provider x market ==\n", tables["provider_x_market"])
print("\n== top 20 index vols ==\n", ivol.sort_values("ann_vol", ascending=False).head(20).round(3).to_string(index=False))
print("\n== cross-provider dispersion (max/min price across providers, same day) ==\n", disp)
print("\n== flagship drift (median across providers, $/GPU-hr) ==\n", drift.iloc[:, [0, 3, 6, -2, -1]])
print("\n== by GPU count x market ==\n", tables["gpu_count_x_market"])
print("\n== by tier x GPU count ==\n", tables["gpu_count_x_tier"])
print("\n== per-GPU price vs 1-GPU instance (same provider/chip/region/day) ==\n", premium)
print("\n", premium_by_provider)
print("\n== chained matched-listing index ==\n", chain_sum.sort_values("ann_vol", ascending=False).head(30))
