# GPU rental price volatility

How volatile are cloud GPU rental prices, and along which axes (provider, spot vs on-demand,
GPU tier, region, GPUs per instance)?

## Data

[`afhubbard/gpu-prices`](https://huggingface.co/datasets/afhubbard/gpu-prices) (CC-BY-4.0):
listed prices from 13+ cloud providers (AWS, GCP, Azure, OCI, Lambda, RunPod, Vast.ai,
Nebius, Verda, Vultr, ...), scraped 1-3x daily via `gpuhunt`, Jan-Nov 2026.
One row = one listing at one snapshot (provider, instance type, region, GPU type/count/memory,
spot flag, $/hr). These are *list* prices: no transaction, volume or reserved-contract data.

## Method

**Cleaning:** GPU listings only; price normalized to $/GPU-hour; DataCrunch dropped (rebranded to
Verda, duplicate rows); fractional vGPU slices (Vultr, Azure A10, GCP RTX PRO 6000) converted to
effective GPUs by memory share; Vast.ai excluded from index-level results (offers churn daily and
only a few are sampled per GPU).

1. **Listing-level volatility:** a series = one exact SKU (provider, instance, region, spot, config).
   Daily log returns on consecutive days only. Prices are sticky and move in jumps, so volatility is
   split into **change frequency** (share of days the price moves) and **jump size** (median |return|
   when it moves), alongside the usual std. Summarized by provider, provider class, market, GPU tier,
   region group and GPU count.
2. **Provider index:** daily median $/GPU-hr per provider x GPU x market.
3. **Matched-listing (chained) index:** each day's return is the mean return over listings present on
   both days, so the trend is not distorted by listings entering/leaving.
4. **Cross-provider dispersion:** max/min $/GPU-hr across providers for the same GPU on the same day.
5. **GPU count:** volatility by GPUs per instance, and per-GPU price of n-GPU vs 1-GPU instances.

## Findings (so far)

- Spot prices change on ~17% of days; on-demand on <1%.
- Providers run very different spot regimes: AWS moves ~75% of days in ~1.5% steps; GCP ~8% of days
  in ~5% steps; Nebius rarely but ~38% jumps; Azure almost never. Verda is the only provider with
  frequently changing on-demand prices.
- Matched-listing index, Jan to Nov 2026: H100 spot +96%, B200 spot +114%, while on-demand B200 +7%,
  H200 +12%. Spot absorbs demand pressure; list prices barely move.
- Same chip, same day: priciest/cheapest provider is ~4.6x for H100, ~7.8x for A100.
- Per-GPU pricing is mostly linear in node size; 16+ GPU nodes are ~8-19% cheaper per GPU.

## Caveats

Listing-weighted (GCP/AWS regional SKUs dominate); list prices only; √365 annualization is a poor fit
for a jump process; region effects are largely provider mix; ~230 days of one market regime.

## Run

```bash
pip install pandas pyarrow numpy matplotlib
python download.py   # fetch parquet snapshots into ./data
python analysis.py   # tables + charts into ./out
```
