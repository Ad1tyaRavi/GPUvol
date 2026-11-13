# GPU rental price volatility

How do cloud GPU rental prices move, and does their volatility cluster the way it does in
financial markets?

Data: [`afhubbard/gpu-prices`](https://huggingface.co/datasets/afhubbard/gpu-prices), listed
prices from 13+ providers (AWS, GCP, Azure, OCI, Lambda, RunPod, Nebius, Verda, Vultr, ...),
scraped 1-3 times a day from January to November 2026. About 3.7M GPU listing rows.

**Short answer: mostly not.** Most providers set prices by hand or on a schedule. The only large
market that reprices every day is AWS spot (~520 listings, a change on 94% of days), so that is
where the interesting results are. There, what stands out is not volatility clustering but
momentum: price moves carry on for a few days, enough for a simple forecast to beat "no change".

## AWS spot

**1. Prices have momentum: today's move predicts the next week.** On average, the next 7 days
add another 1.45x today's move, in the same direction. The direction carries on 60% of the time,
and 69% after the largest moves. Fit through August and tested on September-November, it holds
up (1.33x, 61%). Momentum lasts about five days (autocorrelation 0.50, 0.29, 0.18 at lags 1-3),
then turns into a slight pullback for several weeks. In a traded market this would be traded
away; here nobody can arbitrage GPU-hours, and AWS adjusts spot prices gradually by design. It
is the strongest and most robust result in the repo: it held through every data fix and works
out of sample.

![AWS spot momentum](docs/img/aws_momentum.png)

**2. AWS spot is hundreds of local markets, not one market per chip.** Two listings' price
changes barely move together, even for the same instance type in different regions (median
correlation 0.014 daily, 0.035 weekly). What little they share is regional: different chips in
the same region correlate 0.074 daily and 0.123 weekly. All of these beat a baseline where
listings are shifted in time, but they are small. So there is no single "H100 spot price":
prices seem to follow local capacity, and a chip-level index averages many largely independent
paths.

![AWS spot local markets](docs/img/aws_local_markets.png)

**3. "Clustering" here is about when prices change, not how much.** Volatility clustering in
finance means big moves follow big moves. For AWS spot:
- **Timing clusters strongly.** Listings change on 94% of days, but the quiet days come in runs:
  the chance of a change is 99% after a day with a change but 19% after a quiet day, and 73% of
  quiet days fall in runs of 3+ days.
- **About a quarter of those quiet days are the on-demand ceiling.** Spot never goes above the
  on-demand price for the same instance. At that ceiling a listing changes on 2% of days, against
  94% below it. Listings sit there on 2% of days but those days make up 25% of all quiet days.
  H100 and H200 spot sit at the on-demand price on 11-12% of days, with no discount at all.
- **Size clusters only weakly.** On consecutive change days, the size of one move correlates 0.39
  with the next (0.21 by chance), but most of that is the momentum above. With momentum removed
  it is 0.17 against 0.14 by chance: real but small. GARCH(1,1) agrees, finding clustering in 8%
  of listings against 5% by chance.

**4. A simple forecast beats "no change", modestly.** Each week, models are refit on the past only
and forecast every listing's price change over the next 1, 7 and 14 days:

| Horizon | Model | Error vs no change (absolute) | Error vs no change (squared) | Direction right | After the largest moves |
|---|---|---|---|---|---|
| 1 day | momentum | −14% | −2% (n.s.) | 71% | 87% |
| 1 day | recent history | −15% | −5% (n.s.) | 70% | 86% |
| 7 days | momentum | −5% | −9% | 60% | 70% |
| 7 days | recent history | −5% | −9% | 60% | 70% |
| 14 days | momentum | −2% | −3% | 55% | 63% |
| 14 days | recent history | −2% | −4% | 56% | 63% |

*Momentum* forecasts a multiple of today's change; *recent history* uses the last five days'
changes. The test covers Feb-Nov for 1 and 7 days and Aug-Nov for 14 days. Every gain has a 95%
interval (from resampling whole weeks) that excludes zero, except the 1-day squared error, which
a few huge jumps dominate. The last five days add almost nothing beyond today's change. AWS spot
is partly predictable, mostly in direction and for a few days: useful for timing spot jobs, but
the errors stay large.

## Across providers

**Every provider has its own pricing machine.** AWS nudges spot prices almost daily. GCP
reprices in batches, mostly on Mondays. Verda moves on-demand prices in steady runs. Nebius was
static until 8 Oct 2026, then switched to daily dynamic pricing. OCI, Vultr, Cudo and Daytona
never changed a price.

![Four providers, four pricing styles](docs/img/pricing_styles.png)

**Providers reprice alone.** Across providers, 1 of 36 activity and direction tests beat chance,
and 0 of 45 tests for the same chip: no more than chance would produce. Within GCP and Verda,
listings change together (implied pairwise correlation 0.55 and 0.29 after removing weekly
schedules).

**Spot prices rose; on-demand rose only in a few list-price hikes.** Spot prices went up across
most chips. The direction holds however it is measured, but the size depends on moves we never
saw happen. Counting only day-to-day changes, H100 spot rose 38% and B200 46%. Also counting moves
across data gaps, H100 rose 77% and B200 88%. On-demand prices changed on 0.2% of listing-days;
the visible on-demand rises are list-price hikes by Lambda (about +14%, twice) and Verda, all of
which fell on days the scraper missed.

![Spot price index by chip](docs/img/spot_index.png)

**Smaller findings.** The same chip costs very different amounts at different providers (H100
~4.4x between the dearest and cheapest provider on a typical day, B200 ~2.6x; A100 ~6.3x mixes
40/80 GB and SXM/PCIe). Most providers charge a flat per-GPU price across node sizes.

## Method

- One series per listing (provider, instance, region, spot/on-demand, config). The daily price
  is the snapshot closest to 09:00 UTC, so consecutive prices are ~24h apart (5-95%: 20-27h).
- Volatility, momentum and clustering use log returns between consecutive days only.
- Listings that move in lockstep (one price copied across regions or node sizes) are counted
  once. Listings that never move are all kept.
- Every clustering and co-movement result is compared with a shuffled baseline: shuffles within
  the same weekday and 30-day block (so weekly schedules and slow drifts don't count), or
  shifting whole listings in time by whole weeks.
- Trends use a matched-listing index. A return that spans a gap the listing couldn't have been
  seen in (no snapshots, or a stale feed for its provider) is kept, so moves made while
  unobserved still count; a listing missing while the rest of its provider is scraped isn't
  bridged.
- AR filter, Ljung-Box and GARCH(1,1) are written out by hand (numpy/scipy) to handle missing
  days.

## Data quirks

`audit.py` checks for most of these; `common.py` holds the fixes.

- **Gaps.** No snapshots at all from 2026-03-10 to 05-07, plus short outages (01-19, 02-10..13, 02-28,
  07-09..13, 09-07).
- **Frozen feed.** Every provider except Vast.ai froze from 2026-05-08 to 05-24 and on 06-01;
  AWS and GCP froze from 06-25 to 07-22. Frozen snapshots look like real ones; the only giveaway
  is AWS spot, which normally changes daily, showing no changes (`audit.py` lists such days).
  Frozen days are treated as missing. Each catch-up jump the next day counts toward trends but
  not volatility.
- **Irregular snapshot times.** One snapshot a day at ~09:20 UTC before March; from May, 1-3 a
  day at 09:00, 21:00 and irregular times in between. At least one irregular snapshot (05-30
  10:44) is a stale copy. Hence the 09:00 sampling.
- **No availability data.** `available` is true on every row and `availability_zone` is always
  empty.
- **Coarse GPU labels.** "A100" mixes 40 and 80 GB, SXM and PCIe. GPU slices (vGPU) are labeled
  as one GPU; for Vultr, Azure and GCP they are converted to fractions of a card using GPU memory.
- **Copied prices.** GCP and Verda set one price for many regions or node sizes; identical price
  paths are counted once.
- **Catalog changes.** Nebius switched pricing systems on 2026-10-08 (two separate series).
  DataCrunch rows duplicate Verda after its rebrand (dropped). RunPod swapped most of its catalog
  on 05-25. Vast.ai offers last about a day, so Vast.ai is left out of indexes and most tests.
- **CPU-only rows.** About 55% of rows are CPU-only machines (`quality == "cpu_only"`) and are
  dropped.

## Caveats

- List prices only: no transactions, volumes, availability or reserved contracts. The dataset
  marks every listing as available, so it can't show when capacity runs out.
- About 230 days, all in a rising market, with ~2 months of it unobserved.
- The regional link between AWS listings could partly come from the scraper fetching a region
  in one batch. Quiet runs could partly be one listing's price going stale in the feed.
- Chip indexes for marketplace and workstation GPUs rest on 3-8 listings a day.

## Files

| File | |
|---|---|
| `download.py` | fetch the dataset into `data/` |
| `aws_spot.py` | AWS spot momentum, local markets, timing vs size, on-demand ceiling |
| `forecast.py` | rolling out-of-sample forecasts of AWS spot price changes |
| `analysis.py` | volatility by axis, price indexes, cross-provider spreads |
| `clustering.py` | autocorrelation and Ljung-Box tests |
| `event_clustering.py` | do price changes come in bursts? |
| `market_sync.py` | do listings and providers reprice together? |
| `garch.py` | GARCH(1,1) with calibrated significance |
| `audit.py` | checks for scraper and schedule artifacts |
| `common.py` | shared loading, daily sampling and cleaning |
| `readme_figures.py` | the charts in this README (`docs/img/`) |

Results and charts go to `out/`. `all.parquet` and the return cache are rebuilt automatically
when new snapshots are downloaded. Set `GPUVOL_KEEP_SUSPECT=1` to rerun without the stale-feed
fixes (outputs in `out/with_suspect_days/`).

```bash
pip install pandas numpy scipy matplotlib pyarrow
python download.py
python analysis.py && python aws_spot.py && python forecast.py
python clustering.py && python event_clustering.py && python market_sync.py && python garch.py
python readme_figures.py
```
