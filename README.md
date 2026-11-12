# GPU rental price volatility

How do cloud GPU rental prices move, and does their volatility cluster the way it does in
financial markets?

Data: [`afhubbard/gpu-prices`](https://huggingface.co/datasets/afhubbard/gpu-prices), listed
prices from 13+ providers (AWS, GCP, Azure, OCI, Lambda, RunPod, Nebius, Verda, Vultr, ...),
scraped 1-3 times a day from January to November 2026. About 3.7M GPU listing rows.

## Findings

**Spot moves, on-demand doesn't.** Spot prices change on about 17% of days, on-demand on less
than 1%. Over the year H100 spot roughly doubled while on-demand B200 and H200 rose 7-12%.

**Every provider has its own pricing machine.** AWS nudges spot prices almost daily in ~1.5%
steps. GCP reprices in batches on Mondays. Verda is the only one with dynamic on-demand prices,
and it goes quiet on weekends. Nebius was static until 20 Oct 2026, then switched to daily
dynamic pricing. OCI, Lambda and Vultr barely move at all.

**AWS spot volatility clusters.** A price change makes another one far more likely over the
next week or two (about 11x vs 2x by chance). It's strongest for Hopper and Blackwell chips. The
clustering is in many small moves rather than big jumps, so GARCH only picks up a weak version
of it.

**AWS spot prices trend.** Day-to-day returns are autocorrelated (~0.26 at lag 1), which you
wouldn't see in a traded market. Nobody can arbitrage GPU-hours, so nothing removes it.

**Providers reprice alone.** Within a provider, listings change together (pairwise correlation
0.70-0.92). Across providers there's no more co-movement than chance, even for the same chip.

**Same chip, very different prices.** On a typical day the priciest A100 is ~7.8x the cheapest,
H100 ~4.6x, B200 ~2.6x. Older chips are priced much less consistently.

**Other axes matter less.** GPU tier, region and node size differences mostly reflect which
providers sell what. Most providers charge a flat per-GPU price; 16+ GPU nodes are 8-19% cheaper
per GPU.

## What didn't hold up

Several early results turned out to be artifacts, caught by `audit.py`:

- Clustering in the H200/B200 indexes and in Nebius was weekly schedules and Nebius's October
  regime switch.
- Cross-provider links came from one or two odd days, mainly 2026-07-23, when AWS spot jumped
  ~10% across the board and GCP repriced off-schedule. Those days are excluded.
- GCP and Verda list one price under many regions, so their listings were double counted.
- The textbook GARCH likelihood-ratio test flagged ~30% of *shuffled* series. Significance is
  calibrated against shuffled data instead.

## Method

- One series per listing (provider, instance, region, spot/on-demand, config); daily log returns
  on consecutive days only.
- Volatility split into how often a price changes and how big the change is, since prices sit
  flat and then jump.
- Trends use a matched-listing index (only listings present on both days), so listings coming
  and going don't move it.
- Clustering tests are compared against shuffles within the same weekday and 30-day block, so
  weekly schedules and slow regime changes don't count.
- AR filter, Ljung-Box and GARCH(1,1) are written out by hand (numpy/scipy) to handle missing
  days.

## Caveats

List prices only: no transactions, volumes or reserved contracts. About 230 days, all in a
rising market. Listings are equally weighted, so AWS and GCP dominate. `analysis.py` predates the
audit fixes, so its trend numbers (e.g. H100 spot +96%) include 2026-07-23 and GCP duplicates.

## Files

| File | |
|---|---|
| `download.py` | fetch the dataset into `data/` |
| `analysis.py` | volatility by axis, price indexes, cross-provider spreads |
| `clustering.py` | autocorrelation and Ljung-Box tests |
| `event_clustering.py` | do price changes come in bursts? |
| `market_sync.py` | do listings and providers reprice together? |
| `garch.py` | GARCH(1,1) with calibrated significance |
| `audit.py` | checks for scraper and schedule artifacts |
| `common.py` | shared loading and cleaning |

Results and charts go to `out/`. Set `GPUVOL_KEEP_SUSPECT=1` to rerun with the excluded days
kept (outputs in `out/with_suspect_days/`).

```bash
pip install pandas numpy scipy matplotlib pyarrow
python download.py
python analysis.py
python clustering.py && python event_clustering.py && python market_sync.py && python garch.py
```
