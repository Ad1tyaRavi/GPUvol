# GPU rental price volatility

Do cloud GPU rental prices behave like a financial market? In particular, does volatility
cluster, with big moves followed by more big moves?

The data is [`afhubbard/gpu-prices`](https://huggingface.co/datasets/afhubbard/gpu-prices): list
prices from about 15 providers, scraped 1-3 times a day from January to November 2026 (~3.7M
rows).

Most providers don't behave like a market at all. They change prices by hand or on a schedule.
The one big market that reprices daily is AWS spot (~520 listings, a change on 94% of days).

## AWS spot

**Moves keep going.** If a listing moves today, the next 7 days add another 1.45x that move on
average (95% interval 1.13-1.73), in the same direction 60% of the time (about 70% after big
moves). The effect is gone by day 5. Fitted through August, the slope still works on September to
November: right direction 60% of the time, R² 0.13. The interval resamples whole two-week blocks,
since nearby days overlap and listings move together.

![AWS spot momentum](docs/img/aws_momentum.png)

**It's a bit forecastable.** A forecast from today's move, refit weekly on past data, compared
with assuming no change. The September to November test uses listings chosen on earlier data only:

| | 1 day | 7 days | 14 days |
|---|---|---|---|
| squared error, Sep-Nov | -22% | -12% | -5% |
| squared error, whole year | -2% (not significant) | -9% | -3%* |
| absolute error | -14% | -5% | -2% |
| direction right | 71% | 59% | 55% |

\*The 14-day forecast is only tested from mid-August. Absolute error and direction are about the
same over the whole year. Squared error is not: earlier in the year a few big jumps that nothing
predicts dominate it. Good enough to time spot jobs, not to predict prices.

**Listings barely move together.** The same instance type in two regions correlates at 0.014
daily. Different chips in the same region correlate at 0.074. So region seems to matter more than
chip, though the scraper fetching one region at a time could explain some of that. There's no
single "H100 spot price".

![AWS spot local markets](docs/img/aws_local_markets.png)

**Volatility clusters, mostly over weeks.** After a quiet day there's a 19% chance of a change,
against 35% when each listing's days are shuffled within the month. It isn't just prices stuck at
the on-demand cap (spot never goes above it): without those days it's 24% against 39%.

Move sizes cluster too. With momentum removed, consecutive move sizes correlate at 0.17, against
0.00 when shuffled. Shuffling within each month still gives 0.11, so most of it is calm and busy
stretches lasting weeks. A third of listings beat their own within-month shuffle (5% expected).
GARCH(1,1) finds little: 8% of listings beat a calibrated threshold, against 5% on shuffled data.
It's a poor fit for mostly tiny moves with the odd jump.

## Other providers

Each one prices its own way. GCP reprices spot in weekly batches, usually on Mondays. Verda moves
on-demand prices in runs. Nebius was static until 8 October, then went dynamic. OCI, Vultr, Cudo
and Daytona never changed a price. I found no sign of providers repricing together, but outside
AWS there are too few changes to say much.

![Four providers, four pricing styles](docs/img/pricing_styles.png)

Spot prices rose over the year: H100 spot is up 38% counting only day-to-day changes, or 77%
including moves made during data gaps. On-demand changes rarely but in big steps (H100 +16%, H200
+18%, B200 +32%), almost all on days the scraper missed.

![Spot price index by chip](docs/img/spot_index.png)

## Method

- One series per listing (provider, instance type, region, spot or on-demand), priced daily from
  the snapshot closest to 09:00 UTC, using consecutive-day log returns.
- Listings with identical price paths count once (GCP copies one price across many regions).
- Clustering is tested against each listing's own data shuffled within the same weekday and
  month, so weekly schedules don't count.

## Data problems

`audit.py` checks for these.

- No data from 10 March to 7 May, plus a few short outages.
- The feed froze for every provider from 8 to 24 May and on 1 June, and for AWS and GCP from 25
  June to 22 July. I treat those days as missing. This matters: left in, the momentum slope drops
  to 0.58x and the forecast gains vanish.
- Snapshot times are irregular, and `available` is always true, so there's no capacity signal.
- "A100" mixes 40/80 GB and SXM/PCIe cards, and GPU slices are labeled as whole GPUs.
- It's list prices only, over one year in a rising market.

## Running it

```bash
pip install pandas numpy scipy matplotlib pyarrow
python download.py
python analysis.py && python aws_spot.py && python forecast.py
python clustering.py && python event_clustering.py && python market_sync.py && python garch.py
python audit.py && python readme_figures.py
```

Everything writes to `out/`. Set `GPUVOL_KEEP_SUSPECT=1` to rerun without the frozen-feed fixes.
