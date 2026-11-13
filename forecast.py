"""How predictable is AWS spot? Rolling out-of-sample forecasts of each listing's price change.

Target: log price change over the next h days (h = 1, 7, 14), only where every day is observed.
Models (all pooled across listings, no intercept, so none of them bets on the overall drift):
  1. no change        forecast 0 (the price stays where it is)
  2. momentum         forecast = b * today's change
  3. recent history   forecast = b0 * today's change + ... + b4 * the change 4 days ago
Test: every week, refit on rows whose target window ended before that week starts, then forecast
every listing-day in the week. Models never see the future. All models are scored on the same rows.
Scores: RMSE of the h-day change (percentage points of log price), skill = 1 - MSE / MSE(no change),
mean absolute error and its skill (1 - MAE / MAE(no change); squared errors are dominated by a few
huge jumps, absolute errors much less),
share of moves whose direction is called right (rows where the price actually moved), and the same
after today's largest moves (top quarter of |today's change|). Skill gets a 95% interval from
resampling whole test weeks (forecast errors within a week overlap and share shocks).
Universe and cleaning: common.aws_spot_active on the cleaned return grid (stale-feed days and
catch-up jumps excluded). Outputs: out/forecast_scores.csv, out/forecast_coefficients.csv.
"""
import os

import numpy as np
import pandas as pd

from common import EPS, OUT, aws_spot_active, load_returns

HORIZONS = (1, 7, 14)
LAGS = 5
MIN_TRAIN = 3000  # training rows before the first test week
N_BOOT = 1000
rng = np.random.default_rng(5)

R, meta, days = load_returns()
A = aws_spot_active(R, meta)
X = A.to_numpy()
T, N = X.shape
t_idx = np.repeat(np.arange(T)[:, None], N, axis=1)

# lagged features: column k = change k days before t (k = 0 is today)
L = np.stack([np.vstack([np.full((k, N), np.nan), X[:T - k]]) for k in range(LAGS)], axis=-1)  # (T, N, LAGS)


def forward(h):
    Y = np.full_like(X, np.nan)
    for t in range(T - h):
        w = X[t + 1:t + 1 + h]
        Y[t] = np.where(np.isnan(w).any(0), np.nan, w.sum(0))
    return Y


def ols(Z, y):
    return np.linalg.lstsq(Z, y, rcond=None)[0]


week = (A.index - A.index[0]).days.to_numpy() // 7
score_rows, coef_rows = [], []
for h in HORIZONS:
    Y = forward(h)
    ok = ~np.isnan(Y) & ~np.isnan(L).any(-1)
    y, F, t = Y[ok], L[ok], t_idx[ok]
    wk = week[t]
    preds = {"1. no change": np.full(len(y), np.nan), "2. momentum": np.full(len(y), np.nan),
             "3. recent history": np.full(len(y), np.nan)}
    for w in np.unique(wk):
        test = wk == w
        start = np.where(week == w)[0][0]
        train = t + h < start  # target window fully observed before the week begins
        if train.sum() < MIN_TRAIN:
            continue
        b2 = ols(F[train, :1], y[train])
        b3 = ols(F[train], y[train])
        preds["1. no change"][test] = 0.0
        preds["2. momentum"][test] = F[test, :1] @ b2
        preds["3. recent history"][test] = F[test] @ b3
        coef_rows.append({"horizon_days": h, "test_week_start": A.index[start].date(),
                          "momentum_b": b2[0], **{f"history_b{k}": v for k, v in enumerate(b3)}})
    tested = ~np.isnan(preds["1. no change"])
    y, F, wk = y[tested], F[tested], wk[tested]
    preds = {k: v[tested] for k, v in preds.items()}
    big = np.abs(F[:, 0]) >= np.quantile(np.abs(F[:, 0]), 0.75)
    moved = np.abs(y) > EPS
    weeks = np.unique(wk)
    se0 = (y - preds["1. no change"]) ** 2
    for name, p in preds.items():
        se = (y - p) ** 2
        # skill CI: resample whole test weeks
        wk_se = pd.DataFrame({"w": wk, "se": se, "se0": se0, "ae": np.abs(y - p), "ae0": np.abs(y)}).groupby("w").sum()
        boot, boot_mae = [], []
        for _ in range(N_BOOT):
            s = wk_se.loc[rng.choice(weeks, len(weeks))].sum()
            boot.append(1 - s.se / s.se0)
            boot_mae.append(1 - s.ae / s.ae0)
        hit = np.sign(p[moved]) == np.sign(y[moved])
        hit_big = np.sign(p[moved & big]) == np.sign(y[moved & big])
        score_rows.append({
            "horizon_days": h, "model": name, "test_rows": len(y), "test_weeks": len(weeks),
            "first_test_day": A.index[np.where(week == weeks.min())[0][0]].date(),
            "rmse_pct": 100 * np.sqrt(se.mean()),
            "skill_vs_no_change": 1 - se.sum() / se0.sum(),
            "skill_p025": np.percentile(boot, 2.5), "skill_p975": np.percentile(boot, 97.5),
            "mae_pct": 100 * np.abs(y - p).mean(),
            "mae_skill": 1 - np.abs(y - p).sum() / np.abs(y).sum(),
            "mae_skill_p025": np.percentile(boot_mae, 2.5), "mae_skill_p975": np.percentile(boot_mae, 97.5),
            "direction_right": np.nan if name.startswith("1") else hit.mean(),
            "rmse_pct_after_big_moves": 100 * np.sqrt(se[big].mean()),
            "skill_after_big_moves": 1 - se[big].sum() / se0[big].sum(),
            "direction_right_after_big_moves": np.nan if name.startswith("1") else hit_big.mean(),
        })

scores = pd.DataFrame(score_rows)
scores.round(4).to_csv(os.path.join(OUT, "forecast_scores.csv"), index=False)
coefs = pd.DataFrame(coef_rows)
coefs.round(4).to_csv(os.path.join(OUT, "forecast_coefficients.csv"), index=False)

pd.set_option("display.width", 250, "display.max_columns", 20)
print(f"AWS spot listings: {N}")
print(scores.round(3).to_string(index=False))
print("\ncoefficients (last test week):\n", coefs.groupby("horizon_days").last().round(3).to_string())
