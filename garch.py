"""GARCH(1,1) on AR-filtered spot returns: how persistent is volatility?

    e_t = sigma_t z_t,    sigma_t^2 = omega + alpha e_{t-1}^2 + beta sigma_{t-1}^2

alpha        = how much yesterday's surprise raises today's variance (reaction)
beta         = how much yesterday's variance carries over (memory)
alpha + beta = persistence; half-life of a volatility shock = ln(0.5) / ln(alpha + beta) days

Fitted by Gaussian quasi-maximum likelihood (QMLE), written out here rather than via the `arch`
package so every step is visible. Missing days: the variance recursion still runs, with the unseen
e^2 replaced by its expectation, i.e. sigma^2 <- omega + (alpha + beta) sigma^2.

Series: AWS spot listings and spot GPU indexes (same universe as clustering.py), after the same AR
filter, so GARCH only sees the unpredictable part of each move.
Tests:
  * LR test vs constant variance (alpha = beta = 0): LR = 2 (ll_garch - ll_const).
    The textbook chi2(2) threshold is badly oversized here: on shuffled residuals (no clustering
    by construction) it rejects ~30-45% of the time, because the residuals are zero-heavy with
    occasional jumps, far from Gaussian. So the threshold is calibrated instead: refit after
    shuffling each series within weekday x 30-day blocks (keeps weekly schedules and slow regime
    shifts, destroys short-horizon timing), and call a series significant if its LR beats
    the shuffled 95th percentile (per series, 20 shuffles, for the indexes; pooled over listings,
    1 shuffle each, for AWS).
  * Persistence alpha + beta only means something when alpha > 0: with alpha = 0 the variance is
    constant and beta is not identified. Persistence and half-life are reported for alpha >= 0.01.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from common import BLUE, MUTED, ORANGE, OUT, ar_filter, aws_spot_active, load_returns, shuffle_cols_blocked, spot_chained_indexes

SCALE = 100.0  # fit on percent returns for numerical stability
rng = np.random.default_rng(3)


def _unpack(x):
    x = np.clip(x, -30, 30)
    omega = np.exp(x[0])
    persist = 0.999 / (1 + np.exp(-x[1]))   # alpha + beta in (0, 0.999)
    share = 1 / (1 + np.exp(-x[2]))         # alpha's share of persistence
    return omega, persist * share, persist * (1 - share)


def neg_loglik(x, e):
    omega, alpha, beta = _unpack(x)
    obs = e[~np.isnan(e)]
    h = obs.var() if obs.var() > 0 else 1e-6
    ll = 0.0
    for v in e:
        if v == v:  # observed (not NaN)
            ll += np.log(h) + v * v / h
            h = omega + alpha * v * v + beta * h
        else:       # missing day: expected update
            h = omega + (alpha + beta) * h
        h = max(h, 1e-12)
    return 0.5 * ll


def fit_garch(e):
    """Fit GARCH(1,1) to one residual series (NaN = missing day). Returns a dict of results."""
    e = e * SCALE
    first = np.argmax(~np.isnan(e))
    e = e[first:]
    obs = e[~np.isnan(e)]
    n, s2 = len(obs), obs.var()
    if n < 60 or s2 <= 0:
        return None
    ll_const = -0.5 * (n * np.log(s2) + n)
    best = None
    for start in ([np.log(s2 * 0.05), 2.2, -2.0], [np.log(s2 * 0.5), -0.8, 0.0], [np.log(s2 * 0.2), 0.8, -0.5]):
        r = minimize(neg_loglik, np.array(start), args=(e,), method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    omega, alpha, beta = _unpack(best.x)
    ll = -best.fun
    pers = alpha + beta
    lr = max(2 * (ll - ll_const), 0.0)
    return {"n": n, "omega": omega, "alpha": alpha, "beta": beta, "persistence": pers,
            "half_life_days": np.log(0.5) / np.log(pers) if 0 < pers < 1 else np.nan,
            "lr_stat": lr, "lr_p": chi2.sf(lr, 2), "converged": bool(best.success)}


ALPHA_MIN = 0.01


def fit_all(E, label, reps=1):
    """Fit every column, plus `reps` shuffled refits per column for the calibrated LR threshold."""
    rows, null_rows = {}, []
    X = E.to_numpy()
    for j, col in enumerate(E.columns):
        r = fit_garch(X[:, j])
        if r is not None:
            rows[col] = r
    for k in range(reps):
        S = shuffle_cols_blocked(X, rng, E.index)
        for j, col in enumerate(E.columns):
            rn = fit_garch(S[:, j])
            if rn is not None:
                null_rows.append({"series": col, **rn})
    res = pd.DataFrame(rows).T.astype(float)
    null = pd.DataFrame(null_rows).set_index("series").astype(float)
    if reps > 1:   # per-series threshold
        res["lr_crit"] = null.groupby(level=0).lr_stat.quantile(0.95)
        null["sig_calibrated"] = null.lr_stat > res.lr_crit.reindex(null.index).to_numpy()
    else:          # pooled threshold across series
        res["lr_crit"] = null.lr_stat.quantile(0.95)
        null["sig_calibrated"] = null.lr_stat > res.lr_crit.iloc[0]
    res["sig_calibrated"] = res.lr_stat > res.lr_crit
    for d in (res, null):
        d.loc[d.alpha < ALPHA_MIN, ["persistence", "half_life_days"]] = np.nan

    def summ(d, name):
        return pd.Series({"fits": len(d), "median_alpha": d.alpha.median(),
                          "share_alpha_ge_0.01": (d.alpha >= ALPHA_MIN).mean(),
                          "median_persistence_if_alpha": d.persistence.median(),
                          "median_half_life_days_if_alpha": d.half_life_days.median(),
                          "share_LR_sig_textbook_chi2": (d.lr_p < 0.05).mean(),
                          "share_LR_sig_calibrated": d.sig_calibrated.mean(),
                          "share_converged": d.converged.mean()}, name=name)

    return res, null, pd.concat([summ(res, label), summ(null, label + " (shuffled)")], axis=1).T


R, meta, days = load_returns()

E_aws, _ = ar_filter(aws_spot_active(R, meta))
aws_res, aws_null, aws_sum = fit_all(E_aws, "AWS spot listings")
aws_res.join(meta).round(4).to_csv(os.path.join(OUT, "garch_aws_spot_series.csv"))

E_idx, _ = ar_filter(spot_chained_indexes(R, meta))
idx_res, idx_null, idx_sum = fit_all(E_idx, "spot GPU indexes", reps=20)
idx_res.round(4).sort_values("persistence", ascending=False).to_csv(os.path.join(OUT, "garch_spot_index_series.csv"))

summary = pd.concat([aws_sum, idx_sum])
summary.round(4).to_csv(os.path.join(OUT, "garch_summary.csv"))

# ------------------------------------------------------------------ chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
bins = np.linspace(0, 0.6, 31)
ax.hist(aws_null.alpha.clip(upper=0.6), bins=bins, color=MUTED, alpha=0.35, label="shuffled (no clustering)")
ax.hist(aws_res.alpha.clip(upper=0.6), bins=bins, histtype="step", lw=2.2, color=ORANGE, label="actual")
ax.set_xlabel("alpha: how much a surprise raises next day's variance")
ax.set_ylabel("AWS spot listings")
ax.set_title(f"AWS spot listings (n={len(aws_res)})", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(axis="x", visible=False)

ax = axes[1]
t = idx_res.assign(p=idx_res.persistence.fillna(0)).sort_values(["p", "alpha"])
y = np.arange(len(t))
ax.barh(y, t.p, color=[ORANGE if s else MUTED for s in t.sig_calibrated], height=0.6)
for yi, (p, a, s) in enumerate(zip(t.persistence, t.alpha, t.sig_calibrated)):
    txt = f" {p:.2f}" if a >= ALPHA_MIN else " alpha~0: no GARCH effect"
    ax.text(0 if np.isnan(p) else p, yi, txt + ("" if s else "  (n.s.)"), va="center", fontsize=8, color=MUTED)
ax.set_yticks(y, t.index)
ax.set_xlim(0, 1.35)
ax.set_xlabel("persistence alpha + beta  (orange = beats shuffled LR threshold)")
ax.set_title("Spot GPU indexes", loc="left", fontsize=11)
ax.grid(axis="y", visible=False)
fig.suptitle("GARCH(1,1): how long do volatility shocks last?", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "12_garch_persistence.png"), dpi=150)
plt.close(fig)

pd.set_option("display.width", 220, "display.max_columns", 20)
print("== summary ==\n", summary.round(3))
print("\n== spot indexes ==\n", idx_res[["n", "alpha", "beta", "persistence", "half_life_days", "lr_stat",
                                       "lr_crit", "sig_calibrated", "lr_p"]].round(3).sort_values("alpha", ascending=False))
