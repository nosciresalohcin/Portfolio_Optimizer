# Full-universe portfolio analysis (ready for Colab / local)
# Requirements: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, statsmodels
# Optional but recommended: joblib (for parallel), pandas_datareader (for T-bill)
# If LedoitWolf is slow or unavailable, the script falls back to a fast shrink.

import os, math, warnings, numpy as np, pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt, seaborn as sns
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')


# ---------------- USER CONFIG ----------------
CSV_PATH = "sp500_clean_wide.csv"   # change if needed
DATE_COL = 0            # index or name of the date column (0 means first column)
RF_ANNUAL = 0.04
TRADING_DAYS = 252
MIN_OBS = 2500           # minimum observations to include asset
MAX_WEIGHT = 0.05
MIN_WEIGHT = 0.0
OUTPUT_DIR = "optimization_full_universe_results"
SHRINK_ALPHA = 0.2      # fast shrink intensity for fallback cov
# ----------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)
# normalize column names
df.columns = [c.strip() for c in df.columns]
if isinstance(DATE_COL, int):
    date_col = df.columns[DATE_COL]
else:
    date_col = DATE_COL
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
df.set_index(date_col, inplace=True)
# numeric convert
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Filter assets with enough history
asset_cols = [c for c in df.columns if df[c].dropna().shape[0] >= MIN_OBS]
prices = df[asset_cols].copy()
rets = prices.pct_change().dropna(how='all').dropna(axis=1, how='all')
assets = list(rets.columns)
n = len(assets)
print(f"Using {n} assets, data from {rets.index.min().date()} to {rets.index.max().date()} ({len(rets)} obs).")

# Asset-level stats
ann_returns = (1 + rets).prod()**(TRADING_DAYS / len(rets)) - 1
ann_vols = rets.std() * (TRADING_DAYS ** 0.5)
asset_stats = pd.DataFrame({'ann_return':ann_returns, 'ann_vol':ann_vols, 'obs':[rets[c].dropna().shape[0] for c in assets]})
asset_stats.to_csv(os.path.join(OUTPUT_DIR, "asset_stats.csv"))

# Covariance: try Ledoit-Wolf, else fallback to sample+capped shrink
print("Estimating covariance (Ledoit-Wolf preferred)...")
try:
    lw = LedoitWolf().fit(rets.fillna(method='ffill').fillna(method='bfill').values)
    Sigma = lw.covariance_
    print("Ledoit-Wolf done.")
except Exception as e:
    print("Ledoit-Wolf failed or slow; using fast shrinked sample covariance:", str(e))
    sample_cov = rets.cov() * TRADING_DAYS
    diag = np.diag(np.diag(sample_cov))
    Sigma = SHRINK_ALPHA * sample_cov.values + (1 - SHRINK_ALPHA) * diag

cov_df = pd.DataFrame(Sigma, index=assets, columns=assets)
cov_df.to_csv(os.path.join(OUTPUT_DIR, "covariance.csv"))

# Shrink expected returns (simple James-Stein style)
hist_mu = asset_stats['ann_return'].values
cross_mean = np.nanmean(hist_mu)
lam = 0.6
mu = pd.Series(lam*hist_mu + (1-lam)*cross_mean, index=assets)
mu.to_csv(os.path.join(OUTPUT_DIR, "shrunk_mu.csv"))

# Quick helper: project weights to bounds and renormalize
def project_and_normalize(w, min_w=MIN_WEIGHT, max_w=MAX_WEIGHT):
    w = np.clip(w, min_w, max_w)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    return w

# Use pseudoinverse for stability
Sigma_reg = Sigma + np.eye(n) * 1e-8
invSigma = np.linalg.pinv(Sigma_reg)

# 1) Approx min-variance (approx closed-form): invSigma * 1 then project
ones = np.ones(n)
w_mv = invSigma.dot(ones)
w_mv = np.maximum(w_mv, 0.0)  # ensure non-negative for long-only
w_mv = project_and_normalize(w_mv)

# 2) Approx tangency (max-sharpe): invSigma * (mu - rf) then project
excess = mu.values - RF_ANNUAL
w_tan = invSigma.dot(excess)
w_tan = np.maximum(w_tan, 0.0)
w_tan = project_and_normalize(w_tan)

# 3) Risk parity via iterative scaling (fast approximate)
def risk_parity_iterative(cov, initial=None, max_iter=2000, tol=1e-8):
    m = cov.shape[0]
    if initial is None:
        w = np.ones(m) / m
    else:
        w = initial.copy()
    for i in range(max_iter):
        sigma_p = math.sqrt(max(1e-16, w.dot(cov).dot(w)))
        mrc = cov.dot(w)
        rc = w * mrc
        target = sigma_p**2 / m
        # avoid dividing by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            adj = target / rc
            adj = np.nan_to_num(adj, nan=1.0, posinf=1.0, neginf=1.0)
        w_new = w * np.sqrt(adj)
        w_new = np.clip(w_new, MIN_WEIGHT, MAX_WEIGHT)
        if w_new.sum() == 0:
            w_new = np.ones_like(w) / len(w)
        w_new = w_new / w_new.sum()
        if np.linalg.norm(w_new - w) < tol:
            return w_new
        w = w_new
    return w

w_rp = risk_parity_iterative(Sigma_reg, initial=None, max_iter=500)

# Project and renormalize final rp to ensure bounds
w_rp = project_and_normalize(w_rp)

# 4) Blended: scalarized objective maximize (Sharpe - lam * Herfindahl)
# We'll use a few quick iterations of a gradient-free local refinement (optional)
def portfolio_return(w):
    return float(np.dot(w, mu.values))
def portfolio_vol(w):
    return float(np.sqrt(w.dot(Sigma_reg).dot(w)))
def sharpe_of(w):
    v = portfolio_vol(w)
    return (portfolio_return(w) - RF_ANNUAL) / v if v > 0 else -9e9

# Start blended from tangency
w_blend = w_tan.copy()
lam_c = 3.0

# small local improvement loop (projected gradient-like)
for _ in range(200):
    # try tiny random perturbations and keep improvements (simple simulated annealing)
    cand = w_blend + np.random.normal(scale=1e-3, size=n)
    cand = project_and_normalize(cand)
    val_curr = sharpe_of(w_blend) - lam_c * np.sum(w_blend**2)
    val_cand = sharpe_of(cand) - lam_c * np.sum(cand**2)
    if val_cand > val_curr:
        w_blend = cand

# ensure bounds
w_blend = project_and_normalize(w_blend)

# Summarize and save weights and top contributors
def risk_contribs(w):
    mrc = Sigma_reg.dot(w)
    rc = w * mrc
    total_var = w.dot(Sigma_reg).dot(w)
    return rc / total_var, rc

portfolios = {
    "min_variance": w_mv,
    "max_sharpe": w_tan,
    "risk_parity": w_rp,
    "blended": w_blend
}

summary_rows = []
for name, w in portfolios.items():
    r = float(np.dot(w, mu.values))
    v = float(np.sqrt(w.dot(Sigma_reg).dot(w)))
    s = (r - RF_ANNUAL) / v if v > 0 else np.nan
    contrib_pct, rc = risk_contribs(w)
    dfc = pd.DataFrame({
        'ticker': assets,
        'weight': w,
        'ann_return': asset_stats.loc[assets,'ann_return'].values,
        'ann_vol': asset_stats.loc[assets,'ann_vol'].values,
        'contrib_pct': contrib_pct
    }).sort_values('contrib_pct', ascending=False)
    # save
    pd.Series(w, index=assets).to_csv(os.path.join(OUTPUT_DIR, f"{name}_weights_full.csv"))
    dfc.head(50).to_csv(os.path.join(OUTPUT_DIR, f"{name}_top_contributors_full.csv"))
    summary_rows.append({'portfolio': name, 'ann_return': r, 'ann_vol': v, 'sharpe': s, 'n_nonzero': np.count_nonzero(w>1e-8)})
    # display top 10 directly
    print("\nTop 10 risk contributors for", name)
    print(dfc.head(10).to_string(index=False))

summary_table = pd.DataFrame(summary_rows).set_index('portfolio')
summary_table.to_csv(os.path.join(OUTPUT_DIR, "portfolio_summary_table_full.csv"))
print("\nSummary table:")
print(summary_table)

# Save covariance and mu already done; save picks
print("\nAll outputs saved to:", OUTPUT_DIR)
#print("If you'd like, run the next step to compute backtests, VaR, rolling metrics, heatmaps, or request specific portfolio exports.")
