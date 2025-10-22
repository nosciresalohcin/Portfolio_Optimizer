# portfolio_pipeline_reports.py
"""
Generates:
 - portfolio composition CSVs for min_variance, max_sharpe, risk_parity, blended
 - rolling quarterly re-optimization (max_sharpe) with transaction costs (0.05% per trade)
 - PCA factor diagnostics (first 5 PCs) and portfolio regressions onto PCs
 - out-of-sample test (train through 2022, test 2023 onward)
Outputs go to ./portfolio_reports (csv + small plots)
Fast / Full mode available (FAST_MODE True trims universe to speed up)
"""

import os, math, numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.linalg import pinv
from datetime import timedelta

# ---- USER CONFIG ----
DATA_CSV = "sp500_clean_wide.csv"
OUT_DIR = "portfolio_reports"
os.makedirs(OUT_DIR, exist_ok=True)

TRADING_DAYS = 252
MIN_OBS = 200
MAX_WEIGHT = 0.05
RF_ANNUAL = 0.04
TC_PER_TRADE = 0.0005   # 0.05%
LOOKBACK_YEARS = 3
FAST_MODE = True        # <--- set False to run the full universe (slower)
FAST_TOP_N = 300        # only used if FAST_MODE True
# ----------------------

# load prices
df = pd.read_csv(DATA_CSV)
df.columns = [c.strip() for c in df.columns]
date_col = df.columns[0]
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
df.set_index(date_col, inplace=True)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# returns
rets = df.pct_change().dropna(how='all')

# choose assets with MIN_OBS
asset_cols = [c for c in df.columns if df[c].dropna().shape[0] >= MIN_OBS]
if FAST_MODE:
    # rank by #observations (or use volatility/market cap if available)
    obs_counts = pd.Series({c: df[c].dropna().shape[0] for c in asset_cols})
    top = obs_counts.sort_values(ascending=False).head(FAST_TOP_N).index.tolist()
    asset_cols = [c for c in asset_cols if c in top]

prices = df[asset_cols].copy()
rets = rets[asset_cols].copy()
assets = list(rets.columns)
n = len(assets)

# compute ann stats
ann_returns = (1 + rets).prod() ** (TRADING_DAYS / len(rets)) - 1
ann_vols = rets.std() * math.sqrt(TRADING_DAYS)
asset_stats = pd.DataFrame({'ann_return': ann_returns, 'ann_vol': ann_vols, 'obs':[rets[c].dropna().shape[0] for c in assets]})
asset_stats.to_csv(os.path.join(OUT_DIR, "asset_stats.csv"))

# covariance with Ledoit-Wolf
lw = LedoitWolf().fit(rets.fillna(method='ffill').fillna(method='bfill').values)
Sigma = lw.covariance_
cov_df = pd.DataFrame(Sigma, index=assets, columns=assets)
cov_df.to_csv(os.path.join(OUT_DIR, "covariance.csv"))

# shrink mu
hist_mu = asset_stats['ann_return'].values
cross_mean = np.nanmean(hist_mu)
lam = 0.6
mu = pd.Series(lam*hist_mu + (1-lam)*cross_mean, index=assets)
mu.to_csv(os.path.join(OUT_DIR, "shrunk_mu.csv"))

# helper projection
def project_and_normalize(w, min_w=0.0, max_w=MAX_WEIGHT):
    w = np.clip(w, min_w, max_w)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w)/len(w)
    else:
        w = w / s
    return w

# compute basic portfolios
Sigma_reg = Sigma + np.eye(n)*1e-8
invSigma = pinv(Sigma_reg)
ones = np.ones(n)

# min variance
w_mv = invSigma.dot(ones)
w_mv = np.maximum(w_mv, 0)
w_mv = project_and_normalize(w_mv)

# tangency / max sharpe
excess = mu.values - RF_ANNUAL
w_tan = invSigma.dot(excess)
w_tan = np.maximum(w_tan, 0)
w_tan = project_and_normalize(w_tan)

# risk parity (iterative)
def risk_parity_iterative(cov, initial=None, max_iter=2000, tol=1e-8):
    m = cov.shape[0]
    if initial is None:
        w = np.ones(m)/m
    else:
        w = initial.copy()
    for i in range(max_iter):
        sigma_p = math.sqrt(max(1e-16, w.dot(cov).dot(w)))
        mrc = cov.dot(w)
        rc = w * mrc
        target = sigma_p**2 / m
        with np.errstate(divide='ignore', invalid='ignore'):
            adj = target / rc
            adj = np.nan_to_num(adj, nan=1.0, posinf=1.0, neginf=1.0)
        w_new = w * np.sqrt(adj)
        w_new = np.clip(w_new, 0.0, MAX_WEIGHT)
        if w_new.sum() == 0:
            w_new = np.ones_like(w)/len(w)
        w_new = w_new / w_new.sum()
        if np.linalg.norm(w_new - w) < tol:
            return w_new
        w = w_new
    return w

w_rp = risk_parity_iterative(Sigma_reg)

# blended (local random improvement)
w_blend = w_tan.copy()
for _ in range(300):
    cand = w_blend + np.random.normal(scale=1e-3, size=n)
    cand = project_and_normalize(cand)
    def port_ret(w): return float(np.dot(w, mu.values))
    def port_vol(w): return float(math.sqrt(max(1e-16, w.dot(Sigma_reg).dot(w))))
    def sharpe(w):
        v = port_vol(w)
        return (port_ret(w) - RF_ANNUAL)/v if v>0 else -9e9
    val_curr = sharpe(w_blend) - 3.0 * np.sum(w_blend**2)
    val_cand = sharpe(cand) - 3.0 * np.sum(cand**2)
    if val_cand > val_curr:
        w_blend = cand

portfolios = {
    "min_variance": pd.Series(w_mv, index=assets),
    "max_sharpe": pd.Series(w_tan, index=assets),
    "risk_parity": pd.Series(w_rp, index=assets),
    "blended": pd.Series(w_blend, index=assets)
}

# save weights
for name, ser in portfolios.items():
    ser.to_csv(os.path.join(OUT_DIR, f"{name}_weights.csv"))

# Function to build composition table (latest price, returns, CAGR, vol, contribution)
def build_composition(weights, asof_date):
    wpos = weights[weights>0]
    tickers = list(wpos.index)
    last = prices[tickers].loc[:asof_date].ffill().iloc[-1]
    def pct_back(days):
        idx_asof = prices.index.get_indexer([asof_date], method='ffill')[0]
        start_idx = max(0, idx_asof - days)
        start_price = prices[tickers].iloc[start_idx].ffill()
        return (last / start_price - 1)
    r1 = pct_back(252)
    r3 = pct_back(3*252)
    r5 = pct_back(5*252)
    total = (last / prices[tickers].iloc[0].ffill()) - 1
    years = (asof_date - prices.index[0]).days / 365.25
    cagr = (last / prices[tickers].iloc[0].ffill()) ** (1/years) - 1
    vol1 = rets[tickers].loc[asof_date - pd.Timedelta(days=252*1.5):asof_date].std() * math.sqrt(TRADING_DAYS)
    approx_contrib = wpos.values * vol1.fillna(0).values
    df = pd.DataFrame({
        "ticker": tickers,
        "weight": wpos.values,
        "latest_price": last.values,
        "1yr_return": r1.values,
        "3yr_return": r3.values,
        "5yr_return": r5.values,
        "total_return": total.values,
        "CAGR": cagr.values,
        "ann_vol_1yr": vol1.values,
        "approx_contrib": approx_contrib
    }).sort_values("weight", ascending=False)
    return df

asof = prices.index.max()
for name, w in portfolios.items():
    comp = build_composition(w, asof)
    comp.to_csv(os.path.join(OUT_DIR, f"{name}_composition_asof_{asof.date()}.csv"), index=False)
    comp.head(50).to_csv(os.path.join(OUT_DIR, f"{name}_top50.csv"), index=False)

# Rolling quarterly re-optimization for max_sharpe with transaction costs
lookback_days = LOOKBACK_YEARS * TRADING_DAYS
start_idx = prices.index.get_indexer([prices.index[0]])[0] + lookback_days
rebalance_dates = prices.index[start_idx::63]  # ~quarterly
rebalance_dates = rebalance_dates[rebalance_dates <= prices.index.max()]

rolling_w = {}
prev = None
for d in rebalance_dates:
    idx = prices.index.get_indexer([d])[0]
    est_start_idx = max(0, idx - lookback_days)
    est_prices = prices.iloc[est_start_idx:idx]
    est_rets = est_prices.pct_change().dropna(how='all')
    if est_rets.shape[0] < 60:
        continue
    try:
        lw_local = LedoitWolf().fit(est_rets.fillna(method='ffill').fillna(method='bfill').values)
        Sigma_local = lw_local.covariance_
    except Exception:
        Sigma_local = est_rets.cov() * TRADING_DAYS
    mu_local = (1+est_rets.mean())**TRADING_DAYS - 1
    mu_local = 0.6 * mu_local + 0.4 * mu_local.mean()
    excess_local = mu_local.values - RF_ANNUAL
    invS = pinv(Sigma_local + np.eye(Sigma_local.shape[0])*1e-8)
    w_raw = invS.dot(excess_local)
    w_raw = np.maximum(w_raw, 0)
    w_final = project_and_normalize(w_raw)
    w_series = pd.Series(w_final, index=est_rets.columns).reindex(rets.columns).fillna(0.0)
    if prev is None:
        tc = w_series.abs().sum() * TC_PER_TRADE
    else:
        tc = (w_series - prev).abs().sum() * TC_PER_TRADE
    rolling_w[d] = {"weights": w_series, "tc": tc}
    prev = w_series

# construct daily weights and compute portfolio returns with TC on rebalance days
if len(rolling_w) > 0:
    # create daily weight matrix
    daily_weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    sorted_dates = sorted(rolling_w.keys())
    for i, d in enumerate(sorted_dates):
        start = d
        end = sorted_dates[i+1] if i+1 < len(sorted_dates) else rets.index.max()+pd.Timedelta(days=1)
        mask = (rets.index >= start) & (rets.index < end)
        daily_weights.loc[mask, :] = rolling_w[d]["weights"].values
    # fill before first rebalance
    daily_weights.loc[rets.index < sorted_dates[0], :] = rolling_w[sorted_dates[0]]["weights"].values

    pf_daily = (rets * daily_weights.shift(1)).sum(axis=1)
    tc_series = pd.Series(0.0, index=rets.index)
    for d in sorted_dates:
        if d in rets.index:
            tc_series.loc[d] = -rolling_w[d]["tc"]
    pf_daily_adj = pf_daily + tc_series
    (pd.DataFrame({"pf_rets": pf_daily_adj})).to_csv(os.path.join(OUT_DIR, "rolling_quarterly_max_sharpe_rets.csv"))
    ((1+pf_daily_adj).cumprod()).to_csv(os.path.join(OUT_DIR, "rolling_quarterly_max_sharpe_cum.csv"))
    # turnover
    turnover_list = []
    prevw = None
    for d in sorted_dates:
        w = rolling_w[d]["weights"]
        turnover = w.abs().sum() if prevw is None else (w-prevw).abs().sum()
        turnover_list.append({"date": d, "turnover": turnover, "tc": rolling_w[d]["tc"]})
        prevw = w
    pd.DataFrame(turnover_list).to_csv(os.path.join(OUT_DIR, "quarterly_rebalance_turnover.csv"), index=False)
else:
    print("No rolling rebalance dates were produced (insufficient data).")

# PCA factor diagnostics
pca = PCA(n_components=5)
rets_fill = rets.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
pcs = pca.fit_transform(rets_fill.values)
pc_df = pd.DataFrame(pcs, index=rets_fill.index, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
pca_explained = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(pcs.shape[1])], "explained_variance_ratio": pca.explained_variance_ratio_})
pca_explained.to_csv(os.path.join(OUT_DIR, "pca_explained_variance_ratio.csv"), index=False)

factor_regs = []
for name, w in portfolios.items():
    w_full = w.reindex(rets.columns).fillna(0.0)
    port_daily = (rets * w_full).sum(axis=1)
    common = port_daily.index.intersection(pc_df.index)
    X = pc_df.loc[common].values
    Y = port_daily.loc[common].values.reshape(-1,1)
    lr = LinearRegression().fit(X, Y)
    factor_regs.append({
        "portfolio": name,
        "r2": float(lr.score(X,Y)),
        "coefs": lr.coef_.flatten().tolist()
    })
pd.DataFrame(factor_regs).to_csv(os.path.join(OUT_DIR, "pca_factor_regression_results.csv"), index=False)

# Out-of-sample test: train up to 2022-12-31, test 2023 onwards (train weight computed using LOOKBACK_YEARS)
train_end = pd.Timestamp("2022-12-31")
test_start = pd.Timestamp("2023-01-01")
train_end = prices.index[prices.index.get_indexer([train_end], method='ffill')[0]]
test_start = prices.index[prices.index.get_indexer([test_start], method='bfill')[0]]

def optimize_max_sharpe_for_date(end_date):
    idx = prices.index.get_indexer([end_date])[0]
    est_start_idx = max(0, idx - lookback_days)
    est_prices = prices.iloc[est_start_idx:idx+1]
    est_rets = est_prices.pct_change().dropna(how='all')
    if est_rets.shape[0] < 60:
        return None
    try:
        lw3 = LedoitWolf().fit(est_rets.fillna(method='ffill').fillna(method='bfill').values)
        Sigma3 = lw3.covariance_
    except Exception:
        Sigma3 = est_rets.cov() * TRADING_DAYS
    hist_mu3 = (1+est_rets.mean())**TRADING_DAYS - 1
    mu3 = 0.6*hist_mu3 + 0.4*hist_mu3.mean()
    excess3 = mu3.values - RF_ANNUAL
    invS3 = pinv(Sigma3 + np.eye(Sigma3.shape[0])*1e-8)
    w_raw3 = np.maximum(invS3.dot(excess3), 0)
    w3 = project_and_normalize(w_raw3)
    w3 = pd.Series(w3, index=est_rets.columns).reindex(rets.columns).fillna(0.0)
    return w3

train_w = optimize_max_sharpe_for_date(train_end)
if train_w is not None:
    train_w.to_csv(os.path.join(OUT_DIR, "oos_train_max_sharpe_weights_2015_2022.csv"))
    test_rets = rets.loc[test_start:]
    test_pf = test_rets[train_w.index].dot(train_w.values)
    test_perf = {
        "ann_return": float(test_pf.mean()*TRADING_DAYS),
        "ann_vol": float(test_pf.std()*math.sqrt(TRADING_DAYS)),
        "sharpe": float((test_pf.mean()*TRADING_DAYS - RF_ANNUAL)/(test_pf.std()*math.sqrt(TRADING_DAYS)))
    }
    pd.DataFrame([test_perf]).to_csv(os.path.join(OUT_DIR, "oos_test_performance_2023_onward.csv"), index=False)

print("All done. Output folder:", OUT_DIR)
