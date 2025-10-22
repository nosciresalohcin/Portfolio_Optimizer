# chatGPT_backtest.py
import os, math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -------- CONFIG --------
PRICES_CSV = "sp500_clean_wide.csv"   # update if different
WEIGHTS_DIR = "optimization_full_universe_results"   # folder your optimizer saved into
OUTPUT_DIR = "backtest_results"
RF_ANNUAL = 0.04
TRADING_DAYS = 252
ROLL_WINDOW = 126   # ~6 months
# ------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load prices
df = pd.read_csv(PRICES_CSV)
df.columns = [c.strip() for c in df.columns]
date_col = df.columns[0]
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
df.set_index(date_col, inplace=True)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Daily returns
rets = df.pct_change(fill_method=None).dropna(how='all')

# Load weight files (expect files named like: min_variance_weights_full.csv, max_sharpe_weights_full.csv, etc.)
weight_files = [f for f in os.listdir(WEIGHTS_DIR) if "_weights" in f and f.endswith(".csv")]
if not weight_files:
    raise SystemExit(f"No weight files found in {WEIGHTS_DIR}. Check the folder name.")

port_returns = {}
port_weights = {}
for f in weight_files:
    path = os.path.join(WEIGHTS_DIR, f)
    # read robustly: either one-column series or two-column CSV with tickers+weights
    tmp = pd.read_csv(path)
    if tmp.shape[1] == 1:
        # sometimes pandas preserves index as first column; try reading with index
        tmp2 = pd.read_csv(path, index_col=0, squeeze=True)
        ser = pd.Series(tmp2).astype(float)
    else:
        ser = pd.Series(tmp.iloc[:,1].values, index=tmp.iloc[:,0].values).astype(float)
    # normalize and align
    name = f.split("_weights")[0]
    ser = ser[ser.index.isin(rets.columns)]
    if ser.sum() <= 0:
        continue
    ser = ser / ser.sum()
    port_weights[name] = ser
    port_returns[name] = rets[ser.index].dot(ser.values)

# Build DataFrame
port_rets = pd.DataFrame(port_returns)
port_rets.to_csv(os.path.join(OUTPUT_DIR, "port_daily_returns.csv"))

# Cumulative equity
cum = (1 + port_rets).cumprod()
cum.to_csv(os.path.join(OUTPUT_DIR, "port_cumulative.csv"))

# Backtest summary
rf_daily = (1+RF_ANNUAL)**(1/TRADING_DAYS) - 1
summary_rows = []
for col in port_rets.columns:
    s = port_rets[col].dropna()
    ann_r = s.mean() * TRADING_DAYS
    ann_vol = s.std() * math.sqrt(TRADING_DAYS)
    sharpe = (ann_r - RF_ANNUAL) / ann_vol if ann_vol>0 else np.nan
    cumseries = (1+s).cumprod()
    max_dd = (cumseries / cumseries.cummax() - 1).min()
    # Beta vs benchmark if present
    benchmark = None
    if "^GSPC" in rets.columns:
        benchmark = rets["^GSPC"]
    elif "SPY" in rets.columns:
        benchmark = rets["SPY"]
    beta = np.nan
    if benchmark is not None:
        common = s.index.intersection(benchmark.index)
        if len(common) > 30:
            Y = s.loc[common] - rf_daily
            X = benchmark.loc[common] - rf_daily
            model = sm.OLS(Y, sm.add_constant(X)).fit()
            beta = float(model.params.iloc[1])
    summary_rows.append({"portfolio": col, "ann_return": ann_r, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_dd, "beta": beta, "n_obs": s.shape[0]})

summary = pd.DataFrame(summary_rows).set_index("portfolio")
summary.to_csv(os.path.join(OUTPUT_DIR, "backtest_summary.csv"))
print(summary)

# Rolling metrics and plots (limit to first 6 portfolios for plots)
rolling_vol = port_rets.rolling(ROLL_WINDOW).std() * math.sqrt(TRADING_DAYS)
rolling_sharpe = ((port_rets.rolling(ROLL_WINDOW).mean() * TRADING_DAYS) - RF_ANNUAL) / rolling_vol

summary.to_csv(os.path.join(OUTPUT_DIR, "backtest_summary.csv"))
rolling_vol.to_csv(os.path.join(OUTPUT_DIR, "rolling_vol.csv"))
rolling_sharpe.to_csv(os.path.join(OUTPUT_DIR, "rolling_sharpe.csv"))

# Save plots per portfolio (up to 6)
display_list = list(port_rets.columns)[:6]
for name in display_list:
    # equity curve
    plt.figure(figsize=(10,4))
    plt.plot(cum.index, cum[name])
    plt.title(f"Cumulative return - {name}")
    plt.xlabel("Date"); plt.ylabel("Growth of $1"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cum_{name}.png"))
    plt.close()

    # rolling vol
    plt.figure(figsize=(10,4))
    plt.plot(rolling_vol.index, rolling_vol[name])
    plt.title(f"Rolling {ROLL_WINDOW}-day Annualized Vol - {name}")
    plt.xlabel("Date"); plt.ylabel("Volatility"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"rolling_vol_{name}.png"))
    plt.close()

    # rolling sharpe
    plt.figure(figsize=(10,4))
    plt.plot(rolling_sharpe.index, rolling_sharpe[name])
    plt.title(f"Rolling {ROLL_WINDOW}-day Sharpe - {name}")
    plt.xlabel("Date"); plt.ylabel("Sharpe"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"rolling_sharpe_{name}.png"))
    plt.close()

print("Backtest complete. Results saved to", OUTPUT_DIR)
