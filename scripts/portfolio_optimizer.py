from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -------------------
# Ticker validation utilities
# -------------------
def validate_tickers_quick(tickers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Quick validation: attempts to download a very short window for each ticker.
    Returns (valid_symbols, invalid_symbols).
    Invalid = symbols that return completely empty for a basic query.
    """
    valid_symbols: List[str] = []
    invalid_symbols: List[str] = []

    for t in tickers:
        try:
            df = yf.download(t, period="5d", auto_adjust=True, progress=False, interval="1d")
            if isinstance(df, pd.DataFrame) and not df.empty:
                valid_symbols.append(t)
            else:
                invalid_symbols.append(t)
        except Exception:
            invalid_symbols.append(t)

    return valid_symbols, invalid_symbols


# -------------------
# Data classes
# -------------------

@dataclass
class PortfolioInputs:
    tickers: List[str]
    start: str
    end: str
    risk_free: float
    min_weight: float = 0.0
    max_weight: float = 1.0
    trading_days: int = 252
    return_method: str = "average"  # "average" or "cagr"

@dataclass
class PortfolioData:
    prices: pd.DataFrame
    returns_daily: pd.DataFrame
    cov_daily: np.ndarray
    mean_daily: np.ndarray

@dataclass
class IntervalMetrics:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    mean_annual: float
    vol_annual: float
    sharpe: float


# -------------------
# Data loading
# -------------------

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns are strings, stripped, and unique in order.
    If duplicates exist, keep the first occurrence and drop subsequent duplicates.
    """
    cols = pd.Index([str(c).strip() for c in df.columns])
    df = df.copy()
    df.columns = cols
    # Drop duplicate columns, keeping the first occurrence to preserve covariance structure
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetch a batch of tickers from Yahoo Finance."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        interval="1d",
    )

    if isinstance(data, pd.DataFrame):
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.levels[0]:
                prices = data["Adj Close"]
            else:
                prices = data.xs(data.columns.levels[0][0], axis=1, level=0)
        else:
            prices = data
    else:
        prices = pd.DataFrame(data)

    # Keep only columns that have some data
    prices = prices.dropna(axis=1, how="all")

    # Sort index; forward-fill per column; back-fill only initial NaNs per column
    prices = prices.sort_index()
    prices = prices.ffill().bfill()

    # Do NOT drop rows globally (avoid prices.dropna() without axis)
    # Only warn if a column starts later than requested
    for t in prices.columns:
        first_date = prices[t].first_valid_index()
        if first_date and pd.Timestamp(first_date) > pd.Timestamp(start):
            print(f"Warning: {t} only has data from {pd.Timestamp(first_date).date()} "
                f"(later than requested {start})")

    return prices



def fetch_prices_clean(tickers, start, end, batch_size=50, max_loops=5):
    """
    Fetch prices for a large list of tickers in batches.
    Returns (prices DataFrame, failed_tickers list).
    """
    remaining = list(tickers)
    all_prices = pd.DataFrame()
    failed_total = []

    for loop in range(max_loops):
        if not remaining:
            break

        frames = []
        failed = []

        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i+batch_size]
            try:
                df = fetch_prices(batch, start, end)
                if not df.empty:
                    frames.append(df)
                    valid = df.columns.tolist()
                    failed.extend([t for t in batch if t not in valid])
                else:
                    # Batch failed, try individually
                    for t in batch:
                        try:
                            df_single = fetch_prices([t], start, end)
                            if not df_single.empty:
                                frames.append(df_single)
                            else:
                                failed.append(t)
                        except Exception:
                            failed.append(t)
            except Exception:
                # Batch crashed, try individually
                for t in batch:
                    try:
                        df_single = fetch_prices([t], start, end)
                        if not df_single.empty:
                            frames.append(df_single)
                        else:
                            failed.append(t)
                    except Exception:
                        failed.append(t)

        if frames:
            all_prices = pd.concat(frames, axis=1)

        if not failed:
            print(f"Loop {loop+1}: all tickers succeeded.")
            break

        print(f"Loop {loop+1}: {len(failed)} tickers failed and will be dropped: {failed[:10]}...")
        failed_total.extend(failed)
        remaining = [t for t in remaining if t not in failed]

    return all_prices, failed_total


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # Sort index and compute percent change per column
    rets = prices.sort_index().pct_change()

    # Drop only the first row (pct_change introduces a NaN per column at the start)
    # Do NOT drop rows globally due to any NaN elsewhere
    if len(rets) > 0:
        rets = rets.iloc[1:]
    return rets



def build_portfolio_data(
    tickers: List[str],
    start: str,
    end: str,
    min_obs: int = 252
) -> Tuple[PortfolioData, List[str], List[str], List[str], List[str]]:
    """
    Build PortfolioData from tickers, dropping those with insufficient history.
    Returns (PortfolioData, valid_tickers, invalid_symbols, failed_downloads, dropped_for_history).

    - valid_tickers: tickers that survived all checks
    - invalid_symbols: symbols that appear invalid (pre-validation failed)
    - failed_downloads: symbols that looked valid but full-range download failed
    - dropped_for_history: downloaded symbols with < min_obs valid returns
    """

    requested = list(tickers)

    # Pre-validate symbols quickly to filter out obvious invalids early
    valid_symbols, invalid_symbols = validate_tickers_quick(requested)
    if not valid_symbols:
        return (
            PortfolioData(
                prices=pd.DataFrame(),
                returns_daily=pd.DataFrame(),
                cov_daily=np.array([]),
                mean_daily=np.array([])
            ),
            [],
            invalid_symbols,  # invalid
            [],               # failed_downloads
            []                # dropped_for_history
        )

    # Fetch prices (batching + cleaning) for valid symbols only
    prices, failed_downloads = fetch_prices_clean(valid_symbols, start, end, batch_size=50)

    downloaded_tickers = prices.columns.tolist()

    if prices.empty or len(downloaded_tickers) == 0:
        return (
            PortfolioData(
                prices=pd.DataFrame(),
                returns_daily=pd.DataFrame(),
                cov_daily=np.array([]),
                mean_daily=np.array([])
            ),
            [],
            invalid_symbols,
            list(set(valid_symbols)),  # all failed to download
            []                         # dropped_for_history
        )

    # Compute daily returns without global drops
    prices = prices.sort_index()
    returns_daily = prices.pct_change()
    if len(returns_daily) > 0:
        returns_daily = returns_daily.iloc[1:]  # remove the first pct_change NaN

    # Per‑ticker min_obs filtering
    dropped_for_history: List[str] = []
    if min_obs > 0:
        valid_obs = returns_daily.notna().sum()  # per-column non-NaN counts
        sufficient_history = valid_obs >= min_obs
        dropped_for_history = valid_obs.index[~sufficient_history].tolist()

        # Keep only survivors
        returns_daily = returns_daily.loc[:, sufficient_history]
        prices = prices.loc[:, returns_daily.columns]

    # Deduplicate and drop truly empty columns
    returns_daily = returns_daily.loc[:, ~returns_daily.columns.duplicated()]
    returns_daily = returns_daily.dropna(axis=1, how="all")
    prices = prices.loc[:, returns_daily.columns]

    valid_tickers = returns_daily.columns.tolist()

    if returns_daily.empty or len(valid_tickers) == 0:
        return (
            PortfolioData(
                prices=pd.DataFrame(),
                returns_daily=pd.DataFrame(),
                cov_daily=np.array([]),
                mean_daily=np.array([])
            ),
            [],
            invalid_symbols,
            failed_downloads,
            dropped_for_history
        )

    # Covariance and mean returns
    cov_daily = returns_daily.cov().values
    mean_daily = returns_daily.mean().values

    # Optional reconciliation log
    print(f"Requested: {len(requested)} | Pre‑valid: {len(valid_symbols)} | "
          f"Invalid symbols: {len(invalid_symbols)} | Downloaded: {len(downloaded_tickers)} | "
          f"Failed downloads: {len(failed_downloads)} | Survived min_obs: {len(valid_tickers)} | "
          f"Dropped history: {len(dropped_for_history)}")

    return (
        PortfolioData(
            prices=prices,
            returns_daily=returns_daily,
            cov_daily=cov_daily,
            mean_daily=mean_daily,
        ),
        valid_tickers,
        invalid_symbols,
        failed_downloads,
        dropped_for_history
    )




# -------------------
# Portfolio math
# -------------------

def annualize_vol(weights: np.ndarray, cov_daily: np.ndarray, trading_days: int = 252) -> float:
    var_daily = float(weights @ cov_daily @ weights)
    return float(np.sqrt(trading_days * var_daily))


def annualized_return_from_total_period(
    returns_daily: pd.DataFrame,
    weights: np.ndarray,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    trading_days: int = 252,
    method: str = "average",
) -> float:
    if start is not None or end is not None:
        mask = pd.Series(True, index=returns_daily.index)
        if start is not None:
            mask &= returns_daily.index >= pd.Timestamp(start)
        if end is not None:
            mask &= returns_daily.index <= pd.Timestamp(end)
        rets = returns_daily.loc[mask]
    else:
        rets = returns_daily

    # Defensive check against dimension mismatch
    if rets.shape[1] != len(weights):
        raise ValueError(
            f"Dimension mismatch: returns has {rets.shape[1]} columns, "
            f"weights has length {len(weights)}"
        )

    if rets.empty:
        return 0.0

    port_daily = rets.values @ weights
    cum_growth = float(np.prod(1.0 + port_daily))
    total_return = cum_growth - 1.0

    days = (rets.index[-1] - rets.index[0]).days
    years = max(days / 365.25, 1e-9)

    if method.lower() == "cagr":
        if cum_growth <= 0:
            return -np.inf
        return float(cum_growth ** (1.0 / years) - 1.0)
    else:
        return float(total_return / years)


def sharpe_ratio(
    weights: np.ndarray,
    cov_daily: np.ndarray,
    returns_daily: pd.DataFrame,
    risk_free: float,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    trading_days: int = 252,
    method: str = "average",
) -> float:
    avg_ret_annual = annualized_return_from_total_period(
        returns_daily, weights, start, end, trading_days, method=method
    )
    vol_annual = annualize_vol(weights, cov_daily, trading_days)
    if vol_annual <= 0 or not np.isfinite(vol_annual):
        return -np.inf
    return float((avg_ret_annual - risk_free) / vol_annual)

def compute_shrinkage_covariance(returns_daily: pd.DataFrame, shrinkage: float = 0.1) -> np.ndarray:
    """
    Compute a shrinkage covariance: (1 - shrinkage) * sample_cov + shrinkage * diag(diag(sample_cov)).
    Keeps scale, reduces condition number, improves optimizer stability.
    """
    sample_cov = returns_daily.cov().values
    if sample_cov.size == 0:
        return sample_cov
    diag_cov = np.diag(np.diag(sample_cov))
    cov_shrunk = (1.0 - shrinkage) * sample_cov + shrinkage * diag_cov
    return cov_shrunk


# -------------------
# Correlation metrics
# -------------------

def portfolio_avg_correlation(
    weights: np.ndarray,
    returns_daily: pd.DataFrame,
    tickers: Optional[List[str]] = None,
) -> float:
    if tickers is not None:
        returns_daily = returns_daily[tickers]

    corr = returns_daily.corr().values
    w = np.array(weights)
    if corr.shape[0] != w.shape[0]:
        raise ValueError(
            f"Dimension mismatch: weights={w.shape[0]} vs corr={corr.shape[0]} "
            f"(did you pass the matching tickers subset?)"
        )

    mask = np.ones_like(corr, dtype=bool)
    np.fill_diagonal(mask, 0)
    num = (w[:, None] * w[None, :] * corr)[mask].sum()
    den = (w[:, None] * w[None, :])[mask].sum()
    return float(num / den) if den > 0 else 0.0


# -------------------
# Optimization
# -------------------

def optimize_max_sharpe(
    data: PortfolioData,
    inputs: PortfolioInputs,
    interval_start: Optional[str] = None,
    interval_end: Optional[str] = None,
    diversification_alpha: float = 0.0,
    cov_shrinkage: float = 0.1,
    weight_eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict]:
    """
    Optimize portfolio weights to maximize Sharpe ratio.
    Subset-enforcing logic: optimize with min_weight=0, then
    filter final allocations so that only assets >= inputs.min_weight remain.
    Improved fallback: if no assets survive, allocate respecting max_weight
    across the best assets instead of a single 100% allocation.
    """

    if data.returns_daily.empty or data.returns_daily.shape[1] == 0:
        return np.array([]), {
            "success": False,
            "message": "No valid data to optimize on",
            "iterations": 0,
            "avg_return_annual": np.nan,
            "vol_annual": np.nan,
            "sharpe": np.nan,
            "diversification_alpha": diversification_alpha,
            "avg_correlation": np.nan,
            "tickers": [],
        }

    # Clean returns: deduplicate and drop empty/zero-variance columns
    returns_clean = data.returns_daily.loc[:, ~data.returns_daily.columns.duplicated()]
    returns_clean = returns_clean.dropna(axis=1, how="all")
    var_series = returns_clean.var()
    if isinstance(var_series, pd.Series):
        returns_clean = returns_clean.loc[:, var_series > 0]

    tickers_all = np.array(returns_clean.columns)
    n = len(tickers_all)

    if n == 0:
        return np.array([]), {
            "success": False,
            "message": "No usable assets after cleaning",
            "iterations": 0,
            "avg_return_annual": np.nan,
            "vol_annual": np.nan,
            "sharpe": np.nan,
            "diversification_alpha": diversification_alpha,
            "avg_correlation": np.nan,
            "tickers": [],
        }

    # Stabilized covariance
    cov_use = compute_shrinkage_covariance(returns_clean, shrinkage=cov_shrinkage)

    # Bounds and constraints (min_weight=0 internally)
    bounds = [(0.0, inputs.max_weight) for _ in range(n)]
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    # Initial guess
    w0 = np.ones(n) / n

    start_ts = pd.Timestamp(interval_start) if interval_start else None
    end_ts = pd.Timestamp(interval_end) if interval_end else None

    corr_full = returns_clean.corr().values if diversification_alpha > 0 else None
    corr_offdiag = corr_full - np.eye(n) if corr_full is not None else None

    def objective(w: np.ndarray) -> float:
        s = sharpe_ratio(
            w, cov_use, returns_clean,
            inputs.risk_free, start_ts, end_ts,
            inputs.trading_days, method=inputs.return_method
        )
        obj = -s
        if diversification_alpha > 0 and corr_offdiag is not None:
            obj += diversification_alpha * float(w @ corr_offdiag @ w)
        return obj

    result = minimize(
        objective, w0, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-9}
    )

    w_raw = result.x
    w_raw = np.clip(w_raw, 0.0, inputs.max_weight)
    w_raw = w_raw / max(w_raw.sum(), 1e-12)

    # Subset-enforcing logic
    mask = w_raw >= inputs.min_weight
    idx = np.where(mask)[0]
    w_opt = w_raw[mask]
    tickers_opt = tickers_all[idx].tolist()

    if len(w_opt) == 0:
        # Improved fallback: respect max_weight and fill remainder
        sorted_idx = np.argsort(returns_clean.mean().values)[::-1]
        w_opt = []
        tickers_opt = []
        remaining = 1.0
        for j in sorted_idx:
            add_weight = min(inputs.max_weight, remaining)
            if add_weight <= 0:
                break
            w_opt.append(add_weight)
            tickers_opt.append(tickers_all[j])
            remaining -= add_weight
            if remaining <= 1e-8:
                break
        w_opt = np.array(w_opt)

        # Subset returns for chosen tickers
        rd_sub = returns_clean.loc[:, tickers_opt]
    else:
        # Renormalize surviving weights
        w_opt = w_opt / w_opt.sum()
        rd_sub = returns_clean.iloc[:, idx]

    cov_sub = compute_shrinkage_covariance(rd_sub, shrinkage=cov_shrinkage)

    avg_ret_annual = annualized_return_from_total_period(
        rd_sub, w_opt, start_ts, end_ts,
        inputs.trading_days, method=inputs.return_method
    )

    vol_annual = annualize_vol(w_opt, cov_sub, inputs.trading_days)
    sharpe = (avg_ret_annual - inputs.risk_free) / vol_annual if vol_annual > 0 else -np.inf

    try:
        avg_corr = portfolio_avg_correlation(w_opt, rd_sub, tickers_opt)
    except Exception:
        avg_corr = np.nan

    info = {
        "success": result.success,
        "message": result.message,
        "iterations": result.nit,
        "avg_return_annual": avg_ret_annual,
        "vol_annual": vol_annual,
        "sharpe": sharpe,
        "diversification_alpha": diversification_alpha,
        "avg_correlation": avg_corr,
        "tickers": tickers_opt,
    }

    return w_opt, info


# -------------------
# Efficient frontier
# -------------------

def efficient_frontier(
    data: PortfolioData,
    inputs: PortfolioInputs,
    n_points: int = 50,
    cov_shrinkage: float = 0.1,
    weight_eps: float = 1e-8,
) -> Dict[str, List]:
    """
    Compute efficient frontier portfolios with shrinkage covariance for stability.

    Subset-enforcing logic: optimize with min_weight=0, then filter
    final allocations so that only assets >= inputs.min_weight remain.
    Improved fallback: if no assets survive, allocate respecting max_weight
    across the best assets instead of skipping the point.
    """

    # Clean returns
    returns_clean = data.returns_daily.loc[:, ~data.returns_daily.columns.duplicated()]
    returns_clean = returns_clean.dropna(axis=1, how="all")
    var_series = returns_clean.var()
    if isinstance(var_series, pd.Series):
        returns_clean = returns_clean.loc[:, var_series > 0]

    tickers_all = np.array(returns_clean.columns)
    n = len(tickers_all)
    results = {"return": [], "vol": [], "weights": [], "sharpe": [], "avg_corr": [], "fallback": []}

    if n == 0:
        return results

    cov_use = compute_shrinkage_covariance(returns_clean, shrinkage=cov_shrinkage)

    # Standalone asset returns
    asset_returns = []
    for i in range(n):
        w = np.zeros(n)
        w[i] = 1.0
        r = annualized_return_from_total_period(
            returns_clean, w, method=inputs.return_method
        )
        asset_returns.append(r)

    r_min, r_max = min(asset_returns), max(asset_returns)
    target_returns = np.linspace(r_min, r_max, n_points)

    for target in target_returns:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w, t=target:
                annualized_return_from_total_period(
                    returns_clean, w, method=inputs.return_method
                ) - t},
        )
        bounds = [(0.0, inputs.max_weight)] * n
        w0 = np.ones(n) / n

        try:
            res = minimize(
                lambda w: annualize_vol(w, cov_use, inputs.trading_days),
                w0, method="SLSQP", bounds=bounds, constraints=cons,
                options={"maxiter": 1000, "ftol": 1e-9}
            )
        except Exception:
            continue

        if not res.success:
            continue

        w_raw = res.x
        w_raw = np.clip(w_raw, 0.0, inputs.max_weight)
        w_raw = w_raw / max(w_raw.sum(), 1e-12)

        # Subset-enforcing logic
        mask = w_raw >= inputs.min_weight
        idx = np.where(mask)[0]
        w_opt = w_raw[mask]
        tickers_opt = tickers_all[idx].tolist()
        fallback_used = False

        if len(w_opt) == 0:
            # Improved fallback
            sorted_idx = np.argsort(returns_clean.mean().values)[::-1]
            w_opt = []
            tickers_opt = []
            remaining = 1.0
            for j in sorted_idx:
                add_weight = min(inputs.max_weight, remaining)
                if add_weight <= 0:
                    break
                w_opt.append(add_weight)
                tickers_opt.append(tickers_all[j])
                remaining -= add_weight
                if remaining <= 1e-8:
                    break
            w_opt = np.array(w_opt)
            rd_sub = returns_clean.loc[:, tickers_opt]
            fallback_used = True
        else:
            w_opt = w_opt / w_opt.sum()
            rd_sub = returns_clean.iloc[:, idx]

        cov_sub = compute_shrinkage_covariance(rd_sub, shrinkage=cov_shrinkage)
        r = annualized_return_from_total_period(rd_sub, w_opt, method=inputs.return_method)
        v = annualize_vol(w_opt, cov_sub, inputs.trading_days)
        s = (r - inputs.risk_free) / v if v > 0 and np.isfinite(v) else -np.inf

        try:
            c = portfolio_avg_correlation(w_opt, rd_sub, tickers_opt)
        except Exception:
            c = np.nan

        results["return"].append(r)
        results["vol"].append(v)
        results["weights"].append((tickers_opt, w_opt))
        results["sharpe"].append(s)
        results["avg_corr"].append(c)
        results["fallback"].append(fallback_used)

    return results


def plot_frontier(
    results: Dict[str, List],
    color_by: str = "sharpe",
    scale_size_by_assets: bool = True
):
    """
    Plot the efficient frontier.
    
    Parameters
    ----------
    results : dict
        Output from efficient_frontier().
    color_by : str, optional
        Metric to color points by: "sharpe" (default) or "corr".
    scale_size_by_assets : bool, optional
        If True, marker size is scaled by number of assets in each portfolio.
    """

    if not results or len(results["vol"]) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Efficient Frontier (no feasible portfolios)")
        ax.axis("off")
        return fig

    # Choose coloring metric
    if color_by == "corr":
        color_metric = results["avg_corr"]
        cbar_label = "Average weighted correlation"
        cmap = "coolwarm"
    else:
        color_metric = results["sharpe"]
        cbar_label = "Sharpe ratio"
        cmap = "viridis"

    # Marker sizes
    if scale_size_by_assets and "weights" in results:
        num_assets = [len(w[0]) for w in results["weights"]]
        sizes = [50 + 20 * n for n in num_assets]
    else:
        sizes = 60

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        results["vol"], results["return"],
        c=color_metric, cmap=cmap,
        s=sizes, edgecolor="k", alpha=0.85
    )

    ax.set_xlabel("Volatility (annual)")
    ax.set_ylabel("Return (annual)")
    ax.set_title(f"Efficient Frontier (colored by {cbar_label})")
    ax.grid(True, alpha=0.4)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    return fig

def build_frontier_table(results: Dict[str, List]) -> pd.DataFrame:
    """
    Build a DataFrame from efficient frontier results, including
    number of assets and a 'Fallback Used' flag.
    """
    if not results or len(results.get("vol", [])) == 0:
        return pd.DataFrame(
            columns=["Return", "Volatility", "Sharpe", "Avg Corr", "Num Assets", "Fallback Used"]
        )

    # Count number of assets in each portfolio
    num_assets = [len(w[0]) for w in results["weights"]]

    # Build DataFrame
    frontier_df = pd.DataFrame({
        "Return": results["return"],
        "Volatility": results["vol"],
        "Sharpe": results["sharpe"],
        "Avg Corr": results["avg_corr"],
        "Num Assets": num_assets,
        "Fallback Used": results.get("fallback", [False] * len(num_assets))
    })

    return frontier_df


# -------------------
# Interval metrics
# -------------------

def compute_interval_metrics(
    data: PortfolioData,
    weights: np.ndarray,
    risk_free: float,
    intervals: Dict[str, Tuple[Optional[str], Optional[str]]],
    trading_days: int,
    method: str,
    tickers: Optional[List[str]] = None
) -> List[IntervalMetrics]:
    metrics: List[IntervalMetrics] = []

    if tickers is not None:
        returns_daily = data.returns_daily[tickers]
        cov_daily = returns_daily.cov().values
    else:
        returns_daily = data.returns_daily
        cov_daily = data.cov_daily

    if returns_daily.shape[1] != len(weights):
        raise ValueError(
            f"Interval metrics mismatch: returns has {returns_daily.shape[1]} columns, "
            f"weights length {len(weights)}"
        )

    for name, (start, end) in intervals.items():
        start_ts = pd.Timestamp(start) if start else None
        end_ts = pd.Timestamp(end) if end else None

        avg_ret_annual = annualized_return_from_total_period(
            returns_daily, weights, start_ts, end_ts, trading_days, method=method
        )
        vol_annual = annualize_vol(weights, cov_daily, trading_days)
        sharpe = (avg_ret_annual - risk_free) / vol_annual if vol_annual > 0 else -np.inf

        metrics.append(IntervalMetrics(
            name=name,
            start=start_ts if start_ts else returns_daily.index[0],
            end=end_ts if end_ts else returns_daily.index[-1],
            mean_annual=avg_ret_annual,
            vol_annual=vol_annual,
            sharpe=sharpe
        ))
    return metrics


# -------------------
# Correlation analysis and plots
# -------------------

def portfolio_correlation_summary(
    weights: np.ndarray,
    returns_daily: pd.DataFrame,
    tickers: Optional[List[str]] = None
) -> Tuple[float, List[Tuple[str, str, float, float]]]:
    """Return average weighted correlation and top 3 weighted-impact correlated pairs."""

    # Deduplicate and drop empty columns before any subsetting
    returns_daily = returns_daily.loc[:, ~returns_daily.columns.duplicated()]
    returns_daily = returns_daily.dropna(axis=1, how="all")

    if tickers is not None:
        # Force exact column set and order
        returns_daily = returns_daily.loc[:, tickers]

    corr_df = returns_daily.corr()
    tickers_effective = corr_df.columns.tolist()
    W = np.array(weights)

    if len(tickers_effective) != len(W):
        raise ValueError(
            f"Dimension mismatch: weights={len(W)} vs corr={len(tickers_effective)} "
            f"(ensure tickers list exactly matches non-zero weights and columns are unique/non-empty)."
        )

    corr = corr_df.values
    mask_matrix = np.ones_like(corr, dtype=bool)
    np.fill_diagonal(mask_matrix, 0)
    denom = (W[:, None] * W[None, :])[mask_matrix].sum()
    weighted_corr = ((W[:, None] * W[None, :] * corr)[mask_matrix].sum() / denom) if denom > 0 else 0.0

    pair_scores: List[Tuple[str, str, float, float]] = []
    for i in range(len(tickers_effective)):
        for j in range(i + 1, len(tickers_effective)):
            cij = float(corr_df.iloc[i, j])
            score = float(W[i] * W[j] * cij)
            pair_scores.append((tickers_effective[i], tickers_effective[j], cij, score))

    pair_scores.sort(key=lambda x: abs(x[3]), reverse=True)
    top_pairs = pair_scores[:3]

    return float(weighted_corr), top_pairs


def plot_weighted_corr_heatmap(
    returns_daily: pd.DataFrame,
    tickers: List[str],
    weights: np.ndarray,
    min_weight: float = 1e-4
):
    """
    Plot a weighted correlation heatmap for non-zero allocations.
    Labels are shortened and rotated for readability.
    """
    # Filter to non-zero allocations
    mask = weights > min_weight
    tickers_f = [t for t, m in zip(tickers, mask) if m]
    weights_f = weights[mask]

    if len(tickers_f) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Weighted correlation heatmap (no non-zero allocations)")
        ax.axis("off")
        return fig

    # Subset returns
    returns_daily_f = returns_daily[tickers_f]
    corr = returns_daily_f.corr().values
    weighted_corr = corr * (weights_f[:, None] * weights_f[None, :])

    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(tickers_f) * 0.4),  # dynamic width
                                    max(6, len(tickers_f) * 0.4))) # dynamic height
    cax = ax.imshow(weighted_corr, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_title("Weighted correlation heatmap (non-zero allocations)")

    # Format labels: ticker + weight %, rotated for readability
    ax.set_xticks(range(len(tickers_f)))
    ax.set_yticks(range(len(tickers_f)))
    ax.set_xticklabels([f"{t} ({w:.1%})" for t, w in zip(tickers_f, weights_f)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"{t} ({w:.1%})" for t, w in zip(tickers_f, weights_f)],
                       fontsize=8)

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Weighted correlation")
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(
    returns_daily: pd.DataFrame,
    tickers: List[str],
    weights: np.ndarray,
    min_weight: float = 0.01,
    max_assets: int = 50
):
    """
    Plot a weighted correlation heatmap for tickers with allocation >= min_weight.
    Caps the number of assets and figure size to avoid huge rasters.
    """
    mask = weights >= min_weight
    tickers_f = [t for t, m in zip(tickers, mask) if m]
    weights_f = weights[mask]

    if len(tickers_f) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Correlation heatmap (no tickers above threshold)")
        ax.axis("off")
        return fig

    # Limit to top-N by weight to avoid gigantic plots
    if len(tickers_f) > max_assets:
        idx = np.argsort(weights_f)[::-1][:max_assets]
        tickers_f = [tickers_f[i] for i in idx]
        weights_f = weights_f[idx]

    returns_f = returns_daily[tickers_f]
    corr = returns_f.corr().values
    weighted_corr = corr * (weights_f[:, None] * weights_f[None, :])

    # Cap figure size
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(weighted_corr, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_title(f"Weighted correlation heatmap (≥ {min_weight:.2%} allocation)")

    ax.set_xticks(range(len(tickers_f)))
    ax.set_yticks(range(len(tickers_f)))
    ax.set_xticklabels([f"{t} ({w:.1%})" for t, w in zip(tickers_f, weights_f)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"{t} ({w:.1%})" for t, w in zip(tickers_f, weights_f)],
                       fontsize=8)

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Weighted correlation")
    plt.tight_layout()
    return fig


# -------------------
# CLI entrypoint
# -------------------

# -------------------
# CLI entrypoint
# -------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ETF Portfolio Sharpe Optimizer")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of tickers to include")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--rf", type=float, required=True, help="Risk-free rate (annual, decimal)")
    parser.add_argument("--min_weight", type=float, default=0.0, help="Minimum asset weight in final portfolio")
    parser.add_argument("--max_weight", type=float, default=1.0, help="Maximum asset weight")
    parser.add_argument("--return_method", choices=["average", "cagr"], default="average", help="Return calculation method")
    parser.add_argument("--plot_frontier", action="store_true", help="Plot efficient frontier")
    parser.add_argument("--diversification_alpha", type=float, default=0.0, help="Penalty factor for diversification (higher = more penalty)")
    parser.add_argument("--min_obs", type=int, default=252, help="Minimum number of daily returns required for a ticker")
    parser.add_argument("--cov_shrinkage", type=float, default=0.1, help="Shrinkage intensity for covariance matrix")

    args = parser.parse_args()

    inputs = PortfolioInputs(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        risk_free=args.rf,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        return_method=args.return_method,
    )

    # Build data (5‑tuple)
    data, valid_tickers, invalid_symbols, failed_downloads, dropped_for_history = build_portfolio_data(
        inputs.tickers, inputs.start, inputs.end, min_obs=args.min_obs
    )

    # Handle empty data
    if data.returns_daily.empty or data.returns_daily.shape[1] == 0:
        print("❌ No valid return data available for the given tickers/date range.")
        if invalid_symbols:
            print(f"⚠️ {len(invalid_symbols)} invalid symbols (sample: {invalid_symbols[:10]})")
        if failed_downloads:
            print(f"⚠️ {len(failed_downloads)} failed downloads (sample: {failed_downloads[:10]})")
        if dropped_for_history:
            print(f"⚠️ {len(dropped_for_history)} dropped for insufficient history (sample: {dropped_for_history[:10]})")
        exit(1)

    # Report diagnostics
    if invalid_symbols:
        print(f"⚠️ {len(invalid_symbols)} invalid symbols (sample: {invalid_symbols[:10]})")
    if failed_downloads:
        print(f"⚠️ {len(failed_downloads)} failed downloads (sample: {failed_downloads[:10]})")
    if dropped_for_history:
        print(f"⚠️ {len(dropped_for_history)} dropped for insufficient history (sample: {dropped_for_history[:10]})")

    # Update inputs to only valid tickers
    inputs.tickers = valid_tickers

    # Run optimization
    w_opt, info = optimize_max_sharpe(
        data, inputs,
        interval_start=inputs.start,
        interval_end=inputs.end,
        diversification_alpha=args.diversification_alpha,
        cov_shrinkage=args.cov_shrinkage
    )

    used_tickers = [t for t, w in zip(info["tickers"], w_opt) if w > 1e-8]

    print("\n✅ Optimization result:")
    print(f"Success: {info['success']} ({info['message']})")
    print(f"Iterations: {info['iterations']}")
    print(f"Avg annual return: {info['avg_return_annual']:.2%}")
    print(f"Annual volatility: {info['vol_annual']:.2%}")
    print(f"Sharpe: {info['sharpe']:.3f}")
    print(f"Average weighted correlation: {info['avg_correlation']:.2f}")

    print("\n📊 Portfolio composition:")
    print(f"Universe size (post‑filter): {len(inputs.tickers)}")
    print(f"Used in optimized portfolio: {len(used_tickers)}")
    print(f"Zero-weighted in optimization: {len(inputs.tickers) - len(used_tickers)}")

    print("\nOptimized weights:")
    for t, w in zip(info["tickers"], w_opt):
        if w > 1e-8:
            print(f"{t}: {w:.2%}")

    if args.plot_frontier:
        print("\n📈 Efficient Frontier Portfolios:")
        res = efficient_frontier(data, inputs, n_points=20, cov_shrinkage=args.cov_shrinkage)
        frontier_df = build_frontier_table(res)
        for i, row in frontier_df.iterrows():
            tickers_i, weights_i = res["weights"][i]
            sorted_pairs = sorted(zip(tickers_i, weights_i), key=lambda x: x[1], reverse=True)
            top3 = ", ".join([f"{t} ({w:.1%})" for t, w in sorted_pairs[:3]])
            print(f"Point {i+1}: Return={row['Return']:.2%}, Vol={row['Volatility']:.2%}, "
                  f"Sharpe={row['Sharpe']:.3f}, NumAssets={row['Num Assets']}, Top Holdings={top3}")


