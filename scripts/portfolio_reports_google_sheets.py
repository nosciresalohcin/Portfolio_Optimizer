#!/usr/bin/env python3
"""
portfolio_reports_google_sheets.py
Corrected and cleaned version — optimizer + Google Sheets export only.
No backtest code is included in this file.
"""

import os
import sys
import time
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
from scipy.linalg import pinv
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from dotenv import load_dotenv

# ---------------- USER CONFIG / PATHS ----------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config', '.env')
if os.path.exists(CONFIG_PATH):
    load_dotenv(CONFIG_PATH)
else:
    # Only warn — do not exit (keeps script flexible)
    print("⚠️ Warning: config/.env not found — proceeding with defaults.")

DATA_CSV = os.getenv('DATA_CSV', os.path.join(ROOT_DIR, 'data', 'sp500_clean_wide_latest.csv'))
SECTOR_MAP_CSV = os.getenv('SECTOR_MAP_CSV', os.path.join(ROOT_DIR, 'data', 'ticker_sector_map.csv'))
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')  # optional
SHEET_TITLE = os.getenv('SHEET_TITLE', 'Portfolio_Reports')
OUT_FOLDER = os.getenv('OUT_FOLDER', os.path.join(ROOT_DIR, 'data', 'portfolio_reports_google'))

# Configurable tuning params (env overrides allowed)
FAST_MODE = True
FAST_TOP_N = int(os.getenv('FAST_TOP_N', '300'))
MIN_OBS = int(os.getenv('MIN_OBS', '200'))
TRADING_DAYS = int(os.getenv('TRADING_DAYS', '252'))
MAX_WEIGHT = float(os.getenv('MAX_WEIGHT', '0.05'))
RF_ANNUAL = float(os.getenv('RF_ANNUAL', '0.04'))
LOOKBACK_YEARS = int(os.getenv('LOOKBACK_YEARS', '3'))
BATCH_SIZE_GOOGLEFINANCE = int(os.getenv('BATCH_SIZE_GOOGLEFINANCE', '80'))

os.makedirs(OUT_FOLDER, exist_ok=True)

# ---------- Helper: Google Sheets auth ----------
def gsheets_auth(credentials_json_path):
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_json_path, scopes)
    client = gspread.authorize(creds)
    return client

# ---------- safer sheet writer ----------
def write_df_to_sheet(sh, df, sheet_name):
    """Write DataFrame `df` to gspread.Spreadsheet `sh` as worksheet `sheet_name`."""
    if df is None or df.empty:
        print(f"Skipping writing empty DataFrame to sheet {sheet_name}")
        return
    try:
        # Delete existing worksheet if present
        try:
            wks = sh.worksheet(sheet_name)
            sh.del_worksheet(wks)
        except Exception:
            pass
        rows = max(10, len(df) + 5)
        cols = max(2, len(df.columns) + 1)
        wks = sh.add_worksheet(title=sheet_name, rows=str(rows), cols=str(cols))
        data = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
        wks.update('A1', data, value_input_option='USER_ENTERED')
    except Exception as e:
        print(f"Failed to write sheet {sheet_name}: {e}")

# ---------- update local CSV with Google Sheets GOOGLEFINANCE (hardened) ----------
def update_csv_with_googlefinance(local_csv, service_account_file, batch_size=BATCH_SIZE_GOOGLEFINANCE):
    print("Starting update via Google Sheets GOOGLEFINANCE...")
    if not os.path.exists(local_csv):
        raise FileNotFoundError(f"Local CSV not found: {local_csv}")
    df = pd.read_csv(local_csv)
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col).reset_index(drop=True)
    max_date = df[date_col].max()
    if pd.isna(max_date):
        raise RuntimeError('Could not determine max date in local CSV (dates may be malformed).')

    today = datetime.utcnow().date()
    start_fetch = (max_date + timedelta(days=1)).date()
    if start_fetch > today:
        print('CSV is already up-to-date through', max_date.date())
        return df

    tickers = [c for c in df.columns[1:]]
    client = gsheets_auth(service_account_file)
    temp_sheet_title = f"tmp_gf_{int(time.time())}"
    sheet = client.create(temp_sheet_title)

    all_new = []
    try:
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            wks = sheet.add_worksheet(title=f"batch_{i//batch_size}", rows=2000, cols=10)
            rows_to_write = [["ticker", "date", "price"]]
            for tk in batch:
                formula = f'=GOOGLEFINANCE("{tk}", "price", DATE({start_fetch.year},{start_fetch.month},{start_fetch.day}), DATE({today.year},{today.month},{today.day}), "DAILY")'
                rows_to_write.append([tk, formula, ""])
            wks.update('A1', rows_to_write)
            time.sleep(1.5 + min(3.0, 0.05 * len(batch)))
            vals = wks.get_all_values()
            for row_idx, row in enumerate(vals):
                if row_idx == 0:
                    continue
                tk = row[0].strip()
                if tk == "":
                    continue
                j = row_idx
                block = []
                while j < len(vals):
                    r = vals[j]
                    try:
                        cand_date = pd.to_datetime(r[1], errors='coerce')
                    except Exception:
                        break
                    cand_price = None
                    for c in r[1:4]:
                        try:
                            v = float(str(c).replace(',', ''))
                            cand_price = v
                            break
                        except Exception:
                            continue
                    if pd.isna(cand_date) or cand_price is None:
                        break
                    block.append((cand_date.date(), cand_price))
                    j += 1
                if block:
                    for d, p in block:
                        all_new.append({'ticker': tk, 'date': pd.to_datetime(d), 'price': p})
            try:
                sheet.del_worksheet(wks)
            except Exception:
                pass

        if not all_new:
            print('No new data returned by GOOGLEFINANCE (possible rate-limiting or parsing failure).')
            return df

        new_df = pd.DataFrame(all_new)
        pivot = new_df.pivot_table(index='date', columns='ticker', values='price')
        pivot = pivot.sort_index()
        pivot.index.name = df.columns[0]
        existing = df.set_index(df.columns[0])
        merged = pd.concat([existing, pivot[~pivot.index.isin(existing.index)]], sort=False)
        merged = merged.reset_index()
        merged.to_csv(local_csv, index=False)
        print('CSV updated and saved to', local_csv)
        return merged
    finally:
        try:
            client.del_spreadsheet(sheet.id)
        except Exception:
            pass

# ---------- Fama-French regression (robust detection) ----------
def run_fama_french_regressions(portfolio_weights, prices_df, ff_df):
    results = {}
    ff = ff_df.copy()
    ff.index = pd.to_datetime(ff.index)

    # identify RF column
    rf_candidates = [c for c in ff.columns if c.strip().upper() in {"RF", "RISKFREE", "RISK-FREE", "RISK FREE"}]
    rf_col = rf_candidates[0] if rf_candidates else None

    # identify factor columns (case/format tolerant)
    def find_factor_cols(candidates):
        found = []
        for cand in candidates:
            for col in ff.columns:
                if col.replace(' ', '').replace('-', '').replace('_', '').upper() == cand.replace('-', '').replace('_', '').upper():
                    if col not in found:
                        found.append(col)
        return found

    factor_candidates = find_factor_cols(['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MKT'])
    if not factor_candidates:
        # fallback: any numeric columns except RF
        factor_candidates = [c for c in ff.columns if c != rf_col]

    for name, w in portfolio_weights.items():
        w_full = w.reindex(prices_df.columns).fillna(0.0)
        port_daily = prices_df.pct_change(fill_method=None).dropna(how='all').dot(w_full.values)
        common_index = port_daily.index.intersection(ff.index)
        if len(common_index) < 30:
            print(f"⚠️ Skipping regression for {name}: insufficient overlapping days ({len(common_index)}).")
            continue

        if rf_col is not None and rf_col in ff.columns:
            Y = port_daily.loc[common_index].values - ff.loc[common_index, rf_col].values
        else:
            print(f"⚠️ RF column not found in FF data; using portfolio returns (no RF subtraction) for {name}.")
            Y = port_daily.loc[common_index].values

        X = ff.loc[common_index, factor_candidates].values
        if X is None or len(X) == 0 or len(Y) == 0:
            print(f"⚠️ Skipping regression for {name}: no overlapping factor data.")
            continue

        try:
            reg = LinearRegression().fit(X, Y)
            r2 = reg.score(X, Y)
            coefs = dict(zip(ff.loc[common_index, factor_candidates].columns.tolist(), reg.coef_.tolist()))
            results[name] = {"r2": r2, "factor_cols": factor_candidates, "coefs": coefs}
        except Exception as e:
            print(f"⚠️ Regression failed for {name}: {e}")
            continue

    return results

# ---------- optimization helpers ----------
def series_from_weights(w_array, assets_local, prices_df_local):
    s = pd.Series(w_array, index=assets_local)
    return s.reindex(prices_df_local.columns).fillna(0.0)

def annualize_mean(daily_mean, trading_days=TRADING_DAYS):
    return daily_mean * trading_days

def annualize_vol(daily_std, trading_days=TRADING_DAYS):
    return daily_std * (trading_days ** 0.5)

def estimate_covariance(rets):
    try:
        lw = LedoitWolf().fit(
            rets.ffill().bfill().values  # <- modern syntax, same behavior
        )
        Sigma = lw.covariance_
    except Exception as e:
        print('LedoitWolf failed, falling back to sample covariance:', e)
        sample_cov = rets.cov() * TRADING_DAYS
        Sigma = sample_cov.values
    return Sigma

def build_portfolios(prices_df, allow_short=False, rf_annual=RF_ANNUAL, max_weight=MAX_WEIGHT, min_obs=MIN_OBS):
    rets = prices_df.pct_change(fill_method=None).dropna(how='all').dropna(axis=1, how='all')
    valid_cols = [c for c in rets.columns if rets[c].dropna().shape[0] >= min_obs]
    if not valid_cols:
        raise RuntimeError("No assets with sufficient observations. Check MIN_OBS or your data.")
    rets = rets[valid_cols].copy()

    mu_daily = rets.mean()
    mu = annualize_mean(mu_daily)
    Sigma = estimate_covariance(rets)

    tickers = rets.columns.tolist()
    n = len(tickers)
    ones = np.ones(n)

    Sigma = np.array(Sigma)
    if Sigma.shape[0] != n or Sigma.shape[1] != n:
        Sigma = (rets.cov() * TRADING_DAYS).values

    rf = rf_annual

    def project_weights(raw_w, allow_short_local=allow_short):
        w = np.array(raw_w, dtype=float)
        if not allow_short_local:
            w = np.clip(w, 0.0, max_weight)
        else:
            w = np.clip(w, -max_weight, max_weight)
        s = w.sum()
        if abs(s) < 1e-12:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
        if not allow_short_local:
            for _ in range(10):
                over = np.maximum(0, w - max_weight)
                if over.sum() <= 1e-12:
                    break
                w = np.minimum(w, max_weight)
                rem = 1.0 - w.sum()
                if rem <= 0:
                    non_max = (w < max_weight - 1e-12)
                    if non_max.sum() == 0:
                        break
                    w[non_max] = w[non_max] / w[non_max].sum() * rem
                else:
                    non_max = (w < max_weight - 1e-12)
                    if non_max.sum() == 0:
                        break
                    w[non_max] = w[non_max] + (w[non_max] / w[non_max].sum()) * rem
        else:
            w = w / w.sum()
        return pd.Series(w, index=tickers)

    try:
        x = mu.values - rf
        Sigma_inv = pinv(Sigma)
        raw_tan = Sigma_inv.dot(x)
        w_tangency = project_weights(raw_tan, allow_short)
    except Exception as e:
        print("⚠️ Tangency computation failed, falling back to equal-weight:", e)
        w_tangency = pd.Series(np.ones(n) / n, index=tickers)

    try:
        raw_minvar = pinv(Sigma).dot(ones)
        w_minvar = project_weights(raw_minvar, allow_short)
    except Exception as e:
        print("⚠️ Min-variance computation failed, falling back to equal-weight:", e)
        w_minvar = pd.Series(np.ones(n) / n, index=tickers)

    w_equal = pd.Series(np.ones(n) / n, index=tickers)

    try:
        vols = rets.std() * (TRADING_DAYS ** 0.5)
        inv_vol = 1.0 / (vols.replace(0, np.nan))
        inv_vol = inv_vol.fillna(inv_vol.mean())
        raw_invvol = inv_vol.values
        w_invvol = project_weights(raw_invvol, allow_short=False)
    except Exception:
        w_invvol = w_equal.copy()

    portfolios = {
        'tangency_max_sharpe': w_tangency,
        'min_variance': w_minvar,
        'equal_weight': w_equal,
        'inv_vol': w_invvol
    }

    diagnostics = {'mu_annual': mu, 'tickers': tickers, 'Sigma': Sigma}
    return portfolios, diagnostics

# ---------- composition & sector helpers ----------
def compute_composition(weights_series, prices_df=None):
    w = weights_series[weights_series.abs() > 0]
    s = w.sort_values(ascending=False)
    top = s.head(20).rename('weight').reset_index().rename(columns={'index': 'ticker'})
    h = (w ** 2).sum()
    return {'top_holdings': top, 'herfindahl': float(h)}

def compute_sector_exposure(weights_series, sector_map_df):
    w = weights_series.copy().rename('weight')
    mapping = sector_map_df.set_index(sector_map_df.columns[0]).iloc[:,0].to_dict()
    mapped = w.rename(index=lambda t: t.strip() if isinstance(t, str) else t).to_frame()
    mapped['sector'] = mapped.index.map(mapping).fillna('Unknown')
    sector_exposure = mapped.groupby('sector')['weight'].sum().sort_values(ascending=False)
    return sector_exposure

# ---------- Google Sheets export orchestration ----------
def export_portfolios_to_gsheets(client_json_path, spreadsheet_title, portfolios_dict, diagnostics, sector_map_csv=None):
    """
    Creates (or opens) a spreadsheet named spreadsheet_title and writes:
      - A worksheet per portfolio with weights (ticker, weight)
      - A diagnostics sheet with mu and simple stats
      - Sector exposures (if sector_map_csv provided)
    Returns the gspread.Spreadsheet object (or None on failure)
    """
    try:
        client = gsheets_auth(client_json_path)
    except Exception as e:
        print("Failed to authenticate to Google Sheets:", e)
        return None

    try:
        sh = client.open(spreadsheet_title)
    except Exception:
        sh = client.create(spreadsheet_title)

    for pname, w in portfolios_dict.items():
        w_df = w.sort_values(ascending=False).rename('weight').reset_index().rename(columns={'index': 'ticker'})
        write_df_to_sheet(sh, w_df, sheet_name=pname)

    mu = diagnostics.get('mu_annual')
    mu_df = pd.DataFrame({'ticker': mu.index, 'mu_annual': mu.values})
    write_df_to_sheet(sh, mu_df.sort_values('mu_annual', ascending=False), sheet_name='diagnostics_mu')

    if sector_map_csv and os.path.exists(sector_map_csv):
        try:
            sm = pd.read_csv(sector_map_csv)
            for pname, w in portfolios_dict.items():
                se = compute_sector_exposure(w, sm).rename('exposure').reset_index().rename(columns={'index': 'sector'})
                write_df_to_sheet(sh, se, sheet_name=f"{pname}_sectors")
        except Exception as e:
            print("Warning: failed to compute/write sector exposures:", e)

    print(f"Exported portfolios to Google Sheets: {spreadsheet_title}")
    return sh

# ---------- persistence helpers ----------
def save_weights_to_csv(portfolios_dict, out_dir=OUT_FOLDER, prefix='weights'):
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = {}
    for pname, w in portfolios_dict.items():
        safe_name = pname.replace(" ", "_").replace("/", "_")
        path = os.path.join(out_dir, f"{prefix}_{safe_name}.csv")
        w.sort_values(ascending=False).to_csv(path, header=['weight'])
        saved_paths[pname] = path
        print(f"💾 Saved weights for {pname} → {path}")
    return saved_paths

# ---------- main orchestration ----------
def main_pipeline(data_csv=DATA_CSV, sector_map_csv=SECTOR_MAP_CSV,
                  service_account_file=SERVICE_ACCOUNT_FILE, sheet_title=SHEET_TITLE,
                  out_folder=OUT_FOLDER, save_weights=False):
    if not os.path.exists(data_csv):
        raise SystemExit(f"❌ Data CSV not found: {data_csv}")

    print(f"📈 Loading price data from {data_csv}...")
    prices = pd.read_csv(data_csv, parse_dates=[0])
    date_col = prices.columns[0]
    prices = prices.sort_values(date_col).set_index(date_col)
    prices = prices.dropna(how='all').dropna(axis=1, how='all')

    print(f"✅ Loaded {prices.shape[0]} rows × {prices.shape[1]} tickers of price data")

    print("🚀 Running optimizer...")
    portfolios, diagnostics = build_portfolios(
        prices,
        allow_short=False,
        rf_annual=RF_ANNUAL,
        max_weight=MAX_WEIGHT,
        min_obs=MIN_OBS
    )

    print(f"✅ Generated {len(portfolios)} portfolio(s): {list(portfolios.keys())}")

    saved = {}
    if save_weights:
        print(f"💾 Saving weight CSVs to {out_folder} ...")
        saved = save_weights_to_csv(portfolios, out_dir=out_folder, prefix='weights')
    else:
        print("ℹ️ Skipping saving weight CSVs (use --save_weights to enable).")

    if service_account_file and os.path.exists(service_account_file):
        print(f"📤 Exporting to Google Sheets: {sheet_title}")
        try:
            sh = export_portfolios_to_gsheets(
                service_account_file,
                sheet_title,
                portfolios,
                diagnostics,
                sector_map_csv=sector_map_csv
            )
            if sh:
                print(f"✅ Google Sheets export succeeded — Spreadsheet ID: {getattr(sh, 'id', 'unknown')}")
        except Exception as e:
            print(f"⚠️ Google Sheets export failed: {e}")
    else:
        print("⚠️ No Google credentials provided, skipping Sheets export.")

    print("🏁 Pipeline completed successfully.")
    return portfolios, diagnostics, saved

# ---------- CLI entrypoint ----------
if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        description="Portfolio construction and Google Sheets reporting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example usage:
              python portfolio_reports_google_sheets.py
              python portfolio_reports_google_sheets.py --data_csv data/sp500_clean_wide_latest.csv
              python portfolio_reports_google_sheets.py --no_sheets
              python portfolio_reports_google_sheets.py --save_weights
        """),
    )

    parser.add_argument("--data_csv", default=DATA_CSV, help="Path to cleaned wide-format price CSV (default: %(default)s)")
    parser.add_argument("--sector_map_csv", default=SECTOR_MAP_CSV, help="Optional path to ticker→sector mapping CSV (default: %(default)s)")
    parser.add_argument("--service_account_file", default=SERVICE_ACCOUNT_FILE, help="Path to Google Service Account credentials JSON (default: %(default)s)")
    parser.add_argument("--sheet_title", default=SHEET_TITLE, help="Google Sheets workbook title for report export (default: %(default)s)")
    parser.add_argument("--out_folder", default=OUT_FOLDER, help="Local folder for weight CSVs and summaries (default: %(default)s)")
    parser.add_argument("--no_sheets", action="store_true", help="Skip Google Sheets export even if credentials are available")
    parser.add_argument("--save_weights", action="store_true", help="Save per-portfolio weight CSVs locally into out_folder (disabled by default)")

    args = parser.parse_args()

    try:
        print("🚀 Starting portfolio optimization pipeline...")
        portfolios_out, diag_out, saved_paths = main_pipeline(
            data_csv=args.data_csv,
            sector_map_csv=args.sector_map_csv,
            service_account_file=None if args.no_sheets else args.service_account_file,
            sheet_title=args.sheet_title,
            out_folder=args.out_folder,
            save_weights=args.save_weights,
        )
        print("\n✅ Done! Generated portfolios:")
        for pname, weights in portfolios_out.items():
            print(f"  • {pname:20s} — {(weights > 0).sum()} holdings")
        if saved_paths:
            print(f"\nSaved weight files to: {os.path.abspath(args.out_folder)}")
            for k, p in saved_paths.items():
                print(f"   - {k}: {p}")
        else:
            print("\nNo local weight files saved (use --save_weights to create them).")

        if not args.no_sheets and args.service_account_file and os.path.exists(args.service_account_file):
            print(f"📤 Results also exported to Google Sheets workbook: {args.sheet_title}")
        print("🏁 Pipeline finished successfully.")
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise
