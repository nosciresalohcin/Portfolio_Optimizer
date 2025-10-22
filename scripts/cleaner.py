import pandas as pd
import numpy as np

# Path to your file
file_path = "spdata2.csv"

# --- STEP 1: Read the raw lines manually ---
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# --- STEP 2: Parse header tickers and column names ---
header_line = lines[0].strip().split(',')
header_tickers = [t for t in header_line if t not in ['', 'Date', 'Close']]

# There are 2 columns per ticker: Date and Close
num_tickers = len(header_tickers)
expected_cols = num_tickers * 2

# --- STEP 3: Load with correct column count ---
df = pd.read_csv(file_path, skiprows=2, header=None, names=sum([[f"{t}_Date", f"{t}_Close"] for t in header_tickers], []), nrows=None)

# --- STEP 4: Convert each ticker’s pair of columns into a long-form series ---
clean_data = []
for t in header_tickers:
    subset = df[[f"{t}_Date", f"{t}_Close"]].dropna(how='all')
    subset = subset.rename(columns={f"{t}_Date": "Date", f"{t}_Close": "Close"})
    subset["Ticker"] = t
    subset["Date"] = pd.to_datetime(subset["Date"], errors="coerce")
    subset["Close"] = pd.to_numeric(subset["Close"], errors="coerce")
    subset = subset.dropna(subset=["Date", "Close"])
    clean_data.append(subset)

# --- STEP 5: Combine all tickers into a unified dataframe ---
long_df = pd.concat(clean_data, ignore_index=True)

# --- STEP 6: Drop duplicates, sort, and pivot if desired ---
long_df = long_df.drop_duplicates(subset=["Ticker", "Date"])
long_df = long_df.sort_values(["Ticker", "Date"])

# --- STEP 7: Optional - create pivot (tickers as columns, date as index)
wide_df = long_df.pivot(index="Date", columns="Ticker", values="Close").sort_index()

# --- STEP 8: Sanity checks ---
print(f"✅ Parsed {len(header_tickers)} tickers.")
print(f"✅ Total rows: {len(long_df):,}")
print(f"✅ Date range: {long_df['Date'].min().date()} → {long_df['Date'].max().date()}")
print(f"✅ Example tickers: {header_tickers[:10]}")

# --- STEP 9: Save cleaned outputs ---
#long_df.to_csv("sp500_clean_long.csv", index=False)
wide_df.to_csv("sp500_clean_wide.csv")
