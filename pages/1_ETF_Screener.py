# pages/1_ETF_Screener.py
import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="ETF Screener", layout="wide")
st.title("ETF Screener – Ratings & Performance")

# ----------------------------
# Load and clean screener results
# ----------------------------
@st.cache_data
def load_screener(path: str) -> pd.DataFrame:
    # Read raw lines and skip the first header line (it causes misalignment)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Drop the header line explicitly
    data_lines = [ln.rstrip("\n") for ln in lines[1:] if ln.strip()]

    rows = []
    for ln in data_lines:
        # Split on 2+ spaces (Morningstar export uses wide spacing)
        parts = re.split(r"\s{2,}", ln.strip())
        # Remove empty tokens
        parts = [p for p in parts if p != ""]

        # Remove any date-stamp fields like "As of 09/30/2025"
        parts = [p for p in parts if not re.search(r"\bAs of\s+\d{2}/\d{2}/\d{4}\b", p)]

        # After removing date fields, we expect:
        # [Symbol, Company, Rating_overall, Rating_3Y, Rating_5Y, Rating_10Y, Return_1Y, Return_3Y, Return_5Y, Return_10Y]
        rows.append(parts)

    # Normalize each row to exactly 10 columns (pad or trim)
    EXPECTED = 10
    normalized = []
    for r in rows:
        if len(r) < EXPECTED:
            r = r + [None] * (EXPECTED - len(r))
        elif len(r) > EXPECTED:
            r = r[:EXPECTED]
        normalized.append(r)

    df = pd.DataFrame(normalized, columns=[
        "Symbol",
        "Company",
        "Rating_overall",
        "Rating_3Y",
        "Rating_5Y",
        "Rating_10Y",
        "Return_1Y",
        "Return_3Y",
        "Return_5Y",
        "Return_10Y",
    ])

    # Strip whitespace
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # Clean ratings: robustly extract a standalone digit 1–5
    def clean_rating(x: str) -> int:
        m = re.search(r"\b([1-5])\b", str(x))
        return int(m.group(1)) if m else 0

    for col in ["Rating_overall", "Rating_3Y", "Rating_5Y", "Rating_10Y"]:
        df[col] = df[col].apply(clean_rating).astype(int)

    # Clean returns: strip % and coerce to float
    def clean_return(x: str) -> float:
        s = str(x).replace("%", "").replace(",", "").strip()
        try:
            return float(s)
        except Exception:
            return float("nan")

    for col in ["Return_1Y", "Return_3Y", "Return_5Y", "Return_10Y"]:
        df[col] = df[col].apply(clean_return)

    # Drop rows missing a symbol (parsing safety)
    df = df[df["Symbol"].str.len() > 0]

    return df


df = load_screener("ScreenerResults.txt")

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")

min_rating = st.sidebar.selectbox("Minimum Morningstar rating (overall)", [1, 2, 3, 4, 5], index=4)

# Robust slider ranges (handle NaNs and equal min/max)
def ret_minmax(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0, 100
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        hi = lo + 1.0
    lo = max(-100.0, lo)
    hi = min(200.0, hi)
    return int(lo), int(hi)

r1_min, r1_max = ret_minmax(df["Return_1Y"])
r3_min, r3_max = ret_minmax(df["Return_3Y"])
r5_min, r5_max = ret_minmax(df["Return_5Y"])
r10_min, r10_max = ret_minmax(df["Return_10Y"])

min_return_1y = st.sidebar.slider("Minimum 1Y return (%)", r1_min, r1_max, r1_min)
min_return_3y = st.sidebar.slider("Minimum 3Y return (%)", r3_min, r3_max, r3_min)
min_return_5y = st.sidebar.slider("Minimum 5Y return (%)", r5_min, r5_max, r5_min)
min_return_10y = st.sidebar.slider("Minimum 10Y return (%)", r10_min, r10_max, r10_min)

# ----------------------------
# Apply filters directly to the single table
# ----------------------------
filtered = df.copy()
filtered = filtered[filtered["Rating_overall"] >= min_rating]

for col, min_val in [
    ("Return_1Y", float(min_return_1y)),
    ("Return_3Y", float(min_return_3y)),
    ("Return_5Y", float(min_return_5y)),
    ("Return_10Y", float(min_return_10y)),
]:
    filtered[col] = pd.to_numeric(filtered[col], errors="coerce").fillna(float("-inf"))
    filtered = filtered[filtered[col] >= min_val]

# ----------------------------
# Show single table (updates live)
# ----------------------------
st.subheader("All ETFs (filtered)")
if filtered.empty:
    st.warning("No ETFs match your filters. Try lowering thresholds.")
else:
    # Reset index to start at 1 instead of 0
    filtered_display = filtered.reset_index(drop=True)
    filtered_display.index = filtered_display.index + 1

    st.dataframe(
        filtered_display.style.format({
            "Return_1Y": "{:.2f}%",
            "Return_3Y": "{:.2f}%",
            "Return_5Y": "{:.2f}%",
            "Return_10Y": "{:.2f}%"
        }),
        width="stretch"
    )


# ----------------------------
# Selection → Optimizer
# ----------------------------
st.subheader("Select ETFs for Portfolio Optimizer")
selected = st.multiselect("Choose ETFs to send to optimizer", filtered["Symbol"].tolist())

if st.button("Send to Portfolio Optimizer"):
    st.session_state["selected_etfs"] = selected
    st.success(f"Sent {len(selected)} ETFs to Portfolio Optimizer")
