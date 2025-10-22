import streamlit as st
import pandas as pd
import datetime
import numpy as np

# Import your utilities
from portfolio_optimizer import (
    PortfolioInputs,
    build_portfolio_data,
    optimize_max_sharpe,
    efficient_frontier,
    plot_frontier,
    build_frontier_table,
    plot_correlation_heatmap
)

# -------------------
# Sidebar inputs
# -------------------
st.sidebar.header("Portfolio Settings")

tickers_str = st.sidebar.text_area("Tickers (comma-separated)", "")
tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]

yesterday = datetime.date.today() - datetime.timedelta(days=1)
start = st.sidebar.date_input("Start date", value=datetime.date(2010, 1, 1))
end = st.sidebar.date_input("End date", value=yesterday)

rf = st.sidebar.number_input("Risk-free (annual, decimal)", value=0.03, step=0.005, format="%.3f")
min_w = st.sidebar.number_input("Min weight (final portfolio)", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
max_w = st.sidebar.number_input("Max weight", value=0.5, min_value=0.0, max_value=1.0, step=0.01)
return_method = st.sidebar.selectbox("Return method", ["average", "cagr"], index=1)

div_alpha = st.sidebar.number_input("Diversification penalty alpha", value=0.0, min_value=0.0, step=0.01)
cov_shrinkage = st.sidebar.slider("Covariance shrinkage", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
min_obs = st.sidebar.number_input("Minimum daily returns required", value=252, min_value=0, step=30)

# Frontier visualization controls
st.sidebar.markdown("### Frontier visualization")
plot_frontier_flag = st.sidebar.checkbox("Show efficient frontier", value=True)
frontier_color_by = st.sidebar.selectbox("Color metric", ["sharpe", "corr"], index=0)
scale_marker_size = st.sidebar.checkbox("Scale marker size by number of assets", value=True)

# Correlation heatmap controls
plot_corr_flag = st.sidebar.checkbox("Show correlation heatmap", value=True)
min_weight_heatmap = st.sidebar.number_input("Min weight threshold for heatmap", value=0.01, step=0.01)

# Run button
run = st.sidebar.button("Run optimization")

# -------------------
# Main app
# -------------------
st.title("ETF Portfolio Optimizer")

if run:
    inputs = PortfolioInputs(
        tickers=tickers,
        start=start,
        end=end,
        risk_free=rf,
        min_weight=min_w,
        max_weight=max_w,
        return_method=return_method,
    )

    # -------------------
    # Data build + logging
    # -------------------
    data, valid_tickers, invalid_symbols, failed_downloads, dropped_for_history = build_portfolio_data(
        inputs.tickers, inputs.start, inputs.end, min_obs=min_obs
    )

    # Handle empty data
    if data.returns_daily.empty or data.returns_daily.shape[1] == 0:
        st.error("❌ No valid return data available for the given tickers/date range.")

        if invalid_symbols:
            st.warning(f"⚠️ {len(invalid_symbols)} invalid symbols.")
            with st.expander("📋 Show invalid symbols"):
                st.write(invalid_symbols)

        if failed_downloads:
            st.warning(f"⚠️ {len(failed_downloads)} failed downloads.")
            with st.expander("📋 Show failed downloads"):
                st.write(failed_downloads)

        if dropped_for_history:
            st.warning(f"⚠️ {len(dropped_for_history)} dropped for insufficient history (< {min_obs} returns).")
            with st.expander("📋 Show dropped for history"):
                st.write(dropped_for_history)

        st.stop()

    # Update inputs to only valid tickers
    inputs.tickers = valid_tickers

    # Report diagnostics (if any survived)
    if invalid_symbols:
        st.warning(f"⚠️ {len(invalid_symbols)} invalid symbols.")
        with st.expander("📋 Show invalid symbols"):
            st.write(invalid_symbols)

    if failed_downloads:
        st.warning(f"⚠️ {len(failed_downloads)} failed downloads.")
        with st.expander("📋 Show failed downloads"):
            st.write(failed_downloads)

    if dropped_for_history:
        st.warning(f"⚠️ {len(dropped_for_history)} dropped for insufficient history (< {min_obs} returns).")
        with st.expander("📋 Show dropped for history"):
            st.write(dropped_for_history)

    if not invalid_symbols and not failed_downloads and not dropped_for_history:
        st.success("✅ All tickers passed the data checks.")


    # -------------------
    # Run optimization
    # -------------------
    w_opt, info = optimize_max_sharpe(
        data, inputs,
        interval_start=inputs.start,
        interval_end=inputs.end,
        diversification_alpha=div_alpha,
        cov_shrinkage=cov_shrinkage
    )

    used_tickers = [t for t, w in zip(info["tickers"], w_opt) if w > 1e-8]

    st.success("✅ Optimization completed successfully")

    st.markdown("### Portfolio diagnostics")
    st.write(f"📊 Universe size: **{len(inputs.tickers)}**")
    st.write(f"✅ Used in optimized portfolio: **{len(used_tickers)}**")
    st.write(f"⚠️ Zero-weighted in optimization: **{len(inputs.tickers) - len(used_tickers)}**")

    # -------------------
    # Optimized weights
    # -------------------
    st.subheader("Optimized weights")

    weights_df = pd.DataFrame({
        "Ticker": info["tickers"],
        "Weight": w_opt
    })
    weights_df = weights_df[weights_df["Weight"] > 1e-8].sort_values("Weight", ascending=False)
    st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))

    # Portfolio summary
    st.markdown("### Portfolio summary")
    summary_df = pd.DataFrame([{
        "Average annual return": info["avg_return_annual"],
        "Total return": (1 + info["avg_return_annual"]) ** ((pd.Timestamp(inputs.end) - pd.Timestamp(inputs.start)).days / 365.25) - 1,
        "Annual volatility": info["vol_annual"],
        "Sharpe ratio": info["sharpe"],
        "Average correlation": info["avg_correlation"]
    }])

    st.dataframe(summary_df.style.format({
        "Average annual return": "{:.2%}",
        "Total return": "{:.2%}",
        "Annual volatility": "{:.2%}",
        "Sharpe ratio": "{:.3f}",
        "Average correlation": "{:.2f}"
    }))


    # -------------------
    # Efficient frontier
    # -------------------
    if plot_frontier_flag:
        st.subheader("Efficient frontier")

        res = efficient_frontier(data, inputs, n_points=40, cov_shrinkage=cov_shrinkage)
        fig2 = plot_frontier(res, color_by=frontier_color_by, scale_size_by_assets=scale_marker_size)
        st.pyplot(fig2)

        frontier_df = build_frontier_table(res)

        # Add Top Holdings column
        top_holdings = []
        for tickers_w in res["weights"]:
            tickers_list, weights_list = tickers_w
            if len(weights_list) == 0:
                top_holdings.append("")
                continue
            sorted_pairs = sorted(zip(tickers_list, weights_list), key=lambda x: x[1], reverse=True)
            top3 = [f"{t} ({w:.1%})" for t, w in sorted_pairs[:3]]
            top_holdings.append(", ".join(top3))
        frontier_df["Top Holdings"] = top_holdings

        # Highlight fallback rows
        def highlight_fallback(row):
            return ['background-color: #fff3cd' if row["Fallback Used"] else '' for _ in row]

        st.dataframe(
            frontier_df.style.format({
                "Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe": "{:.3f}",
                "Avg Corr": "{:.2f}",
                "Num Assets": "{:d}"
            }).apply(highlight_fallback, axis=1),
            width='stretch'
        )


    # -------------------
    # Correlation heatmap
    # -------------------
    if plot_corr_flag:
        st.subheader("Correlation heatmap")
        # If many pass threshold, warn we’re showing top-N only
        num_above = int(np.sum(w_opt >= min_weight_heatmap))
        if num_above > 50:
            st.info(f"ℹ️ {num_above} tickers meet the heatmap threshold; showing top 50 by weight.")

        fig_corr = plot_correlation_heatmap(
            data.returns_daily,
            info["tickers"],
            w_opt,
            min_weight=min_weight_heatmap,
            max_assets=50
        )
        st.pyplot(fig_corr)


