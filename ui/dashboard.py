"""
Streamlit Dashboard for Matka Analyzer Pro.

This dashboard provides interactive visualizations and summaries of the
historical Matka data analysis, including frequency, cycle distribution,
digit strength, momentum, and backtesting results.

It is designed for educational and analytical purposes only, focusing on
pattern recognition in historical time-series data. It explicitly avoids
any form of gambling advice or prediction.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from main import run_engines
from data.data_loader import DataLoader
from simulation.paper_test import PaperBacktest
from config import DATA_FILE, SCHEMA_FILE, DISCLAIMER, MIN_HISTORY_DAYS, TOP_N_PREDICTIONS

st.set_page_config(layout="wide", page_title="Matka Analyzer Pro Dashboard")

st.title("Matka Analyzer Pro: Historical Pattern Analysis")

st.markdown(f"*{DISCLAIMER}*")

st.sidebar.header("Configuration")
selected_data_file = st.sidebar.selectbox("Select Data File", [DATA_FILE])
min_history_days = st.sidebar.slider("Minimum History Days for Backtest", 10, 100, MIN_HISTORY_DAYS)
top_n_predictions = st.sidebar.slider("Top N Predictions for Backtest", 1, 20, TOP_N_PREDICTIONS)

# --- Data Loading and Engine Run ---
@st.cache_data
def load_and_run_analysis(data_file, schema_file, min_hist_days):
    data_loader = DataLoader(file_path=data_file, schema_path=schema_file)
    df = data_loader.load_data()
    # Ensure enough data for analysis
    if len(df) < min_hist_days:
        st.error(f"Not enough historical data. Need at least {min_hist_days} days.")
        st.stop()
    results = run_engines(df)
    return df, results

df, results = load_and_run_analysis(selected_data_file, SCHEMA_FILE, min_history_days)

st.subheader("Current Analysis Snapshot")
st.write(f"Loaded {len(df)} records from `{selected_data_file}`.")
st.write(f"Analysis performed up to `{df['Date'].max().strftime('%Y-%m-%d')}`.")

# --- Display Top Confidence Alignments ---
st.header("Top Confidence Alignments")
confidence_results = results.get("confidence", [])
if confidence_results:
    confidence_df = pd.DataFrame(confidence_results, columns=["Jodi", "Confidence Score", "Tags"])
    st.dataframe(confidence_df.head(top_n_predictions))
else:
    st.info("No confidence alignments to display.")

# --- Frequency Chart ---
st.header("Frequency Analysis")
frequency_data = results.get("frequency", {})
if frequency_data:
    freq_df = pd.DataFrame(frequency_data.items(), columns=["Jodi", "Frequency Score"])
    freq_df = freq_df.sort_values("Frequency Score", ascending=False).head(top_n_predictions)
    fig_freq = px.bar(freq_df, x="Jodi", y="Frequency Score", title=f"Top {top_n_predictions} Jodis by Frequency")
    st.plotly_chart(fig_freq, use_container_width=True)
else:
    st.info("No frequency data to display.")

# --- Cycle Distribution ---
st.header("Cycle Analysis")
cycle_data = results.get("cycles", {})
if cycle_data:
    cycle_status_counts = pd.DataFrame([v["status"] for v in cycle_data.values()], columns=["Status"])
    fig_cycle = px.pie(cycle_status_counts, names="Status", title="Cycle Status Distribution")
    st.plotly_chart(fig_cycle, use_container_width=True)

    st.subheader("Jodis in DUE/EXHAUSTED Cycles")
    due_jodis = [jodi for jodi, data in cycle_data.items() if data["status"] == "DUE"]
    exhausted_jodis = [jodi for jodi, data in cycle_data.items() if data["status"] == "EXHAUSTED"]

    if due_jodis:
        st.write(f"**DUE Cycles**: {', '.join(due_jodis)}")
    if exhausted_jodis:
        st.write(f"**EXHAUSTED Cycles**: {', '.join(exhausted_jodis)}")
    if not due_jodis and not exhausted_jodis:
        st.info("No Jodis currently in DUE or EXHAUSTED cycles.")
else:
    st.info("No cycle data to display.")

# --- Historical Accuracy (Backtesting) ---
st.header("Historical Alignment Rate (Backtesting)")
st.info("Running backtest simulation. This may take a moment...")
backtester = PaperBacktest(selected_data_file, min_history_days=min_history_days)
backtest_stats = backtester.run(top_n=top_n_predictions, verbose=True) # Set verbose to True for detailed results

if backtest_stats:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Days Tested", backtest_stats['total_days_tested'])
    col2.metric(f"Hits (Top-{backtest_stats['top_n_considered']})", backtest_stats['hits'])
    col3.metric("Misses", backtest_stats['misses'])
    st.success(f"**Historical Alignment Rate**: {backtest_stats['historical_alignment_rate']}%")

    if st.checkbox("Show detailed daily backtest results"):
        if "daily_results" in backtest_stats and backtest_stats["daily_results"]:
            daily_bt_df = pd.DataFrame(backtest_stats["daily_results"])
            st.dataframe(daily_bt_df)
        else:
            st.info("No detailed daily results available. Run backtest with verbose=True.")
else:
    st.warning("Could not run backtest simulation.")

st.markdown("---")
st.markdown(f"**Note**: All outputs are historical observations only. {DISCLAIMER}")
