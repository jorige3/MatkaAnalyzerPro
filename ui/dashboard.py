"""
Streamlit Dashboard for Matka Analyzer Pro.

This dashboard provides interactive visualizations and summaries of the
historical Matka data analysis, including frequency, cycle distribution,
digit strength, momentum, and backtesting results.

It is designed for educational and analytical purposes only, focusing on
pattern recognition in historical time-series data. It explicitly avoids
any form of gambling advice or prediction.
"""

import sys
import os

# Get the absolute path of the directory containing dashboard.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (matka_analyzer/)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

import streamlit as st
import pandas as pd
import plotly.express as px
from main import run_engines
from data.data_loader import DataLoader
from simulation.paper_test import PaperBacktest
from config import DATA_FILE, SCHEMA_FILE, DISCLAIMER, MIN_HISTORY_DAYS, TOP_N_PREDICTIONS

st.set_page_config(layout="wide", page_title="Matka Analyzer Pro Dashboard")

st.title("Matka Analyzer Pro: Historical Pattern Analysis")

# Global disclaimer at the top
st.markdown(f"**{DISCLAIMER}**")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
selected_data_file = st.sidebar.selectbox("Select Data File", [DATA_FILE])
min_history_days_config = st.sidebar.slider("Minimum History Days for Backtest", 10, 100, MIN_HISTORY_DAYS)
top_n_predictions_config = st.sidebar.slider("Top N Predictions for Display & Backtest", 1, 20, TOP_N_PREDICTIONS)

# Generate report text for download
def generate_report_text(df, results):
    latest_date = df['Date'].max().strftime('%Y-%m-%d')
    report = f"Matka Analyzer Pro: Analysis Report for {latest_date}\n"
    report += "=" * 50 + "\n\n"
    report += "TOP CONFIDENCE ALIGNMENTS:\n"
    report += f"{'Jodi':<6} | {'Score':<8} | {'Tags'}\n"
    report += "-" * 50 + "\n"
    for jodi, score, tags in results.get("confidence", []):
        report += f"{jodi:<6} | {score:<8.2f} | {', '.join(tags)}\n"
    return report

# --- Data Loading and Engine Run ---
@st.cache_data
def load_and_run_analysis(data_file, schema_file, min_hist_days_param):
    data_loader = DataLoader(file_path=data_file, schema_path=schema_file)
    df = data_loader.load_data()
    # Ensure enough data for analysis
    if len(df) < min_hist_days_param:
        st.error(f"Not enough historical data. Need at least {min_hist_days_param} days.")
        st.stop()
    results = run_engines(df)
    return df, results

# Use selected values from sidebar
df, results = load_and_run_analysis(selected_data_file, SCHEMA_FILE, min_history_days_config)

report_text = generate_report_text(df, results)
st.sidebar.download_button(
    label="Download Analysis Report (.txt)",
    data=report_text,
    file_name=f"analysis_{df['Date'].max().strftime('%Y%m%d')}.txt",
    mime="text/plain"
)

st.subheader("Data Overview")
st.write(f"Loaded {len(df)} records from `{selected_data_file}`.")
st.write(f"Analysis performed up to `{df['Date'].max().strftime('%Y-%m-%d')}`.")

# --- Display Top Confidence Alignments ---
st.header("Top Confidence Alignments")
confidence_results = results.get("confidence", [])
if confidence_results:
    confidence_df = pd.DataFrame(confidence_results, columns=["Jodi", "Confidence Score", "Tags"])
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(confidence_df.head(top_n_predictions_config))
    with col2:
        fig_conf = px.bar(
            confidence_df.head(top_n_predictions_config),
            x="Jodi",
            y="Confidence Score",
            title=f"Top {top_n_predictions_config} Jodis by Confidence",
            hover_data=["Tags"]
        )
        st.plotly_chart(fig_conf, width='stretch')
else:
    st.info("No confidence alignments to display.")

# --- Detailed Analysis Section ---
st.header("Detailed Engine Analysis")
with st.expander("Expand to view detailed engine outputs"):
    # --- Frequency Chart ---
    st.subheader("Frequency Analysis")
    frequency_data = results.get("frequency", {})
    if frequency_data:
        freq_df = pd.DataFrame(frequency_data.items(), columns=["Jodi", "Frequency Score"])
        freq_df = freq_df.sort_values("Frequency Score", ascending=False).head(top_n_predictions_config)
        fig_freq = px.bar(freq_df, x="Jodi", y="Frequency Score", title=f"Top {top_n_predictions_config} Jodis by Frequency")
        st.plotly_chart(fig_freq, width='stretch')
    else:
        st.info("No frequency data to display.")

    # --- Cycle Distribution ---
    st.subheader("Cycle Analysis")
    cycle_data = results.get("cycles", {})
    if cycle_data:
        cycle_status_counts = pd.DataFrame([v["status"] for v in cycle_data.values()], columns=["Status"])
        fig_cycle = px.pie(cycle_status_counts, names="Status", title="Cycle Status Distribution")
        st.plotly_chart(fig_cycle, width='stretch')

        due_jodis = [jodi for jodi, data in cycle_data.items() if data["status"] == "DUE"]
        exhausted_jodis = [jodi for jodi, data in cycle_data.items() if data["status"] == "EXHAUSTED"]

        if due_jodis:
            st.write(f"**Jodis in DUE Cycles**: {', '.join(due_jodis)}")
        if exhausted_jodis:
            st.write(f"**Jodis in EXHAUSTED Cycles**: {', '.join(exhausted_jodis)}")
        if not due_jodis and not exhausted_jodis:
            st.info("No Jodis currently in DUE or EXHAUSTED cycles.")
    else:
        st.info("No cycle data to display.")

    # --- Digit Strength ---
    st.subheader("Digit Strength Analysis")
    digit_results = results.get("digits", {})
    if digit_results:
        digit_jodi_scores_df = pd.DataFrame(
            [{'Jodi': k, 'Digit Score': v['digit_score'], 'Tens Digit': v['tens_digit'], 'Unit Digit': v['unit_digit']}
             for k, v in digit_results.items()]
        )
        digit_jodi_scores_df = digit_jodi_scores_df.sort_values("Digit Score", ascending=False).head(top_n_predictions_config)
        st.dataframe(digit_jodi_scores_df)

        # Individual digit strength visualization
        individual_digit_strength = results.get("individual_digit_strength", {})
        if individual_digit_strength:
            digit_strength_df = pd.DataFrame(individual_digit_strength.items(), columns=["Digit", "Strength Score"])
            digit_strength_df = digit_strength_df.sort_values("Strength Score", ascending=False)
            fig_digit_strength = px.bar(digit_strength_df, x="Digit", y="Strength Score", title="Individual Digit Strength")
            st.plotly_chart(fig_digit_strength, width='stretch')
        else:
            st.info("No individual digit strength data to display.")

    else:
        st.info("No digit analysis data to display.")

    # --- Momentum Analysis ---
    st.subheader("Momentum Analysis")
    momentum_data = results.get("momentum", {})
    if momentum_data:
        momentum_df = pd.DataFrame(momentum_data.items(), columns=["Jodi", "Momentum Score"])
        momentum_df = momentum_df.sort_values("Momentum Score", ascending=False).head(top_n_predictions_config)
        fig_momentum = px.bar(momentum_df, x="Jodi", y="Momentum Score", title=f"Top {top_n_predictions_config} Jodis by Momentum")
        st.plotly_chart(fig_momentum, width='stretch')
    else:
        st.info("No momentum data to display.")

    # --- Entropy Score ---
    st.subheader("Entropy Score")
    entropy_score = results.get("entropy", {}).get("overall_entropy_score")
    if entropy_score is not None:
        st.write(f"**Overall Entropy Score**: {entropy_score:.2f}")
    else:
        st.info("No entropy data to display.")


@st.cache_data
def run_backtest(data_file, min_history_days, top_n_predictions):
    backtester = PaperBacktest(data_file, min_history_days=min_history_days)
    return backtester.run(top_n=top_n_predictions, verbose=True)

# --- Historical Alignment Rate (Backtesting) ---
st.header("Historical Alignment Rate (Backtesting)")
st.info("Running backtest simulation. This may take a moment. The displayed 'min_history_days' is for this backtest only.")

backtest_stats = run_backtest(selected_data_file, min_history_days_config, top_n_predictions_config)

if backtest_stats:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Days Tested", backtest_stats['total_days_tested'])
    with col2:
        st.metric(f"Hits (Top-{backtest_stats['top_n_considered']})", backtest_stats['hits'])
    with col3:
        st.metric("Misses", backtest_stats['misses'])
    
    st.success(f"**Historical Alignment Rate**: {backtest_stats['historical_alignment_rate']}%")

    if st.checkbox("Show detailed daily backtest results"):
        if "daily_results" in backtest_stats and backtest_stats["daily_results"]:
            daily_bt_df = pd.DataFrame(backtest_stats["daily_results"])
            # Convert date column for better display if needed
            daily_bt_df['date'] = daily_bt_df['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(daily_bt_df, width='stretch')
        else:
            st.info("No detailed daily results available. Run backtest with verbose=True.")
else:
    st.warning("Could not run backtest simulation.")

st.markdown("---")
# Global disclaimer at the bottom
st.markdown(f"**Note**: All outputs are historical observations only. {DISCLAIMER}")
