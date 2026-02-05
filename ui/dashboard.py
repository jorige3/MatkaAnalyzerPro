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
from main import run_engines
from data.data_loader import DataLoader
from simulation.paper_test import PaperBacktest
from engines.entropy import EntropyEngine # New import
from config import DATA_FILE, SCHEMA_FILE, DISCLAIMER, MIN_HISTORY_DAYS, TOP_N_PREDICTIONS

# ... (rest of the file)

# New row for Entropy
st.header("Entropy Analysis")
entropy_data = results.get("entropy", {})
if entropy_data:
    st.metric(label="Overall Entropy Score", value=f"{entropy_data.get('overall_entropy_score', 0.0):.2f}")
else:
    st.info("No entropy data to display.")

# --- Historical Alignment Rate (Backtesting) ---
st.header("Historical Alignment Rate (Backtesting)")
st.info("Running backtest simulation. This may take a moment...")
backtester = PaperBacktest(data_path, min_history_days=min_history_days)
backtest_stats = backtester.run(top_n=top_n)

if backtest_stats:
    st.write(f"**Total Days Tested**: {backtest_stats['total_days_tested']}")
    st.write(f"**Hits (Top-{backtest_stats['top_n_considered']})**: {backtest_stats['hits']}")
    st.write(f"**Misses**: {backtest_stats['misses']}")
    st.success(f"**Historical Alignment Rate**: {backtest_stats['historical_alignment_rate']}%")
else:
    st.warning("Could not run backtest simulation.")

st.markdown("---")
st.markdown(f"**Note**: All outputs are historical observations only. {DISCLAIMER}")