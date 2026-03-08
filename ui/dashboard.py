"""
Streamlit Dashboard for Matka Analyzer Pro
------------------------------------------
Interactive visualization of historical pattern analysis.
"""

import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure parent directory is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from main import run_classic_engines
from data.data_loader import DataLoader
from simulation.paper_test import PaperBacktest
from engines.ultimate_ensemble import UltimateEnsemble
from config import DATA_FILE, SCHEMA_FILE, DISCLAIMER, MIN_HISTORY_DAYS, TOP_N_PREDICTIONS

st.set_page_config(layout="wide", page_title="Matka Analyzer Pro Dashboard", page_icon="📊")

st.title("📊 Matka Analyzer Pro: Historical Pattern Analysis")
st.warning(f"**Disclaimer**: {DISCLAIMER}")

# --- Sidebar ---
st.sidebar.header("🔧 Configuration")
min_history = st.sidebar.slider("Min History Days", 30, 365, MIN_HISTORY_DAYS)
top_k = st.sidebar.slider("Top Predictions (K)", 1, 20, TOP_N_PREDICTIONS)

@st.cache_data
def load_data():
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    return loader.load_data()

df = load_data()
st.sidebar.write(f"**Dataset**: {len(df)} records")
st.sidebar.write(f"**Range**: {df['Date'].min().date()} to {df['Date'].max().date()}")

# --- Main Dashboard ---
tab1, tab2, tab3 = st.tabs(["🎯 Current Analysis", "📈 Historical Performance", "🧠 Advanced Ensemble"])

with tab1:
    st.header("Unified Confidence Scores")
    results = run_classic_engines(df)
    
    conf_df = pd.DataFrame(results["confidence"], columns=["Jodi", "Score", "Tags"])
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(conf_df.style.background_gradient(subset=["Score"], cmap="Greens"))
    
    with col2:
        fig = px.bar(conf_df.head(top_k), x="Jodi", y="Score", color="Score",
                     title=f"Top {top_k} Candidates by Confidence Score",
                     hover_data=["Tags"])
        st.plotly_chart(fig, use_container_width=True)

    # Breakdown Expander
    with st.expander("🔍 Detailed Engine Breakdown"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.subheader("Frequency Score")
            freq_df = pd.DataFrame(list(results["frequency"].items()), columns=["Jodi", "Freq"]).sort_values("Freq", ascending=False)
            st.bar_chart(freq_df.set_index("Jodi").head(10))
            
        with c2:
            st.subheader("Momentum (Acceleration)")
            mom_df = pd.DataFrame(list(results["momentum"].items()), columns=["Jodi", "Mom"]).sort_values("Mom", ascending=False)
            st.bar_chart(mom_df.set_index("Jodi").head(10))
            
        with c3:
            st.subheader("Digit Strength")
            digit_strength = results["digits"]
            ds_df = pd.DataFrame(list(digit_strength.items()), columns=["Digit", "Strength"]).sort_values("Digit")
            fig_ds = px.line(ds_df, x="Digit", y="Strength", markers=True)
            st.plotly_chart(fig_ds, use_container_width=True)

with tab2:
    st.header("Backtest Simulation (Walk-Forward)")
    if st.button("🚀 Run Backtest"):
        with st.spinner("Analyzing historical accuracy..."):
            bt = PaperBacktest(DATA_FILE, min_history_days=min_history)
            bt_res = bt.run(top_n=top_k)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Hits", bt_res["hits"])
            m2.metric("Hit Rate", f"{bt_res['historical_alignment_rate']}%")
            m3.metric("Baseline", f"{bt_res['baseline_rate']}%")
            m4.metric("Edge", f"{bt_res['edge_over_baseline']}%", delta=bt_res['edge_over_baseline'])
            
            st.success(f"Historical Alignment is {bt_res['edge_over_baseline']}% higher than random baseline.")

with tab3:
    st.header("Advanced Regime Ensemble")
    st.info("Utilizes Hidden Markov Models (HMM) to detect latent 'regimes' (Stable, Sequential, Volatile).")
    
    if st.button("🧠 Execute Ensemble Inference"):
        ue = UltimateEnsemble(df)
        ue._train_all()
        preds, regime = ue.predict_next(top_n=10)
        
        regimes = {0: "🟢 STABLE (High Predictability)", 
                   1: "🟡 SEQUENTIAL (Lagged Dependencies)", 
                   2: "🔴 VOLATILE (Mean Reversion Dominant)"}
        
        st.subheader(f"Current Market State: {regimes.get(regime)}")
        
        pred_df = pd.DataFrame(preds, columns=["Jodi", "Probability Score"])
        st.table(pred_df)
        
        # Convolution Probability Distribution
        fig_prob = px.area(pred_df, x="Jodi", y="Probability Score", title="Ensemble Probability Curve")
        st.plotly_chart(fig_prob, use_container_width=True)

st.divider()
st.caption(f"Matka Analyzer Pro | Version 1.0.0 | {DISCLAIMER}")
