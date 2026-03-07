"""
Advanced Statistical Analysis Runner
------------------------------------
Executes the newly implemented statistical engines (Markov, Bayesian, Regime)
and generates a consolidated report of identified signals.
"""
from data.data_loader import DataLoader
from config import DATA_FILE, SCHEMA_FILE
from engines.transition_matrix import TransitionEngine
from engines.bayesian_predictor import BayesianEngine
from engines.regime_detector import RegimeDetector
import pandas as pd

def run_advanced_report():
    print("=" * 60)
    print("MATKA ANALYZER PRO: ADVANCED STATISTICAL RESEARCH")
    print("=" * 60)
    
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    # 1. Regime Detection
    print("\n[1] Regime Detection")
    rd = RegimeDetector()
    current_regime = rd.detect_regime(df)
    regime_results = rd.quick_backtest(df)
    print(f"Current Market Regime: {current_regime}")
    print(f"Historical Regime Distribution: {regime_results['regime_distribution']}")
    
    # 2. Transition Matrix (Markov)
    print("\n[2] Sequential Patterns (Markov Transitions)")
    te = TransitionEngine()
    te.train(df)
    top_trans = te.get_top_transitions(top_n=5)
    print("Strongest Historical Transitions (A -> B):")
    for from_s, to_s, prob in top_trans:
        print(f"  {from_s:02} -> {to_s:02} (Prob: {prob:.4f})")
    
    trans_backtest = te.quick_backtest(df, top_k=10)
    print(f"Markov (Top-10) Hit Rate: {trans_backtest['accuracy']:.2%} (Baseline: {trans_backtest['baseline']:.2%})")
    
    # 3. Bayesian Probabilities
    print("\n[3] Bayesian Updates")
    be = BayesianEngine()
    bayesian_backtest = be.quick_backtest(df, top_k=10)
    print(f"Bayesian (Top-10) Hit Rate: {bayesian_backtest['accuracy']:.2%} (Baseline: {bayesian_backtest['baseline']:.2%})")
    
    # Conclusion Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF SIGNALS")
    print("-" * 60)
    
    edge_m = trans_backtest['edge']
    edge_b = bayesian_backtest['edge']
    
    if edge_m > 0.01 or edge_b > 0.01:
        print("SIGNAL FOUND: Statistically significant edge detected (>1%).")
        if edge_m > edge_b:
            print(f"Dominant Pattern: Sequential Dependencies (Markov, Edge: {edge_m:.2%})")
        else:
            print(f"Dominant Pattern: Bayesian Convergence (Frequency, Edge: {edge_b:.2%})")
    else:
        print("NO SIGNAL: Observed hit rates are near or below random baseline.")
        print("Conclusion: Current data shows no evidence of exploitable patterns.")
    
    print("=" * 60)

if __name__ == "__main__":
    run_advanced_report()
