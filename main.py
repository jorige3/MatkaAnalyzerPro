"""
Matka Analyzer Pro - Main Entry Point
-------------------------------------
Orchestrates the full analytical workflow, including:
1. Data loading and normalization
2. Classic Pattern engine runs (Frequency, Cycles, Digits, Momentum)
3. Advanced Ensemble analysis (HMM regimes, Convolutional probabilities)
4. Confidence scoring and ranking
5. Statistical validation and Monte Carlo simulation
"""

import sys
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner CLI output
warnings.filterwarnings("ignore")

from engines.frequency import FrequencyEngine
from engines.cycles import CycleEngine
from engines.digits import DigitEngine
from engines.momentum import MomentumEngine
from engines.entropy import EntropyEngine
from engines.ultimate_ensemble import UltimateEnsemble
from engines.statistical_analysis import StatisticalAnalyzer
from engines.monte_carlo import MonteCarloSimulator
from scoring.confidence import ConfidenceEngine
from data.data_loader import DataLoader
from config import DATA_FILE, SCHEMA_FILE, DISCLAIMER, TOP_N_PREDICTIONS


def run_classic_engines(df: pd.DataFrame):
    """
    Runs the standard pattern engines and confidence scoring.
    """
    fe = FrequencyEngine()
    ce = CycleEngine()
    de = DigitEngine()
    me = MomentumEngine()
    conf = ConfidenceEngine()

    # Engine Executions
    frequency = fe.run(df)
    cycles = ce.run(df)
    digits_out = de.run(df)
    momentum = me.run(df)

    # Scoring
    confidence = conf.run(
        frequency, cycles, digits_out.get("jodi_scores", {}), momentum,
        sample_size=len(df), top_n=TOP_N_PREDICTIONS
    )

    return {
        "confidence": confidence,
        "frequency": frequency,
        "cycles": cycles,
        "digits": digits_out.get("individual_digit_strength", {}),
        "momentum": momentum
    }


def display_results(results: dict):
    """
    Prints the top ranked results in a clean format.
    """
    print("\n" + "="*50)
    print("TOP RANKED CONFIDENCE ALIGNMENTS")
    print("="*50)
    print(f"{'Rank':<4} | {'Jodi':<6} | {'Score':<8} | {'Tags'}")
    print("-" * 50)
    
    for i, (jodi, score, tags) in enumerate(results["confidence"], 1):
        tags_str = ", ".join(tags)
        print(f"{i:<4} | {jodi:<6} | {score:<8.2f} | {tags_str}")


def main():
    """
    Main orchestration loop.
    """
    print("\n" + "#" * 60)
    print(" MATKA ANALYZER PRO: HISTORICAL SIGNAL SCANNER ".center(60, "#"))
    print("#" * 60)
    print(f"\n{DISCLAIMER}")
    print("-" * 60)

    try:
        # 1. DATA LOADING
        loader = DataLoader(DATA_FILE, SCHEMA_FILE)
        df = loader.load_data()
        print(f"[*] Data Loaded: {len(df)} records ({df['Date'].min().date()} to {df['Date'].max().date()})")

        # 2. CLASSIC ANALYSIS
        print("\n[*] Initializing Classic Pattern Scan...")
        classic_results = run_classic_engines(df)
        display_results(classic_results)

        # 3. ADVANCED ENSEMBLE (Optional/Integrated)
        print("\n[*] Running Ultimate Ensemble (HMM + Convolution)...")
        ue = UltimateEnsemble(df)
        ue._train_all()
        ensemble_preds, current_regime = ue.predict_next(top_n=5)
        
        regime_names = {0: "STABLE", 1: "SEQUENTIAL", 2: "VOLATILE"}
        print(f"[*] Detected Latent Regime: {regime_names.get(current_regime, 'UNKNOWN')}")
        print(f"[*] Top Ensemble Candidates: {', '.join([p[0] for p in ensemble_preds])}")

        # 4. STATISTICAL VALIDATION
        print("\n" + "="*50)
        print("STATISTICAL VALIDATION SUMMARY")
        print("="*50)
        sa = StatisticalAnalyzer(df, target_col='Jodi')
        entropy, max_e = sa.shannon_entropy()
        chi, p_val = sa.chi_square_test()
        
        print(f"Shannon Entropy: {entropy:.4f} / {max_e:.4f} (Predictability Score)")
        print(f"Chi-Square P-Value: {p_val:.6f} (Bias Significance)")
        
        # 5. MONTE CARLO BASELINE
        print("\n[*] Running Monte Carlo Simulation (1,000 trials)...")
        mc = MonteCarloSimulator(df["Jodi"], simulations=1000)
        mc_results = mc.run()
        mc.calculate_percentiles(mc_results, mc.real_metrics())
        
        events_mr, actual_rate, baseline_rate = mc.mean_reversion_test()
        print(f"[*] Mean Reversion Delta: {((actual_rate - baseline_rate)*100):+.2f}% vs Random Baseline")

        print("\n" + "=" * 60)
        print(" ANALYSIS COMPLETE ".center(60, "="))
        print("=" * 60)

    except Exception as e:
        print(f"\n[!] FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
