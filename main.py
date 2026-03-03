"""
Matka Analyzer Pro - Main Entry Point
-------------------------------------
Orchestrates the full analytical workflow, including:
1. Data loading and normalization
2. Pattern engine runs (Frequency, Cycles, Digits, Momentum, Entropy)
3. Confidence scoring and ranking
4. Statistical validation (Entropy, Chi-Square, Autocorrelation, Streaks)
5. Monte Carlo simulations for baseline comparison
6. Machine Learning walk-forward validation
"""
import pandas as pd
import numpy as np
from engines.ml_predictor import MLPredictor
from engines.statistical_analysis import StatisticalAnalyzer
from engines.monte_carlo import MonteCarloSimulator
from engines.frequency import FrequencyEngine
from engines.cycles import CycleEngine
from engines.digits import DigitEngine
from engines.momentum import MomentumEngine
from engines.entropy import EntropyEngine
from scoring.confidence import ConfidenceEngine
from data.data_loader import DataLoader
from config import DATA_FILE, SCHEMA_FILE, DISCLAIMER, TOP_N_PREDICTIONS


def run_engines(df):
    """
    Orchestrates and runs all analytical engines on the provided DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The validated and normalized DataFrame containing 'Date' and 'Jodi'.

    Returns
    -------
    dict
        A dictionary containing results from all engines (confidence, frequency,
        cycles, digits, momentum, entropy, etc.).
    """
    # Initialize engines
    fe = FrequencyEngine()
    ce = CycleEngine()
    de = DigitEngine()
    me = MomentumEngine()
    ee = EntropyEngine()
    conf = ConfidenceEngine()

    # Run engines
    frequency = fe.run(df)
    cycles = ce.run(df)
    digits_output = de.run(df)
    digits = digits_output.get("jodi_scores", {})
    individual_digit_strength = digits_output.get("individual_digit_strength", {})
    momentum = me.run(df)
    entropy = ee.run(df)

    # Run confidence scoring
    confidence = conf.run(
        frequency,
        cycles,
        digits,
        momentum,
        sample_size=len(df),
        top_n=TOP_N_PREDICTIONS
    )

    return {
        "confidence": confidence,
        "frequency": frequency,
        "cycles": cycles,
        "digits": digits,
        "individual_digit_strength": individual_digit_strength,
        "momentum": momentum,
        "entropy": entropy
    }


def main():
    """
    Main function to run the full analysis and validation workflow.
    """
    print("=" * 60)
    print("MATKA ANALYZER PRO: HISTORICAL PATTERN ANALYSIS")
    print("=" * 60)
    print(DISCLAIMER)
    print("-" * 60)

    try:
        # 1️⃣ Load Data
        print(f"Loading data from '{DATA_FILE}'...")
        loader = DataLoader(DATA_FILE, SCHEMA_FILE)
        df = loader.load_data()
        print(f"Loaded and normalized {len(df)} records.")
        print(f"Data range: {df['Date'].min().date()} to {df['Date'].max().date()}")

        # 2️⃣ Run Analytical Engines
        print("\n--- Running Pattern Analysis Engines ---")
        results = run_engines(df)
        
        # Display Top Confidence Results
        print("\nTOP RANKED CONFIDENCE ALIGNMENTS:")
        print(f"{'Jodi':<6} | {'Score':<8} | {'Tags'}")
        print("-" * 40)
        for jodi, score, tags in results["confidence"]:
            tags_str = ", ".join(tags)
            print(f"{jodi:<6} | {score:<8.2f} | {tags_str}")

        # 3️⃣ Statistical Analysis
        print("\n--- Performing Statistical Validation ---")
        analyzer = StatisticalAnalyzer(df, target_col='Jodi')

        entropy_val, max_entropy = analyzer.shannon_entropy()
        print(f"Shannon Entropy: {entropy_val:.4f} (Max: {max_entropy:.4f})")

        chi_stat, p_value = analyzer.chi_square_test()
        print(f"Chi-Square Stat: {chi_stat:.4f} (p-value: {p_value:.6f})")

        auto = analyzer.autocorrelation(lag=1)
        print(f"Autocorrelation (Lag 1): {auto:.6f}")

        streak = analyzer.longest_streak()
        print(f"Longest Repeating Streak: {streak}")

        # 4️⃣ Monte Carlo Simulation
        print("\n--- Running Monte Carlo Simulation (Baseline Comparison) ---")
        mc = MonteCarloSimulator(df["Jodi"], simulations=1000) # Use 1000 for faster CLI run
        simulation_results = mc.run()
        real_results = mc.real_metrics()
        
        # Percentiles
        mc.calculate_percentiles(simulation_results, real_results)

        # --- Bias Continuation Test ---
        events, continued, rate = mc.bias_continuation_test()

        print("\n--- Bias Continuation Test ---")
        print("Total Bias Events:", events)
        print("Continuation Events:", continued)
        print("Continuation Rate:", round(rate * 100, 2), "%")
        print("Baseline Probability (10 draws):", round(1 - (0.99 ** 10), 4) * 100, "%")

        # --- Mean Reversion Test ---
        events_mr, actual_rate, baseline_rate = mc.mean_reversion_test()

        print("\n--- Mean Reversion Test ---")
        print("Total Bias Events:", events_mr)
        print("Actual Hit Rate After Bias:", round(actual_rate * 100, 4), "%")
        print("Baseline Hit Rate:", round(baseline_rate * 100, 4), "%")
        print("Difference:", round((actual_rate - baseline_rate) * 100, 4), "%")

        # --- Deep-Dive Analysis for Top Candidates ---
        from analyzer import analyze_jodi
        from plotting import generate_report
        
        print("\n--- Automated Deep-Dive (Top Candidate) ---")
        top_jodi = results["confidence"][0][0] if results["confidence"] else "84"
        deep_dive = analyze_jodi(df, top_jodi)
        generate_report(deep_dive)
        
        print(f"Candidate: {deep_dive['jodi']}")
        print(f"Z-Score: {deep_dive['z_score']:.2f}")
        print(f"Status: Report generated in 'reports/{deep_dive['jodi']}_summary.txt'")

        # --- Automation: Analyze Today's Result ---
        print("\n--- Automation: Latest Result Analysis ---")
        latest_record = df.iloc[-1]
        latest_date = latest_record['Date'].date()
        latest_jodi = latest_record['Jodi']
        print(f"Latest Result: {latest_jodi} ({latest_date})")
        
        today_dive = analyze_jodi(df, latest_jodi)
        generate_report(today_dive)
        print(f"Analysis for {latest_jodi} saved to reports/")

        # 5️⃣ ML Validation (Optional - can be slow)
        # print("\n--- Running ML Walk-Forward Validation ---")
        # predictor = MLPredictor()
        # predictor.walk_forward_validation(df)

    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
