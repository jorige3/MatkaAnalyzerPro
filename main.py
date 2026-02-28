import pandas as pd
import numpy as np
from engines.ml_predictor import MLPredictor
from engines.statistical_analysis import StatisticalAnalyzer
from engines.monte_carlo import MonteCarloSimulator
from config import DATA_FILE, DISCLAIMER



def main():
    """
    Main function to run the ML validation and statistical analysis workflow.
    """
    print(DISCLAIMER)
    print("\nStarting Matka Analyzer Statistical Validation...")

    try:
        # 1️⃣ Load Data
        print(f"Loading data from '{DATA_FILE}'...")
        df = pd.read_csv(DATA_FILE)

        if df.empty:
            print("Error: Data file is empty. Cannot proceed.")
            return

        df.columns = df.columns.str.lower()
        print(f"Loaded {len(df)} records.")

        # 2️⃣ Statistical Analysis
        analyzer = StatisticalAnalyzer(df)

        entropy, max_entropy = analyzer.shannon_entropy()
        print("\n--- Entropy Analysis ---")
        print(f"Entropy: {entropy:.4f}")
        print(f"Maximum Possible Entropy: {max_entropy:.4f}")

        chi_stat, p_value = analyzer.chi_square_test()
        print("\n--- Chi-Square Test ---")
        print(f"Chi-Square Statistic: {chi_stat:.4f}")
        print(f"P-Value: {p_value:.6f}")

        auto = analyzer.autocorrelation(lag=1)
        print("\n--- Autocorrelation (Lag 1) ---")
        print(f"Autocorrelation: {auto:.6f}")

        streak = analyzer.longest_streak()
        print("\n--- Longest Streak ---")
        print(f"Longest Repeating Streak: {streak}")

        

        print("\n--- Monte Carlo Simulation ---")
        mc = MonteCarloSimulator(df["jodi"], simulations=300)

        simulation_results = mc.run()
        real_results = mc.real_metrics()

        print("\nReal Data Metrics:")
        for k, v in real_results.items():
            print(f"{k}: {v}")

            print("\nSimulation Averages:")
            print(f"Average Streak: {np.mean(simulation_results['streaks']):.2f}")
            print(f"Average Entropy: {np.mean(simulation_results['entropies']):.4f}")
            print(f"Average Max Frequency: {np.mean(simulation_results['max_freqs']):.2f}")
 
            mc.plot_results(simulation_results, real_results)
        # 3️⃣ ML Validation
        predictor = MLPredictor()
        predictor.walk_forward_validation(df)
        predictor.shuffle_test(df)

    except FileNotFoundError:
        print(f"FATAL ERROR: The data file '{DATA_FILE}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()