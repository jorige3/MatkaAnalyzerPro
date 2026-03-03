import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


class MonteCarloSimulator:
    def __init__(self, real_series, simulations=10000):
        self.real_series = real_series.astype(int).values
        self.n = len(real_series)
        self.simulations = simulations

    def _longest_streak(self, series):
        max_streak = 1
        current_streak = 1

        for i in range(1, len(series)):
            if series[i] == series[i - 1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1

        return max_streak

    def _entropy(self, series):
        counts = Counter(series)
        probs = np.array(list(counts.values())) / len(series)
        return -np.sum(probs * np.log2(probs))

    def _max_frequency(self, series):
        counts = Counter(series)
        return max(counts.values())

    def run(self):
        """
        Run Monte Carlo simulations to generate baseline distributions.
        """
        streaks = []
        entropies = []
        max_freqs = []

        for _ in range(self.simulations):
            simulated = np.random.randint(0, 100, self.n)
            streaks.append(self._longest_streak(simulated))
            entropies.append(self._entropy(simulated))
            max_freqs.append(self._max_frequency(simulated))

        return {
            "streaks": streaks,
            "entropies": entropies,
            "max_freqs": max_freqs,
        }

    def frequency_z_scores(self):
        """
        Calculate Z-scores for each number (0-99) based on observed vs expected frequency.
        Expected frequency = n / 100.
        Std Dev = sqrt(expected * (1 - 1/100)).
        """
        expected = self.n / 100
        # Standard deviation for binomial distribution B(n, p) where p = 1/100
        std_dev = np.sqrt(expected * (1 - 1/100))
        
        counts = Counter(self.real_series)
        z_scores = {}
        for i in range(100):
            observed = counts.get(i, 0)
            z_scores[i] = (observed - expected) / std_dev
        return z_scores

    def detect_signals(self, threshold=2):
        """
        Identify numbers with statistically significant frequency deviations.
        """
        z_scores = self.frequency_z_scores()
        high_bias_numbers = [num for num, z in z_scores.items() if z > threshold]
        low_bias_numbers = [num for num, z in z_scores.items() if z < -threshold]
        
        return high_bias_numbers, low_bias_numbers

    def real_metrics(self):
        """
        Calculate metrics for the real data series and detect bias signals.
        """
        metrics = {
            "real_streak": self._longest_streak(self.real_series),
            "real_entropy": self._entropy(self.real_series),
            "real_max_freq": self._max_frequency(self.real_series),
        }

        # Bias Signal Detection
        high_bias, low_bias = self.detect_signals()
        print("\n--- Bias Signal Detection ---")
        print(f"High Bias Numbers: {high_bias}")
        print(f"Low Bias Numbers: {low_bias}")

        return metrics

    def calculate_percentiles(self, simulation_results, real_results):
        """
        Calculate how the real metrics compare to the simulated distribution.
        """
        percentiles = {}

        # Convert lists to numpy arrays
        sim_streaks = np.array(simulation_results["streaks"])
        sim_max_freqs = np.array(simulation_results["max_freqs"])
        sim_entropies = np.array(simulation_results["entropies"])

        # Streak percentile
        percentiles["streak_percentile"] = (
            np.sum(sim_streaks <= real_results["real_streak"]) / len(sim_streaks)
        ) * 100

        # Max frequency percentile
        percentiles["max_freq_percentile"] = (
            np.sum(sim_max_freqs <= real_results["real_max_freq"]) / len(sim_max_freqs)
        ) * 100

        # Entropy percentile
        percentiles["entropy_percentile"] = (
            np.sum(sim_entropies <= real_results["real_entropy"]) / len(sim_entropies)
        ) * 100

        # Print Monte Carlo Percentiles
        print("\n--- Monte Carlo Percentiles ---")
        print(f"Streak Percentile: {percentiles['streak_percentile']:.2f}%")
        print(f"Max Frequency Percentile: {percentiles['max_freq_percentile']:.2f}%")
        print(f"Entropy Percentile: {percentiles['entropy_percentile']:.2f}%")

        return percentiles

    def plot_results(self, simulation_results, real_results):
        """
        Visualize the simulated distributions against real data points.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1️⃣ Streak Distribution
        axes[0].hist(simulation_results["streaks"], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(real_results["real_streak"], color='red', linestyle="dashed", linewidth=2, label='Real')
        axes[0].set_title("Longest Streak Distribution")
        axes[0].set_xlabel("Streak Length")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        # 2️⃣ Max Frequency Distribution
        axes[1].hist(simulation_results["max_freqs"], bins=20, alpha=0.7, color='salmon', edgecolor='black')
        axes[1].axvline(real_results["real_max_freq"], color='red', linestyle="dashed", linewidth=2, label='Real')
        axes[1].set_title("Max Frequency Distribution")
        axes[1].set_xlabel("Max Frequency")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()

        # 3️⃣ Entropy Distribution
        axes[2].hist(simulation_results["entropies"], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[2].axvline(real_results["real_entropy"], color='red', linestyle="dashed", linewidth=2, label='Real')
        axes[2].set_title("Entropy Distribution")
        axes[2].set_xlabel("Entropy")
        axes[2].set_ylabel("Frequency")
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    def rolling_bias_detection(self, window=200, threshold=2, step=50):
        """
        Perform a rolling window analysis to detect statistical anomalies over time.
        """
        results = []

        for start in range(0, self.n - window + 1, step):
            end = start + window
            segment = self.real_series[start:end]

            counts = Counter(segment)
            expected = window / 100
            std_dev = np.sqrt(expected * (1 - 1/100))

            high_bias = []
            low_bias = []

            for num in range(100):
                observed = counts.get(num, 0)
                z = (observed - expected) / std_dev

                if z > threshold:
                    high_bias.append(num)
                elif z < -threshold:
                    low_bias.append(num)

            results.append({
                "start_index": start,
                "end_index": end,
                "high_bias": high_bias,
                "low_bias": low_bias
            })

        return results

    def bias_continuation_test(self, window=200, threshold=2, forward_look=10):
        """
        Tests how often numbers that show high bias in a window continue to
        appear in the subsequent days.
        """
        from collections import Counter
        
        continuation_events = 0
        total_bias_events = 0
        
        for i in range(window, self.n - forward_look):
            segment = self.real_series[i - window:i]
            counts = Counter(segment)
            
            expected = window / 100
            std_dev = np.sqrt(expected * (1 - 1/100))
            
            for num in range(100):
                observed = counts.get(num, 0)
                z = (observed - expected) / std_dev
                
                if z > threshold:
                    total_bias_events += 1
                    
                    future_segment = self.real_series[i:i + forward_look]
                    future_count = np.sum(future_segment == num)
                    
                    if future_count > 0:
                        continuation_events += 1
        
        if total_bias_events == 0:
            return 0, 0, 0
        
        continuation_rate = continuation_events / total_bias_events
        
        return total_bias_events, continuation_events, continuation_rate

    def mean_reversion_test(self, window=200, threshold=2, forward_look=10):
        """
        Tests if numbers that show high bias in a window appear more or less 
        frequently in the subsequent days compared to the baseline rate.
        """
        from collections import Counter
        
        total_bias_events = 0
        total_future_draws = 0
        total_future_hits = 0
        
        for i in range(window, self.n - forward_look):
            segment = self.real_series[i - window:i]
            counts = Counter(segment)
            
            expected = window / 100
            std_dev = np.sqrt(expected * (1 - 1/100))
            
            for num in range(100):
                observed = counts.get(num, 0)
                z = (observed - expected) / std_dev
                
                if z > threshold:
                    total_bias_events += 1
                    
                    future_segment = self.real_series[i:i + forward_look]
                    future_hits = np.sum(future_segment == num)
                    
                    total_future_hits += future_hits
                    total_future_draws += forward_look
        
        if total_future_draws == 0:
            return 0, 0, 0
        
        actual_rate = total_future_hits / total_future_draws
        baseline_rate = 1 / 100  # 1% per draw
        
        return total_bias_events, actual_rate, baseline_rate


if __name__ == "__main__":
    # Self-test logic to ensure functionality
    sample_data = np.random.randint(0, 100, 1000)
    simulator = MonteCarloSimulator(pd.Series(sample_data))
    
    print("Running simulations...")
    sim_res = simulator.run()
    
    print("Calculating real metrics...")
    real_res = simulator.real_metrics()
    
    print("Calculating percentiles...")
    simulator.calculate_percentiles(sim_res, real_res)

    print("\n--- Rolling Bias Events ---")
    rolling_results = simulator.rolling_bias_detection(window=200)
    for r in rolling_results[:5]:  # Show first 5 windows for brevity
        if r["high_bias"] or r["low_bias"]:
            print(f"Window {r['start_index']} - {r['end_index']}")
            print("High:", r["high_bias"])
            print("Low:", r["low_bias"])
            print()
