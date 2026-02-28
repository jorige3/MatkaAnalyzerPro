import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from math import log2



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

    def real_metrics(self):
        return {
            "real_streak": self._longest_streak(self.real_series),
            "real_entropy": self._entropy(self.real_series),
            "real_max_freq": self._max_frequency(self.real_series),
        }

    
    def plot_results(self, simulation_results, real_results):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1️⃣ Streak Distribution
        axes[0].hist(simulation_results["streaks"], bins=20)
        axes[0].axvline(real_results["real_streak"], linestyle="dashed")
        axes[0].set_title("Longest Streak Distribution")
        axes[0].set_xlabel("Streak Length")
        axes[0].set_ylabel("Frequency")

        # 2️⃣ Max Frequency Distribution
        axes[1].hist(simulation_results["max_freqs"], bins=20)
        axes[1].axvline(real_results["real_max_freq"], linestyle="dashed")
        axes[1].set_title("Max Frequency Distribution")
        axes[1].set_xlabel("Max Frequency")
        axes[1].set_ylabel("Frequency")

        # 3️⃣ Entropy Distribution
        axes[2].hist(simulation_results["entropies"], bins=20)
        axes[2].axvline(real_results["real_entropy"], linestyle="dashed")
        axes[2].set_title("Entropy Distribution")
        axes[2].set_xlabel("Entropy")
        axes[2].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()