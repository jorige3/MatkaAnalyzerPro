import numpy as np
import pandas as pd
from scipy.stats import chisquare
from collections import Counter

class StatisticalAnalyzer:

    def __init__(self, df, target_col='jodi'):
        self.df = df.copy()
        self.target_col = target_col

    # ----------------------------------
    # 1. Shannon Entropy
    # ----------------------------------
    def shannon_entropy(self):
        values = self.df[self.target_col]
        probs = values.value_counts(normalize=True)
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probs))
        return entropy, max_entropy

    # ----------------------------------
    # 2. Chi-Square Uniformity Test
    # ----------------------------------
    def chi_square_test(self):
        observed = self.df[self.target_col].value_counts().sort_index()
        expected = np.ones(len(observed)) * (len(self.df) / len(observed))
        chi_stat, p_value = chisquare(observed, expected)
        return chi_stat, p_value

    # ----------------------------------
    # 3. Autocorrelation
    # ----------------------------------
    def autocorrelation(self, lag=1):
        series = self.df[self.target_col]
        return series.autocorr(lag=lag)

    # ----------------------------------
    # 4. Longest Streak
    # ----------------------------------
    def longest_streak(self):
        values = self.df[self.target_col].tolist()
        max_streak = 1
        current_streak = 1

        for i in range(1, len(values)):
            if values[i] == values[i-1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1

        return max_streak
