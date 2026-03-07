"""
Statistical Analysis Module
---------------------------
Provides various statistical tests and metrics for analyzing Jodi patterns,
including Shannon Entropy, Chi-Square uniformity tests, autocorrelation, 
and streak analysis.
"""
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from typing import Tuple

class StatisticalAnalyzer:
    """
    Performs statistical validation and pattern analysis on historical data.
    """

    def __init__(self, df: pd.DataFrame, target_col: str = 'Jodi'):
        """
        Initializes the StatisticalAnalyzer.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the target column.
        target_col : str, optional
            The name of the column to analyze, by default 'Jodi'.
        """
        self.df = df.copy()
        self.target_col = target_col

    def shannon_entropy(self) -> Tuple[float, float]:
        """
        Calculates the Shannon entropy of the target column.
        Entropy measures the randomness/unpredictability of the data.

        Returns
        -------
        Tuple[float, float]
            (current_entropy, max_theoretical_entropy)
        """
        values = self.df[self.target_col]
        probs = values.value_counts(normalize=True)
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probs)) if len(probs) > 0 else 0.0
        return float(entropy), float(max_entropy)

    def chi_square_test(self) -> Tuple[float, float]:
        """
        Performs a Chi-Square goodness-of-fit test against a uniform distribution.
        Determines if Jodis are appearing with significantly different frequencies.

        Returns
        -------
        Tuple[float, float]
            (chi_statistic, p_value)
        """
        observed = self.df[self.target_col].value_counts().sort_index()
        if observed.empty:
            return 0.0, 1.0
        expected = np.ones(len(observed)) * (len(self.df) / len(observed))
        chi_stat, p_value = chisquare(observed, expected)
        return float(chi_stat), float(p_value)

    def autocorrelation(self, lag: int = 1) -> float:
        """
        Calculates the autocorrelation of the target column at a given lag.
        Measures if the current value is correlated with its predecessors.

        Parameters
        ----------
        lag : int, optional
            The lag distance, by default 1.

        Returns
        -------
        float
            Autocorrelation coefficient.
        """
        # Convert jodi to numeric for autocorr calculation
        series = pd.to_numeric(self.df[self.target_col], errors='coerce')
        return float(series.autocorr(lag=lag))

    def longest_streak(self) -> int:
        """
        Finds the longest repeating streak of the same Jodi value.

        Returns
        -------
        int
            Length of the longest streak.
        """
        values = self.df[self.target_col].tolist()
        if not values:
            return 0
        max_streak = 1
        current_streak = 1

        for i in range(1, len(values)):
            if values[i] == values[i-1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1

        return int(max_streak)
