"""
Frequency Engine
----------------
Computes normalized frequency scores for Jodis over a rolling window.
"""
from typing import Dict
import pandas as pd

from scoring.utils import validate_df


class FrequencyEngine:
    """
    Frequency Analyzer Engine
    --------------------------
    Computes normalized frequency scores for jodis
    over a rolling window of days.
    """

    def __init__(self, window_days: int = 30):
        """
        Initializes the FrequencyEngine.

        Parameters
        ----------
        window_days : int, optional
            The number of historical days to consider for frequency calculation, by default 30.
        """
        self.window_days = window_days

    def run(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Run frequency analysis.
        """
        if df is None or df.empty:
            return {}

        data = validate_df(df)

        # --- Rolling Window Filter ---
        latest_date = data["Date"].max()
        cutoff_date = latest_date - pd.Timedelta(days=self.window_days)

        window_df = data[data["Date"] >= cutoff_date]

        if len(window_df) < 2: # Need at least some data for meaningful frequency
            return {}

        # --- Frequency Count ---
        freq_counts = window_df["Jodi"].value_counts()

        if freq_counts.empty:
            return {}

        max_count = freq_counts.max()
        if max_count == 0:
            return {}

        # --- Normalize to 0–100 ---
        freq_scores = {
            jodi: round((count / max_count) * 100, 2)
            for jodi, count in freq_counts.items()
        }

        return freq_scores
