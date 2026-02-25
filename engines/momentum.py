"""
Momentum Engine
---------------
Measures short-term acceleration of jodi appearances
compared to a longer baseline.
"""
from typing import Dict
import pandas as pd


class MomentumEngine:
    """
    Momentum Analyzer Engine
    ------------------------
    Measures short-term acceleration of jodi appearances
    compared to a longer baseline.
    """

    def __init__(self, recent_days: int = 7, baseline_days: int = 30):
        """
        Initializes the MomentumEngine.

        Parameters
        ----------
        recent_days : int, optional
            The number of recent days to consider for momentum calculation, by default 7.
        baseline_days : int, optional
            The number of baseline days to consider for momentum calculation, by default 30.
        """
        if recent_days >= baseline_days:
            raise ValueError("recent_days must be less than baseline_days")

        self.recent_days = recent_days
        self.baseline_days = baseline_days

    def run(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Run momentum analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: ['Date', 'Jodi']

        Returns
        -------
        Dict[str, float]
            { jodi: momentum_score (0–100) }
        """

        required_cols = {"Date", "Jodi"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols}")

        data = df.copy()
        data["Date"] = pd.to_datetime(data["Date"])

        latest_date = data["Date"].max()

        recent_cutoff = latest_date - pd.Timedelta(days=self.recent_days)
        baseline_cutoff = latest_date - pd.Timedelta(days=self.baseline_days)

        recent_df = data[data["Date"] >= recent_cutoff]
        baseline_df = data[data["Date"] >= baseline_cutoff]

        if baseline_df.empty:
            return {}

        recent_counts = recent_df["Jodi"].value_counts()
        baseline_counts = baseline_df["Jodi"].value_counts()

        results = {}

        for jodi in baseline_counts.index:
            recent = recent_counts.get(jodi, 0)
            baseline = baseline_counts.get(jodi, 0)

            # Normalize
            recent_norm = recent / self.recent_days
            baseline_norm = baseline / self.baseline_days

            if baseline_norm == 0:
                momentum = 0.0
            else:
                momentum = (recent_norm / baseline_norm) * 100

            # Cap to avoid spikes
            momentum = min(momentum, 200)

            results[jodi] = round(momentum, 2)

        return results
