"""
Cycle / Gap Analyzer Engine
---------------------------
Computes recency-based cycle scores for jodis.
"""
from typing import Dict
import pandas as pd

from scoring.utils import validate_df


class CycleEngine:
    """
    Cycle / Gap Analyzer Engine
    ---------------------------
    Computes recency-based cycle scores for jodis.
    """

    def __init__(self):
        """
        Initializes the CycleEngine.
        This engine currently does not require any specific parameters during initialization.
        """


    def run(self, df: pd.DataFrame) -> Dict[str, dict]:
        """
        Run cycle (gap) analysis.
        """
        if df is None or df.empty:
            return {}

        data = validate_df(df)
        if len(data) < 2:
            return {}

        latest_date = data["Date"].max()

        # --- Last Appearance per Jodi ---
        last_seen = (
            data.sort_values("Date")
            .groupby("Jodi")["Date"]
            .last()
        )

        if last_seen.empty:
            return {}

        days_since = (latest_date - last_seen).dt.days

        if days_since.empty:
            return {}

        max_gap = days_since.max()
        min_gap = days_since.min()

        results = {}

        for jodi, gap in days_since.items():
            # Normalize gap to 0–100
            if max_gap == min_gap:
                score = 50.0
            else:
                score = ((gap - min_gap) / (max_gap - min_gap)) * 100

            score = round(score, 2)

            # --- Status Classification ---
            if score >= 70:
                status = "DUE"
            elif score <= 30:
                status = "EXHAUSTED"
            else:
                status = "NORMAL"

            results[jodi] = {
                "days_since": int(gap),
                "cycle_score": score,
                "status": status
            }

        return results
