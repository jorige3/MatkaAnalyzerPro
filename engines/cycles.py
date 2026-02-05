# engines/cycles.py

import pandas as pd
from typing import Dict


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
        pass

    def run(self, df: pd.DataFrame) -> Dict[str, dict]:
        """
        Run cycle (gap) analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: ['Date', 'Jodi']

        Returns
        -------
        Dict[str, dict]
            {
              jodi: {
                'days_since': int,
                'cycle_score': float (0–100),
                'status': 'DUE' | 'NORMAL' | 'EXHAUSTED'
              }
            }
        """

        # --- Validation ---
        required_cols = {"Date", "Jodi"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols}")

        data = df.copy()
        data["Date"] = pd.to_datetime(data["Date"])

        latest_date = data["Date"].max()

        # --- Last Appearance per Jodi ---
        last_seen = (
            data.sort_values("Date")
            .groupby("Jodi")["Date"]
            .last()
        )

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
