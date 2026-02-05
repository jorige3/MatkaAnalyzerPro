# engines/digits.py

import pandas as pd
from typing import Dict


class DigitEngine:
    """
    Digit Strength Analyzer
    -----------------------
    Evaluates strength of single digits (0–9)
    and computes digit-based scores for jodis.
    """

    def __init__(self):
        """
        Initializes the DigitEngine.
        This engine currently does not require any specific parameters during initialization.
        """
        pass

    def run(self, df: pd.DataFrame) -> Dict[str, dict]:
        """
        Run digit analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain column: ['Jodi']

        Returns
        -------
        Dict[str, dict]
        {
          "jodi_scores": {
            jodi: {
              'digit_score': float (0–100),
              'tens_digit': int,
              'unit_digit': int
            }
          },
          "individual_digit_strength": {
            digit: float (0-100)
          }
        }
        """

        if "Jodi" not in df.columns:
            raise ValueError("DataFrame must contain 'Jodi' column")

        data = df.copy()

        # Ensure proper string format
        data["Jodi"] = data["Jodi"].astype(str).str.zfill(2)

        # --- Extract digits ---
        data["tens"] = data["Jodi"].str[0].astype(int)
        data["units"] = data["Jodi"].str[1].astype(int)

        # --- Count digit frequency ---
        digit_counts = (
            pd.concat([data["tens"], data["units"]])
            .value_counts()
            .sort_index()
        )

        if digit_counts.empty:
            return {}

        max_count = digit_counts.max()

        # Normalize digit strength to 0–100
        digit_strength = {
            digit: round((count / max_count) * 100, 2)
            for digit, count in digit_counts.items()
        }

        # --- Score jodis ---
        results = {}

        for jodi in data["Jodi"].unique():
            tens = int(jodi[0])
            units = int(jodi[1])

            tens_score = digit_strength.get(tens, 0)
            units_score = digit_strength.get(units, 0)

            digit_score = round((tens_score + units_score) / 2, 2)

            results[jodi] = {
                "digit_score": digit_score,
                "tens_digit": tens,
                "unit_digit": units
            }

        return {
            "jodi_scores": results,
            "individual_digit_strength": digit_strength
        }
