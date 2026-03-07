"""
Digit Engine
------------
Analyzes the frequency and strength of individual digits (0-9)
within Jodis to identify biases.
"""
from typing import Dict
import pandas as pd


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


    def run(self, df: pd.DataFrame) -> Dict[str, dict]:
        """
        Run digit analysis.
        """
        if df is None or df.empty:
            return {"jodi_scores": {}, "individual_digit_strength": {}}

        if "Jodi" not in df.columns:
            raise ValueError("DataFrame must contain 'Jodi' column")

        data = df.copy()
        if len(data) < 1:
            return {"jodi_scores": {}, "individual_digit_strength": {}}

        # Ensure proper string format
        data["Jodi"] = data["Jodi"].astype(str).str.zfill(2)

        # --- Extract digits ---
        data["tens"] = data["Jodi"].str[0].astype(int)
        data["units"] = data["Jodi"].str[1].astype(int)

        # --- Count digit frequency ---
        all_digits = pd.concat([data["tens"], data["units"]])
        digit_counts = all_digits.value_counts().sort_index()

        if digit_counts.empty:
            return {"jodi_scores": {}, "individual_digit_strength": {}}

        max_count = digit_counts.max()
        if max_count == 0:
            return {"jodi_scores": {}, "individual_digit_strength": {}}

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
