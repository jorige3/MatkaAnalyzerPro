"""
Entropy Engine
--------------
Calculates the entropy of Jodi distributions over time to identify
periods of high or low predictability.
"""
from typing import Dict
import pandas as pd
import numpy as np


class EntropyEngine:
    """
    Analyzes the entropy of Jodi occurrences.
    High entropy suggests randomness, low entropy suggests emerging patterns.
    """

    def __init__(self):
        """
        Initializes the EntropyEngine.
        This engine currently does not require any specific parameters during initialization.
        """


    def run(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates the Shannon entropy of the 'Jodi' column in the DataFrame.
        """
        if df is None or df.empty:
            return {"overall_entropy_score": 0.0}

        if "Jodi" not in df.columns:
            raise ValueError("DataFrame must contain 'Jodi' column")

        if len(df) < 2:
            return {"overall_entropy_score": 0.0}

        # Calculate probabilities of each unique Jodi
        jodi_counts = df["Jodi"].value_counts()
        if jodi_counts.empty:
            return {"overall_entropy_score": 0.0}

        probabilities = jodi_counts / len(df)

        # Calculate Shannon entropy
        # H = - sum(p_i * log2(p_i))
        entropy = -np.sum(probabilities * np.log2(probabilities))

        # Normalize entropy to a 0-100 range
        # Max entropy for N unique items is log2(N)
        num_unique_jodis = len(jodi_counts)
        if num_unique_jodis <= 1: # Handle cases with 0 or 1 unique jodi
            normalized_entropy = 0.0
        else:
            max_possible_entropy = np.log2(num_unique_jodis)
            normalized_entropy = (entropy / max_possible_entropy) * 100

        return {"overall_entropy_score": round(normalized_entropy, 2)}
