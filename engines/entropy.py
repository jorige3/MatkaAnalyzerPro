# engines/entropy.py

import pandas as pd
import numpy as np
from typing import Dict


class EntropyEngine:
    """
    Entropy Analyzer Engine
    -----------------------
    Measures the unpredictability or randomness of Jodi sequences.
    Higher entropy suggests more randomness, lower entropy suggests more predictable patterns.
    """

    def __init__(self):
        """
        Initializes the EntropyEngine.
        This engine currently does not require any specific parameters during initialization.
        """
        pass

    def run(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates the Shannon entropy of the 'Jodi' column in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain column: ['Jodi']

        Returns
        -------
        Dict[str, float]
            A dictionary containing a single key 'overall_entropy_score' with a value
            normalized to a 0-100 range.
        """
        if "Jodi" not in df.columns:
            raise ValueError("DataFrame must contain 'Jodi' column")

        if df.empty:
            return {"overall_entropy_score": 0.0}

        # Calculate probabilities of each unique Jodi
        jodi_counts = df["Jodi"].value_counts()
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
