"""
Confidence Engine
-----------------
Calculates a unified confidence score for each Jodi based on
contributions from Frequency, Cycles, Digits, and Momentum engines.
"""
from typing import Dict, List, Tuple
from config import (
    CONFIDENCE_WEIGHTS,
    HIGH_FREQ_THRESHOLD, LOW_FREQ_THRESHOLD,
    STRONG_DIGIT_THRESHOLD, WEAK_DIGIT_THRESHOLD,
    HIGH_MOMENTUM_THRESHOLD, LOW_MOMENTUM_THRESHOLD,
    BALANCED_SIGNAL_LOWER, BALANCED_SIGNAL_UPPER
)


class ConfidenceEngine:
    """
    Combines various analytical engine outputs to produce a single
    confidence score for each Jodi.
    """

    def __init__(self):
        """
        Initializes the ConfidenceEngine with predefined weights for each
        contributing analysis engine.

        Weights are chosen based on their perceived importance in identifying
        historical patterns:
        - Frequency: High importance, as recent occurrences are often significant.
        - Cycles: Moderate importance, indicating overdue or exhausted patterns.
        - Digits: Moderate importance, reflecting underlying digit strength.
        - Momentum: Lower importance, as short-term acceleration can be volatile.
        """
        self.weights = CONFIDENCE_WEIGHTS

    def data_confidence_factor(self, sample_size: int) -> float:
        """
        Calculates a data confidence factor based on the available sample size.
        This factor scales the final confidence score, giving less weight to
        analyses performed on smaller datasets.

        Parameters
        ----------
        sample_size : int
            The number of data points (e.g., historical days) used for the analysis.

        Returns
        -------
        float
            A factor between 0.40 and 1.00, where higher values indicate
            greater confidence due to a larger sample size.
        """
        if sample_size < 30:
            return 0.40
        if sample_size < 60:
            return 0.60
        if sample_size < 120:
            return 0.75
        if sample_size < 250:
            return 0.90
        return 1.00

    def run(
       self,
       frequency: Dict[str, float],
       cycles: Dict[str, Dict],
       digits: Dict[str, Dict],
       momentum: Dict[str, float],
       sample_size: int,
       top_n: int = 10,
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Calculates a unified confidence score for each Jodi based on
        contributions from Frequency, Cycles, Digits, and Momentum engines.

        Parameters
        ----------
        frequency : Dict[str, float]
            Jodi to frequency score (0-100).
        cycles : Dict[str, dict]
            Jodi to cycle analysis results (e.g., {'cycle_score': 50, 'status': 'NORMAL'}).
        digits : Dict[str, dict]
            Jodi to digit analysis results (e.g., {'digit_score': 75, ...}).
        momentum : Dict[str, float]
            Jodi to momentum score (0-200, capped). Normalized to 0-100 internally.
        sample_size : int
            The number of data points (e.g., historical days) used for the analysis.
        top_n : int, optional
            Number of top Jodis to return, by default 10.

        Returns
        -------
        List[Tuple[str, float, List[str]]]
            A list of tuples, each containing (jodi, confidence_score, tags).
            Confidence score is between 0 and 100.
        """

        scores = {}
        tags = {}

        # Combine all unique jodis from all engines
        all_jodis = (
            set(frequency.keys())
            | set(cycles.keys())
            | set(digits.keys())
            | set(momentum.keys())
        )

        for jodi in all_jodis:
            # Get scores from each engine, defaulting to neutral values if not present
            freq_score = frequency.get(jodi, 0)
            digit_info = digits.get(jodi, {"digit_score": 50})
            digit_score = digit_info.get("digit_score", 50)
            momentum_score_raw = momentum.get(jodi, 100)  # Default to 100 (neutral momentum)
            cycle_info = cycles.get(jodi, {"cycle_score": 50, "status": "NORMAL"})
            cycle_score_raw = cycle_info.get("cycle_score", 50)
            cycle_status = cycle_info.get("status", "NORMAL")

            # --- Normalize Momentum Score (0-200 -> 0-100) ---
            # 100 raw is neutral (50 normalized), 200 raw is max (100 normalized)
            momentum_score_normalized = (momentum_score_raw / 200) * 100
            momentum_score_normalized = min(max(momentum_score_normalized, 0), 100)

            # --- Calculate Weighted Score ---
            weighted_sum = (
                freq_score * self.weights["frequency"]
                + digit_score * self.weights["digits"]
                + momentum_score_normalized * self.weights["momentum"]
            )

            # --- Cycle Contribution (boost based on status) ---
            cycle_contribution = cycle_score_raw * self.weights["cycles"]
            if cycle_status == "DUE":
                cycle_contribution *= 1.25 # 25% boost for DUE signals
            elif cycle_status == "EXHAUSTED":
                cycle_contribution *= 0.75 # 25% penalty for EXHAUSTED signals

            weighted_sum += cycle_contribution
            
            # --- Sample Size Correction ---
            confidence_factor = self.data_confidence_factor(sample_size)
            final_score = round(weighted_sum * confidence_factor, 2)
            
            # Final Cap to 0-100
            final_score = min(max(final_score, 0), 100)
            scores[jodi] = final_score

            # --- Tag generation ---
            tag_list = []
            if freq_score >= HIGH_FREQ_THRESHOLD:
                tag_list.append("HIGH_FREQUENCY")
            elif freq_score <= LOW_FREQ_THRESHOLD:
                tag_list.append("LOW_FREQUENCY")

            if cycle_status == "DUE":
                tag_list.append("DUE_CYCLE")
            elif cycle_status == "EXHAUSTED":
                tag_list.append("EXHAUSTED_CYCLE")

            if digit_score >= STRONG_DIGIT_THRESHOLD:
                tag_list.append("STRONG_DIGIT_BIAS")
            elif digit_score <= WEAK_DIGIT_THRESHOLD:
                tag_list.append("WEAK_DIGIT_BIAS")

            if momentum_score_normalized >= HIGH_MOMENTUM_THRESHOLD:
                tag_list.append("HIGH_MOMENTUM")
            elif momentum_score_normalized <= LOW_MOMENTUM_THRESHOLD:
                tag_list.append("LOW_MOMENTUM")

            # Final Signal Tagging
            if final_score > BALANCED_SIGNAL_UPPER:
                tag_list.append("STRONG_ALIGNMENT")
            elif final_score < BALANCED_SIGNAL_LOWER:
                tag_list.append("WEAK_SIGNAL")
            else:
                tag_list.append("BALANCED_SIGNAL")

            tags[jodi] = tag_list

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return [
            (jodi, score, tags.get(jodi, []))
            for jodi, score in ranked
        ]
