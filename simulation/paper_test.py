# simulation/paper_test.py

import pandas as pd
from typing import Dict, Tuple, List


from engines.frequency import FrequencyEngine
from engines.cycles import CycleEngine
from engines.digits import DigitEngine
from engines.momentum import MomentumEngine
from scoring.confidence import ConfidenceEngine


class PaperBacktest:
    """
    Paper Backtesting Simulation Engine
    -----------------------------------
    Simulates the performance of the Matka Analyzer Pro's confidence scoring
    system against historical data. It operates in a walk-forward manner,
    ensuring no look-ahead bias.

    The simulation calculates how often the actual Jodi for a given day
    appears within the top-N predicted Jodis based on historical analysis
    up to the day before.
    """

    def __init__(self, csv_path: str, min_history_days: int = 30):
        """
        Initializes the PaperBacktest engine.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing historical Matka data.
            Must contain 'Date' and 'Jodi' columns.
        min_history_days : int, optional
            The minimum number of historical days required to start analysis
            for a given day. This acts as the initial training window.
            Defaults to 30.
        """
        """
        Initializes the PaperBacktest engine.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing historical Matka data.
            Must contain 'Date' and 'Jodi' columns.
        min_history_days : int, optional
            The minimum number of historical days required to start analysis
            for a given day. This acts as the initial training window.
            Defaults to 30.
        """
        self.df = pd.read_csv(csv_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values("Date").reset_index(drop=True)
        self.min_history_days = min_history_days

        # Initialize all analysis engines
        self.freq_engine = FrequencyEngine(window_days=min_history_days)
        self.cycle_engine = CycleEngine()
        self.digit_engine = DigitEngine()
        self.momentum_engine = MomentumEngine()
        self.confidence_engine = ConfidenceEngine()

    def run(self, top_n: int = 10, verbose: bool = False) -> Dict[str, any]:
        """
        Executes the walk-forward paper backtest simulation.

        For each day in the historical data (after the initial history window),
        it performs the following:
        1. Gathers historical data up to the previous day.
        2. Runs all analysis engines on this historical data.
        3. Generates top-N confidence-scored Jodis.
        4. Checks if the actual Jodi for the current day is within the top-N.

        Parameters
        ----------
        top_n : int, optional
            The number of top Jodis to consider as a "hit" for the actual result.
            Defaults to 10.
        verbose : bool, optional
            If True, returns a detailed list of daily results. Defaults to False.

        Returns
        -------
        Dict[str, any]
            A dictionary containing simulation statistics:
            - 'total_days_tested': Total number of days for which predictions were made.
            - 'hits': Number of times the actual Jodi was in the top-N predictions.
            - 'misses': Number of times the actual Jodi was NOT in the top-N predictions.
            - 'historical_alignment_rate': Percentage of hits out of total days tested.
            - 'top_n_considered': The 'top_n' value used for the simulation.
            - 'daily_results': (Optional) A list of dictionaries with daily outcomes.
        """
        hits = 0
        misses = 0
        total_days_tested = 0
        daily_results = [] # To store detailed daily outcomes if needed for logging

        # Start simulation after the initial history window
        for i in range(self.min_history_days, len(self.df)):
            # Ensure no look-ahead bias: use data ONLY up to the day before 'i'
            history_df = self.df.iloc[:i].copy()
            actual_jodi = str(self.df.iloc[i]["Jodi"]).zfill(2)
            current_date = self.df.iloc[i]["Date"]

            if history_df.empty:
                continue # Should not happen if min_history_days is respected

            # Run engines on the historical data
            freq_results = self.freq_engine.run(history_df)
            cycle_results = self.cycle_engine.run(history_df)
            digit_results = self.digit_engine.run(history_df)
            momentum_results = self.momentum_engine.run(history_df)

            # Generate confidence scores
            ranked_jodis = self.confidence_engine.run(
                frequency=freq_results,
                cycles=cycle_results,
                digits=digit_results,
                momentum=momentum_results,
                sample_size=len(history_df),
                top_n=top_n
            )

            predicted_jodis = [jodi for jodi, _, _ in ranked_jodis]

            is_hit = actual_jodi in predicted_jodis
            if is_hit:
                hits += 1
            else:
                misses += 1
            total_days_tested += 1

            if verbose:
                actual_jodi_details = next(((j, s, t) for j, s, t in ranked_jodis if j == actual_jodi), None)
                top_predicted_jodi_details = ranked_jodis[0] if ranked_jodis else (None, None, None)

                daily_results.append({
                    "date": current_date,
                    "actual_jodi": actual_jodi,
                    "predicted_top_n": [j for j, _, _ in ranked_jodis],
                    "is_hit": is_hit,
                    "actual_jodi_confidence": actual_jodi_details[1] if actual_jodi_details else None,
                    "actual_jodi_tags": actual_jodi_details[2] if actual_jodi_details else None,
                    "top_predicted_jodi": top_predicted_jodi_details[0],
                    "top_predicted_confidence": top_predicted_jodi_details[1],
                    "top_predicted_tags": top_predicted_jodi_details[2],
                })

        historical_alignment_rate = round((hits / total_days_tested) * 100, 2) if total_days_tested else 0.0

        results = {
            "total_days_tested": total_days_tested,
            "hits": hits,
            "misses": misses,
            "historical_alignment_rate": historical_alignment_rate,
            "top_n_considered": top_n,
        }
        if verbose:
            results["daily_results"] = daily_results
        return results
