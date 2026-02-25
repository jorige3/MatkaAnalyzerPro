"""
Paper Backtest Simulation
-------------------------
Simulates historical performance of the Matka Analyzer Pro by applying
the analysis engines on a rolling window of past data and evaluating
the hit rate of top predictions.
"""
from typing import Dict
import pandas as pd

from engines.frequency import FrequencyEngine
from engines.cycles import CycleEngine
from engines.digits import DigitEngine
from engines.momentum import MomentumEngine
from scoring.confidence import ConfidenceEngine
from config import TOP_N_PREDICTIONS


class PaperBacktest:
    """
    Simulates a paper trading strategy based on the confidence engine's predictions.
    It processes historical data in a walk-forward manner, generating predictions
    and checking if the actual result for the next day aligns with the top N predictions.
    """

    def __init__(self, csv_path: str, min_history_days: int = 180):
        """
        Initializes the PaperBacktest simulation.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing historical Matka data.
        min_history_days : int, optional
            Minimum number of days required for initial training, by default 180.
        """
        self.data_file = csv_path
        self.min_history_days = min_history_days
        self.all_data = pd.read_csv(self.data_file)
        self.all_data["Date"] = pd.to_datetime(self.all_data["Date"])
        self.all_data = self.all_data.sort_values("Date").reset_index(drop=True)

        self.frequency_engine = FrequencyEngine()
        self.cycle_engine = CycleEngine()
        self.digit_engine = DigitEngine()
        self.momentum_engine = MomentumEngine()
        self.confidence_engine = ConfidenceEngine()

    def run(self, top_n: int = TOP_N_PREDICTIONS, verbose: bool = False) -> Dict[str, any]:
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
        total_predictions = 0
        total_hits = 0

        unique_dates = self.all_data["Date"].unique()
        unique_dates.sort()

        # Iterate through each day, pretending to be "today"
        for i, current_date_np in enumerate(unique_dates):
            current_date = pd.to_datetime(current_date_np)

            # Ensure enough historical data is available
            historical_data = self.all_data[
                self.all_data["Date"] < current_date
            ].copy()

            if len(historical_data) < self.min_history_days:
                continue

            # Check if there's a result for "tomorrow" (the day after current_date)
            # This is crucial to avoid look-ahead bias
            next_day_data = self.all_data[
                self.all_data["Date"] == current_date
            ]
            if next_day_data.empty:
                continue # No result for the next day, skip this iteration

            actual_jodi_tomorrow = str(next_day_data["Jodi"].iloc[0]).zfill(2)

            # --- Run Engines on Historical Data (up to current_date - 1) ---
            frequency = self.frequency_engine.run(historical_data)
            cycles = self.cycle_engine.run(historical_data)
            digits_output = self.digit_engine.run(historical_data)
            digits = digits_output["jodi_scores"]
            momentum = self.momentum_engine.run(historical_data)

            # --- Get Top N Predictions for "tomorrow" ---
            confidence_results = self.confidence_engine.run(
                frequency,
                cycles,
                digits,
                momentum,
                sample_size=len(historical_data),
                top_n=top_n
            )
            top_predictions = [jodi for jodi, _, _ in confidence_results]

            total_predictions += 1
            if actual_jodi_tomorrow in top_predictions:
                total_hits += 1

        hit_rate = (total_hits / total_predictions) * 100 if total_predictions > 0 else 0

        return {
            "Total Predictions Made": total_predictions,
            f"Total Hits (within Top {top_n})": total_hits,
            "Historical Alignment Rate": round(hit_rate, 2),
            "Disclaimer": "Historical alignment rate is not a predictor of future outcomes."
        }
