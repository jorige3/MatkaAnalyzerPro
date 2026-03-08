"""
Paper Backtest Simulation
-------------------------
Simulates historical performance of the Matka Analyzer Pro by applying
the analysis engines on a rolling window of past data and evaluating
the hit rate of top predictions.
"""
import sys
import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Ensure parent directory is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from engines.frequency import FrequencyEngine
from engines.cycles import CycleEngine
from engines.digits import DigitEngine
from engines.momentum import MomentumEngine
from scoring.confidence import ConfidenceEngine
from config import TOP_N_PREDICTIONS, DATA_FILE, MIN_HISTORY_DAYS


class PaperBacktest:
    """
    Simulates a paper trading strategy based on the confidence engine's predictions.
    Avoids look-ahead bias by using only strictly historical data for each day.
    """

    def __init__(self, csv_path: str, min_history_days: int = 180):
        self.data_file = csv_path
        self.min_history_days = min_history_days

        self.all_data = pd.read_csv(self.data_file)
        if "Date" not in self.all_data.columns or "Jodi" not in self.all_data.columns:
            raise ValueError("CSV must contain 'Date' and 'Jodi' columns")

        self.all_data["Date"] = pd.to_datetime(self.all_data["Date"])
        self.all_data["Jodi"] = self.all_data["Jodi"].astype(str).str.zfill(2)
        self.all_data = self.all_data.sort_values("Date").reset_index(drop=True)

        # Initialize engines
        self.frequency_engine = FrequencyEngine()
        self.cycle_engine = CycleEngine()
        self.digit_engine = DigitEngine()
        self.momentum_engine = MomentumEngine()
        self.confidence_engine = ConfidenceEngine()

    def run(self, top_n: int = TOP_N_PREDICTIONS, verbose: bool = False) -> Dict[str, Any]:
        """
        Runs the walk-forward simulation.
        """
        total_predictions = 0
        total_hits = 0
        daily_results = []
        hit_series = []

        unique_dates = np.sort(self.all_data["Date"].unique())

        for current_date in unique_dates:
            historical_data = self.all_data[self.all_data["Date"] < current_date].copy()

            if len(historical_data) < self.min_history_days:
                continue

            current_day_data = self.all_data[self.all_data["Date"] == current_date]
            if current_day_data.empty:
                continue

            actual_jodi = str(current_day_data["Jodi"].iloc[0]).zfill(2)

            # --- Run Engines ---
            frequency = self.frequency_engine.run(historical_data)
            cycles = self.cycle_engine.run(historical_data)
            digits_output = self.digit_engine.run(historical_data)
            digits = digits_output.get("jodi_scores", {})
            momentum = self.momentum_engine.run(historical_data)

            # --- Confidence Scoring ---
            confidence_results = self.confidence_engine.run(
                frequency, cycles, digits, momentum,
                sample_size=len(historical_data),
                top_n=top_n
            )

            top_predictions = [jodi for jodi, _, _ in confidence_results]
            is_hit = actual_jodi in top_predictions

            if is_hit:
                total_hits += 1
            
            total_predictions += 1
            hit_series.append(1 if is_hit else 0)

            if verbose:
                daily_results.append({
                    "date": current_date,
                    "actual_jodi": actual_jodi,
                    "top_predictions": ", ".join(top_predictions),
                    "hit": is_hit
                })

        # --- Calculate Statistics ---
        hit_rate = (total_hits / total_predictions * 100) if total_predictions > 0 else 0
        baseline_rate = (top_n / 100.0) * 100
        edge = hit_rate - baseline_rate
        
        # Calculate Rolling Accuracy (Window of 20)
        rolling_accuracy = []
        if len(hit_series) >= 20:
            rolling_accuracy = pd.Series(hit_series).rolling(window=20).mean().tolist()
            max_drawdown_hits = 1.0 - (min(rolling_accuracy) if rolling_accuracy else 0)
        else:
            max_drawdown_hits = 0.0

        results_dict = {
            "total_days_tested": total_predictions,
            "hits": total_hits,
            "misses": total_predictions - total_hits,
            "historical_alignment_rate": round(hit_rate, 2),
            "baseline_rate": round(baseline_rate, 2),
            "edge_over_baseline": round(edge, 2),
            "top_n_considered": top_n,
            "max_volatility_hit_rate": round(max_drawdown_hits, 4)
        }

        if verbose:
            results_dict["daily_results"] = daily_results
            print(f"Backtest Complete: {total_predictions} days, {total_hits} hits ({hit_rate:.2f}%). Edge: {edge:+.2f}%")

        return results_dict

if __name__ == "__main__":
    from config import DATA_FILE, MIN_HISTORY_DAYS
    print("--- Running Advanced Paper Backtest ---")
    backtester = PaperBacktest(DATA_FILE, min_history_days=MIN_HISTORY_DAYS)
    results = backtester.run(top_n=10, verbose=True)
    print(f"Final Stats: {results}")
