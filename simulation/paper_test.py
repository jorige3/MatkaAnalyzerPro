"""
Paper Backtest Simulation
-------------------------
Simulates historical performance of the Matka Analyzer Pro by applying
the analysis engines on a rolling window of past data and evaluating
the hit rate of top predictions.
"""
from typing import Dict, Any
import pandas as pd
import numpy as np

from engines.frequency import FrequencyEngine
from engines.cycles import CycleEngine
from engines.digits import DigitEngine
from engines.momentum import MomentumEngine
from scoring.confidence import ConfidenceEngine
from config import TOP_N_PREDICTIONS


class PaperBacktest:
    """
    Simulates a paper trading strategy based on the confidence engine's predictions.
    """

    def __init__(self, csv_path: str, min_history_days: int = 180):
        self.data_file = csv_path
        self.min_history_days = min_history_days

        self.all_data = pd.read_csv(self.data_file)

        if "Date" not in self.all_data.columns:
            raise ValueError("CSV must contain a 'Date' column")

        if "Jodi" not in self.all_data.columns:
            raise ValueError("CSV must contain a 'Jodi' column")

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

        total_predictions = 0
        total_hits = 0
        daily_results = []

        unique_dates = np.sort(self.all_data["Date"].unique())

        # Walk-forward simulation
        for current_date in unique_dates:

            # Historical data strictly before current date
            historical_data = self.all_data[
                self.all_data["Date"] < current_date
            ].copy()

            # Ensure minimum training window
            if len(historical_data) < self.min_history_days:
                continue

            # Get actual result for current day
            current_day_data = self.all_data[
                self.all_data["Date"] == current_date
            ]

            if current_day_data.empty:
                continue

            actual_jodi = str(current_day_data["Jodi"].iloc[0]).zfill(2)

            # --- Run Engines ---
            frequency = self.frequency_engine.run(historical_data)
            cycles = self.cycle_engine.run(historical_data)
            digits_output = self.digit_engine.run(historical_data)
            digits = digits_output["jodi_scores"]
            momentum = self.momentum_engine.run(historical_data)

            # --- Confidence Scoring ---
            confidence_results = self.confidence_engine.run(
                frequency,
                cycles,
                digits,
                momentum,
                sample_size=len(historical_data),
                top_n=top_n
            )

            top_predictions = [jodi for jodi, _, _ in confidence_results]

            # Check hit
            is_hit = actual_jodi in top_predictions

            if is_hit:
                total_hits += 1

            total_predictions += 1

            if verbose:
                daily_results.append({
                    "date": current_date,
                    "actual_jodi": actual_jodi,
                    "top_predictions": ", ".join(top_predictions), # Convert list to string
                    "hit": is_hit,
                    "reason": "Hit" if is_hit else "Miss"
                })

        misses = total_predictions - total_hits
        hit_rate = (
            (total_hits / total_predictions) * 100
            if total_predictions > 0 else 0
        )

        # Additional stats
        avg_hits_per_day = total_hits / total_predictions if total_predictions > 0 else 0

        results_dict = {
            "total_days_tested": total_predictions,
            "hits": total_hits,
            "misses": misses,
            "historical_alignment_rate": round(hit_rate, 2),
            "top_n_considered": top_n,
            "avg_hits_per_day": round(avg_hits_per_day, 4)
        }

        if verbose:
            results_dict["daily_results"] = daily_results
            print(f"Backtest Complete: {total_predictions} days tested, {total_hits} hits ({hit_rate:.2f}% rate).")

        return results_dict

    def backtest_simple_strategy(self, z_threshold: float = 1.5, due_days: int = 7, verbose: bool = False) -> Dict[str, Any]:
        """
        Backtests a simple strategy: Bet on Jodis with Z-score > threshold and last hit >= due_days ago.
        """
        from analyzer import analyze_jodi
        
        total_days = 0
        total_hits = 0
        total_bets = 0
        
        unique_dates = np.sort(self.all_data["Date"].unique())
        
        for current_date in unique_dates:
            historical_data = self.all_data[self.all_data["Date"] < current_date].copy()
            if len(historical_data) < self.min_history_days:
                continue
                
            current_day_data = self.all_data[self.all_data["Date"] == current_date]
            if current_day_data.empty:
                continue
            actual_jodi = str(current_day_data["Jodi"].iloc[0]).zfill(2)
            
            # Identify candidates
            candidates = []
            for j in range(100):
                j_str = str(j).zfill(2)
                res = analyze_jodi(historical_data, j_str)
                if res['z_score'] > z_threshold and (res['days_since'] is None or res['days_since'] >= due_days):
                    candidates.append(j_str)
            
            if not candidates:
                continue
                
            is_hit = actual_jodi in candidates
            if is_hit:
                total_hits += 1
            
            total_bets += len(candidates)
            total_days += 1
            
        hit_rate = (total_hits / total_days * 100) if total_days > 0 else 0
        efficiency = (total_hits / total_bets * 100) if total_bets > 0 else 0
        
        results = {
            "strategy": f"Z-Score > {z_threshold} & Days Since >= {due_days}",
            "total_days_active": total_days,
            "total_hits": total_hits,
            "total_bets_placed": total_bets,
            "day_hit_rate": round(hit_rate, 2),
            "bet_efficiency": round(efficiency, 2)
        }
        
        if verbose:
            print(f"Simple Strategy Backtest: {results}")
            
        return results
