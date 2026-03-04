"""
Random Baseline Backtest
------------------------
Simulates random top-N predictions to compare against model performance.
"""
import sys
import os

# Get the absolute path of the directory containing random_baseline.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (matka_analyzer/)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from typing import Dict, Any
import pandas as pd
import random
from config import DATA_FILE, MIN_HISTORY_DAYS


class RandomBaselineBacktest:

    def __init__(self, csv_path: str, min_history_days: int = 180):
        self.data_file = csv_path
        self.min_history_days = min_history_days

        self.all_data = pd.read_csv(self.data_file)

        if "Date" not in self.all_data.columns:
            raise ValueError("CSV must contain 'Date' column")

        if "Jodi" not in self.all_data.columns:
            raise ValueError("CSV must contain 'Jodi' column")

        self.all_data["Date"] = pd.to_datetime(self.all_data["Date"])
        self.all_data = self.all_data.sort_values("Date").reset_index(drop=True)

    def run(self, top_n: int = 10, verbose: bool = False) -> Dict[str, Any]:

        total_hits = 0
        total_predictions = 0
        daily_results = []

        unique_dates = sorted(self.all_data["Date"].unique())

        for current_date in unique_dates:

            historical_data = self.all_data[
                self.all_data["Date"] < current_date
            ]

            if len(historical_data) < self.min_history_days:
                continue

            current_day_data = self.all_data[
                self.all_data["Date"] == current_date
            ]

            if current_day_data.empty:
                continue

            actual_jodi = str(current_day_data["Jodi"].iloc[0]).zfill(2)

            # 🎲 Randomly select top_n jodis from 00–99
            all_possible_jodis = [str(i).zfill(2) for i in range(100)]
            random_predictions = random.sample(all_possible_jodis, top_n)

            is_hit = actual_jodi in random_predictions

            if is_hit:
                total_hits += 1

            total_predictions += 1

            if verbose:
                daily_results.append({
                    "date": current_date,
                    "actual_jodi": actual_jodi,
                    "random_predictions": random_predictions,
                    "hit": is_hit
                })

        hit_rate = (
            (total_hits / total_predictions) * 100
            if total_predictions > 0 else 0
        )

        results = {
            "total_days_tested": total_predictions,
            "hits": total_hits,
            "misses": total_predictions - total_hits,
            "hit_rate_percent": round(hit_rate, 2),
            "top_n_considered": top_n
        }

        if verbose:
            results["daily_results"] = daily_results

        return results

if __name__ == "__main__":
    print("--- Running Random Baseline Backtest ---")
    baseline = RandomBaselineBacktest(DATA_FILE, min_history_days=MIN_HISTORY_DAYS)
    res = baseline.run(top_n=10, verbose=False)
    print(f"Random Baseline (Top-10): {res['hit_rate_percent']}% Hit Rate")
