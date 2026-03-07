"""
Tail Risk & Volatility Clustering Analyzer
-------------------------------------------
Hypothesis: Extreme events (Z-score > 2.0) cluster in time.
This module identifies 'bursts' of high-bias events and tests if 
they can predict the next draw's probability of being another extreme.
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Dict, Any
from analyzer import analyze_jodi

class TailRiskAnalyzer:
    """
    Analyzes the clustering behavior of 'extreme' Jodi events.
    """
    def __init__(self, z_threshold: float = 2.0):
        self.z_threshold = z_threshold

    def _is_extreme(self, df: pd.DataFrame, jodi: str) -> bool:
        res = analyze_jodi(df, jodi)
        return res['z_score'] > self.z_threshold

    def run_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Identify which historical draws were 'extreme' at the time of draw
        extremes = []
        # Sample every 5 days for speed in analysis
        for i in range(200, len(df)):
            current_jodi = df.iloc[i]['Jodi']
            # Analysis strictly on data BEFORE the draw
            if self._is_extreme(df.iloc[:i], current_jodi):
                extremes.append(1)
            else:
                extremes.append(0)
        
        extremes_series = pd.Series(extremes)
        
        # Calculate Index of Dispersion (Variance / Mean)
        # For a Poisson process (random), IoD = 1.
        # If IoD > 1, events are clustered (Overdispersion).
        mean_ext = extremes_series.mean()
        var_ext = extremes_series.var()
        iod = var_ext / mean_ext if mean_ext > 0 else 0
        
        return {
            "extreme_event_rate": round(mean_ext, 4),
            "index_of_dispersion": round(iod, 4),
            "is_clustered": iod > 1.1,
            "total_extremes": sum(extremes)
        }

    def quick_backtest(self, df: pd.DataFrame, lookback: int = 3) -> Dict[str, Any]:
        """
        Burst Hypothesis: If an extreme event happened in the last 'lookback' days,
        predict that the next draw will also be an 'extreme' Jodi.
        """
        jodis = df['Jodi'].tolist()
        hits = 0
        total_predictions = 0
        
        # We need to pre-calculate 'extremeness' for each day to avoid O(N^2)
        is_ext_list = []
        for i in range(200, len(df)):
            is_ext_list.append(self._is_extreme(df.iloc[:i], df.iloc[i]['Jodi']))
            
        # Start backtesting after we have some results
        for i in range(lookback, len(is_ext_list) - 1):
            # If any event in the last 'lookback' days was extreme
            if any(is_ext_list[i-lookback:i]):
                total_predictions += 1
                if is_ext_list[i+1]:
                    hits += 1
                    
        hit_rate = hits / total_predictions if total_predictions > 0 else 0
        baseline = sum(is_ext_list) / len(is_ext_list)
        
        # Z-Score
        std_error = np.sqrt((baseline * (1 - baseline)) / total_predictions) if total_predictions > 0 else 0
        z_score = (hit_rate - baseline) / std_error if std_error > 0 else 0
        p_value = 1 - norm.cdf(z_score)

        return {
            "hit_rate": round(hit_rate, 4),
            "baseline": round(baseline, 4),
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "total_predictions": total_predictions,
            "hits": hits
        }

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    analyzer = TailRiskAnalyzer(z_threshold=2.0)
    print("--- Tail Risk & Volatility Analysis ---")
    stats = analyzer.run_analysis(df)
    print(f"Extreme Event Rate: {stats['extreme_event_rate']:.2%}")
    print(f"Index of Dispersion: {stats['index_of_dispersion']}")
    print(f"Clustering Detected: {stats['is_clustered']}")
    
    print("\n--- Burst Hypothesis Backtest (3-day window) ---")
    bt = analyzer.quick_backtest(df, lookback=3)
    print(f"Hit Rate (Extreme after Extreme): {bt['hit_rate']:.2%} (Baseline: {bt['baseline']:.2%})")
    print(f"Z-Score: {bt['z_score']}")
    print(f"P-Value: {bt['p_value']}")
