"""
Distance Autocorrelation Analyzer
---------------------------------
Hypothesis: The absolute distance between consecutive draws |X_t - X_{t-1}| 
exhibits autocorrelation. If the system is 'moving' in small steps, it 
will continue to do so (Inertia).
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from research.tracker import log_experiment
from typing import Dict, Any

class DistanceAutocorrelation:
    """
    Analyzes the 'velocity' of the Jodi sequence.
    """
    def __init__(self, num_classes: int = 100):
        self.num_classes = num_classes

    def _get_distances(self, series: pd.Series) -> pd.Series:
        # We use absolute difference as a measure of 'jump size'
        return series.diff().abs().dropna()

    def run_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int)
        distances = self._get_distances(jodis)
        
        # Calculate Autocorrelation for lag 1 to 5
        acf_values = [distances.autocorr(lag=i) for i in range(1, 6)]
        
        return {
            "acf_lag1": round(acf_values[0], 4),
            "acf_lag2": round(acf_values[1], 4),
            "mean_distance": round(distances.mean(), 2),
            "std_distance": round(distances.std(), 2)
        }

    def quick_backtest(self, df: pd.DataFrame, window_size: int = 10) -> Dict[str, Any]:
        """
        Backtest: If ACF is positive, predict that the next Jodi will be within 
        the 'Mean Distance' of the previous Jodi.
        """
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total_days = 0
        total_bets = 0
        
        for i in range(100, len(jodis) - 1):
            prev_jodi = jodis[i]
            actual_next = jodis[i+1]
            
            # Historical context for mean distance
            hist_dist = self._get_distances(pd.Series(jodis[:i+1]))
            avg_dist = hist_dist.mean()
            
            # Strategy: Predict all Jodis within 1/2 of Mean Distance 
            # (Capturing the 'Low Volatility' cluster)
            radius = int(avg_dist / 2)
            if radius < 5: radius = 5 # Minimum search area
            
            predictions = []
            for d in range(-radius, radius + 1):
                predictions.append((prev_jodi + d) % self.num_classes)
            
            predictions = list(set(predictions))
            
            total_days += 1
            total_bets += len(predictions)
            if actual_next in predictions:
                hits += 1
                
        hit_rate = hits / total_days if total_days > 0 else 0
        avg_bet_size = total_bets / total_days if total_days > 0 else 0
        baseline = avg_bet_size / 100.0
        
        # Significance
        std_error = np.sqrt((baseline * (1 - baseline)) / total_days) if total_days > 0 else 0
        z_score = (hit_rate - baseline) / std_error if std_error > 0 else 0
        p_value = 1 - norm.cdf(z_score)

        results = {
            "hit_rate": round(hit_rate, 4),
            "baseline": round(baseline, 4),
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "avg_radius": radius,
            "total_days": total_days
        }
        
        log_experiment("distance_autocorrelation", results)
        return results

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    da = DistanceAutocorrelation()
    print("--- Distance Autocorrelation Analysis ---")
    stats = da.run_analysis(df)
    print(f"ACF Lag-1: {stats['acf_lag1']}")
    print(f"Mean Jump Size: {stats['mean_distance']}")
    
    print("\n--- 'Low Velocity' Cluster Backtest ---")
    bt = da.quick_backtest(df)
    print(f"Hit Rate: {bt['hit_rate']:.2%} (Baseline: {bt['baseline']:.2%})")
    print(f"Z-Score: {bt['z_score']}")
    print(f"P-Value: {bt['p_value']}")
