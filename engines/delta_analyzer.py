"""
Delta Analyzer Engine
---------------------
Hypothesis: The modular difference between consecutive draws (Deltas) is 
non-uniformly distributed, suggesting "step-wise" patterns or inertia.
"""
import pandas as pd
import numpy as np
from scipy.stats import chisquare, norm
from typing import Dict, Any

class DeltaAnalyzer:
    """
    Analyzes modular differences between consecutive Jodis.
    """
    def __init__(self, num_classes: int = 100):
        self.num_classes = num_classes

    def _get_deltas(self, series: pd.Series) -> pd.Series:
        # Calculate modular difference: (X_t - X_{t-1}) % 100
        return series.diff().dropna().mod(self.num_classes).astype(int)

    def run_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int)
        deltas = self._get_deltas(jodis)
        
        # 1. Chi-Square Test for Uniformity
        observed_freq = deltas.value_counts().reindex(range(self.num_classes), fill_value=0)
        expected_freq = np.ones(self.num_classes) * (len(deltas) / self.num_classes)
        chi_stat, p_val = chisquare(observed_freq, expected_freq)
        
        # 2. Entropy of Deltas
        probs = observed_freq / len(deltas)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        max_entropy = np.log2(self.num_classes)
        
        return {
            "chi_stat": chi_stat,
            "p_value": p_val,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "top_deltas": observed_freq.sort_values(ascending=False).head(10).index.tolist()
        }

    def quick_backtest(self, df: pd.DataFrame, top_k: int = 10) -> Dict[str, Any]:
        """
        Walk-forward backtest using top-K historical deltas.
        """
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total = 0
        
        # We need at least 100 points to establish a delta frequency
        for i in range(100, len(jodis) - 1):
            historical_jodis = pd.Series(jodis[:i+1])
            historical_deltas = self._get_deltas(historical_jodis)
            
            # Find top deltas in history
            top_deltas = historical_deltas.value_counts().head(top_k).index.tolist()
            
            # Predict: Next = (Current + Delta) % 100
            current_jodi = jodis[i]
            predictions = [(current_jodi + d) % self.num_classes for d in top_deltas]
            
            actual_next = jodis[i+1]
            if actual_next in predictions:
                hits += 1
            total += 1
            
        hit_rate = hits / total if total > 0 else 0
        expected_prob = top_k / self.num_classes
        
        # Z-Score for hit rate
        std_error = np.sqrt((expected_prob * (1 - expected_prob)) / total)
        z_score = (hit_rate - expected_prob) / std_error if std_error > 0 else 0
        p_value = 1 - norm.cdf(z_score)

        return {
            "hit_rate": round(hit_rate, 4),
            "expected_prob": expected_prob,
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "total_samples": total,
            "hits": hits
        }

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    analyzer = DeltaAnalyzer()
    print("--- Delta Analysis Metrics ---")
    stats = analyzer.run_analysis(df)
    print(f"Chi-Square P-Value: {stats['p_value']:.6f}")
    print(f"Entropy: {stats['entropy']:.4f} / {stats['max_entropy']:.4f}")
    
    print("\n--- Delta Backtest (Top-10) ---")
    bt = analyzer.quick_backtest(df, top_k=10)
    print(f"Hit Rate: {bt['hit_rate']:.2%} (Baseline: {bt['expected_prob']:.2%})")
    print(f"Z-Score: {bt['z_score']}")
    print(f"P-Value: {bt['p_value']}")
