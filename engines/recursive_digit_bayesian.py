"""
Recursive Bayesian Digit Engine
-------------------------------
Hypothesis: Digits (0-9) exhibit 'magnet' behavior where they deviate 
from a uniform distribution. We use KL-Divergence to detect these signals.
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from research.tracker import log_experiment
from typing import Dict, List, Tuple, Any

class RecursiveDigitBayesian:
    def __init__(self, alpha_prior: float = 1.0):
        # 10 digits (0-9)
        self.num_classes = 10
        self.alpha_prior = alpha_prior
        self.counts = np.zeros(self.num_classes)
        self.history_size = 0

    def update(self, jodi: int):
        """Extracts tens and units digits and updates counts."""
        tens = jodi // 10
        units = jodi % 10
        self.counts[tens] += 1
        self.counts[units] += 1
        self.history_size += 2

    def get_kl_divergence(self) -> np.ndarray:
        """
        Calculates KL-Divergence per digit vs a uniform prior.
        D_KL(P || Q) = sum(P(i) * log(P(i) / Q(i)))
        """
        # Posterior P
        p = (self.counts + self.alpha_prior) / (self.history_size + self.num_classes * self.alpha_prior)
        # Uniform Prior Q
        q = np.ones(self.num_classes) / self.num_classes
        
        # Point-wise KL (contribution of each digit to total divergence)
        kl_per_class = p * np.log(p / q)
        return kl_per_class

    def get_magnet_digits(self, threshold: float = 0.01) -> List[int]:
        """Returns digits that are significantly 'strong' (Magnet)."""
        kl = self.get_kl_divergence()
        # We only care about positive KL where P > Q
        magnets = [i for i, val in enumerate(kl) if val > threshold]
        return magnets

    def quick_backtest(self, df: pd.DataFrame, kl_threshold: float = 0.005) -> Dict[str, Any]:
        """
        Backtest: If a digit is a 'Magnet', predict all Jodis containing it.
        """
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total_days = 0
        total_bets = 0
        
        # Reset state
        self.counts = np.zeros(self.num_classes)
        self.history_size = 0
        
        # Warm up with 50 draws
        for i in range(len(jodis)):
            current_jodi = jodis[i]
            
            if i > 50:
                magnets = self.get_magnet_digits(threshold=kl_threshold)
                if magnets:
                    # Predict all Jodis containing any magnet digit
                    predictions = []
                    for m in magnets:
                        # Jodis with magnet in tens place (m0-m9)
                        predictions.extend([m * 10 + x for x in range(10)])
                        # Jodis with magnet in units place (0m-9m)
                        predictions.extend([x * 10 + m for x in range(10)])
                    
                    predictions = list(set(predictions)) # Unique Jodis
                    
                    total_days += 1
                    total_bets += len(predictions)
                    if current_jodi in predictions:
                        hits += 1
            
            self.update(current_jodi)
            
        hit_rate = hits / total_days if total_days > 0 else 0
        # Average number of Jodis predicted per day
        avg_bets = total_bets / total_days if total_days > 0 else 0
        # Random baseline hit rate for the average bet size
        baseline = avg_bets / 100.0
        
        # Z-Score
        std_error = np.sqrt((baseline * (1 - baseline)) / total_days) if total_days > 0 else 0
        z_score = (hit_rate - baseline) / std_error if std_error > 0 else 0
        p_value = 1 - norm.cdf(z_score)

        results = {
            "hit_rate": round(hit_rate, 4),
            "random_baseline": round(baseline, 4),
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "avg_jodis_played": round(avg_bets, 2),
            "total_test_days": total_days
        }
        
        log_experiment("recursive_digit_bayesian", results)
        return results

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    rdb = RecursiveDigitBayesian()
    print("--- Recursive Bayesian Digit Analysis ---")
    stats = rdb.quick_backtest(df, kl_threshold=0.005)
    
    print(f"Avg Jodis Predicted per Day: {stats['avg_jodis_played']}")
    print(f"Hit Rate: {stats['hit_rate']:.2%} (Baseline: {stats['random_baseline']:.2%})")
    print(f"Z-Score: {stats['z_score']}")
    print(f"P-Value: {stats['p_value']}")
