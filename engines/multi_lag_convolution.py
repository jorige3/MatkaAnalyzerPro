"""
Multi-Lag Convolutional Engine
------------------------------
Hypothesis: Prediction signal is distributed across multiple lags.
This engine sums transition probabilities from Lags 1, 2, and 3, 
weighted by their relative Information Gain (Mutual Information).
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from research.tracker import log_experiment
from typing import Dict, List, Any, Tuple

class MultiLagConvolutionEngine:
    def __init__(self, lags: List[int] = [1, 2, 3], num_classes: int = 100):
        self.lags = lags
        self.num_classes = num_classes
        # Separate transition matrix for each lag
        self.matrices = {lag: np.zeros((num_classes, num_classes)) for lag in lags}
        # Weights will be calculated dynamically or set based on research
        self.weights = {lag: 1.0/len(lags) for lag in lags}

    def train(self, series: pd.Series):
        """Builds all lagged transition matrices."""
        jodis = series.tolist()
        for lag in self.lags:
            matrix = np.zeros((self.num_classes, self.num_classes))
            for i in range(lag, len(jodis)):
                s_from, s_to = jodis[i - lag], jodis[i]
                matrix[s_from, s_to] += 1
            
            # Normalize rows
            row_sums = matrix.sum(axis=1)
            self.matrices[lag] = np.divide(matrix, row_sums[:, np.newaxis], 
                                         where=row_sums[:, np.newaxis] != 0)

    def set_mi_weights(self, mi_scores: Dict[int, float]):
        """Sets weights based on Mutual Information scores."""
        total_mi = sum(mi_scores.get(lag, 0) for lag in self.lags)
        if total_mi > 0:
            self.weights = {lag: mi_scores.get(lag, 0) / total_mi for lag in self.lags}

    def predict_next(self, recent_history: List[int]) -> np.ndarray:
        """
        Calculates the combined probability distribution.
        P_final = sum( weight_lag * P(X_next | X_{now - lag + 1}) )
        """
        combined_probs = np.zeros(self.num_classes)
        
        for lag in self.lags:
            # The source for Lag N is at index [-lag]
            if len(recent_history) >= lag:
                source_val = recent_history[-lag]
                lag_probs = self.matrices[lag][source_val]
                combined_probs += self.weights[lag] * lag_probs
                
        return combined_probs

    def quick_backtest(self, df: pd.DataFrame, top_k: int = 10) -> Dict[str, Any]:
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total = 0
        
        # Minimum history
        start_idx = 300
        
        for i in range(start_idx, len(jodis) - 1):
            sub_series = pd.Series(jodis[:i+1])
            
            # 1. Update Engine (Retrain matrices)
            self.train(sub_series)
            
            # 2. Get combined predictions
            # We pass the history up to day 'i' to predict day 'i+1'
            probs = self.predict_next(jodis[:i+1])
            top_preds = np.argsort(probs)[-top_k:]
            
            actual_next = jodis[i+1]
            if actual_next in top_preds:
                hits += 1
            total += 1
            
        hit_rate = hits / total if total > 0 else 0
        baseline = top_k / 100.0
        
        # Significance
        std_error = np.sqrt((baseline * (1 - baseline)) / total) if total > 0 else 0
        z_score = (hit_rate - baseline) / std_error if std_error > 0 else 0
        p_value = 1 - norm.cdf(z_score)

        results = {
            "hit_rate": round(hit_rate, 4),
            "baseline": round(baseline, 4),
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "weights": self.weights,
            "total_days": total
        }
        
        log_experiment("multi_lag_convolution", results)
        return results

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    # Based on research results:
    # Full Jodi MI Lags 1-3: [2.103, 2.128, 2.119]
    research_mi = {1: 2.103, 2: 2.128, 3: 2.119}
    
    engine = MultiLagConvolutionEngine(lags=[1, 2, 3])
    engine.set_mi_weights(research_mi)
    
    print("--- Multi-Lag Convolution Backtest ---")
    print(f"Weights used: {engine.weights}")
    stats = engine.quick_backtest(df, top_k=10)
    
    print(f"Hit Rate: {stats['hit_rate']:.2%} (Baseline: 10.00%)")
    print(f"Z-Score: {stats['z_score']}")
    print(f"P-Value: {stats['p_value']}")
