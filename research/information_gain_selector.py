"""
Information Gain Selector
-------------------------
Calculates Mutual Information (MI) between analytical engine scores 
and actual outcomes to identify which features contribute real predictive signal.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from typing import Dict, List, Any

class InformationGainSelector:
    """
    Computes MI for engine signals over a rolling window.
    """
    def __init__(self, window_size: int = 100):
        self.window_size = window_size

    def compute_mi_weights(self, engine_history: Dict[str, List[float]], outcomes: List[int]) -> Dict[str, float]:
        """
        Calculates normalized MI weights for each engine.
        X: Historical scores given to the Jodi that actually hit.
        y: The Jodi that hit (0-99).
        """
        if not outcomes or len(outcomes) < 10:
            return {k: 1.0/len(engine_history) for k in engine_history.keys()}

        mi_scores = {}
        # We look at the last 'window_size' samples
        y = np.array(outcomes[-self.window_size:])
        
        for engine_name, scores in engine_history.items():
            # X must be 2D for sklearn
            X = np.array(scores[-self.window_size:]).reshape(-1, 1)
            
            # Discrete target (0-99), Continuous feature (Score)
            # Use small discrete_features=False for continuous scores
            try:
                mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)[0]
                mi_scores[engine_name] = max(mi, 1e-6) # Ensure positive
            except Exception:
                mi_scores[engine_name] = 1e-6

        # Normalize to sum to 1
        total_mi = sum(mi_scores.values())
        normalized_weights = {k: v / total_mi for k, v in mi_scores.items()}
        
        return normalized_weights

if __name__ == "__main__":
    # Quick sanity check with dummy data
    selector = InformationGainSelector(window_size=50)
    outcomes = np.random.randint(0, 100, 100).tolist()
    history = {
        "transition": np.random.rand(100).tolist(),
        "bayesian": np.random.rand(100).tolist(),
        "momentum": np.random.rand(100).tolist()
    }
    weights = selector.compute_mi_weights(history, outcomes)
    print("--- Calculated MI Weights ---")
    for k, v in weights.items():
        print(f"  {k}: {v:.4f}")
