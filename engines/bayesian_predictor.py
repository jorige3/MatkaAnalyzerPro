"""
Bayesian Predictor Engine
-------------------------
Updates probabilities using a Bayesian approach with a Dirichlet-Multinomial 
model. Starts with a uniform prior and updates as data arrives.
"""
import pandas as pd
import numpy as np
from typing import Dict, List

class BayesianEngine:
    """
    Bayesian probability updater for Jodi occurrences.
    """

    def __init__(self, num_classes: int = 100, alpha_prior: float = 1.0):
        """
        alpha_prior: strength of the prior (1.0 = neutral/Laplace smoothing)
        """
        self.num_classes = num_classes
        self.alpha_prior = alpha_prior
        self.counts = np.zeros(num_classes)

    def update(self, jodi: int):
        """Updates counts with a new observation."""
        if 0 <= jodi < self.num_classes:
            self.counts[jodi] += 1

    def get_probabilities(self) -> np.ndarray:
        """Calculates the posterior predictive distribution."""
        # Posterior = Counts + Alpha_Prior
        posterior_alphas = self.counts + self.alpha_prior
        return posterior_alphas / posterior_alphas.sum()

    def quick_backtest(self, df: pd.DataFrame, top_k: int = 10) -> Dict[str, float]:
        """Tests the Bayesian updater's performance."""
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total = 0
        
        # Reset counts for backtest
        self.counts = np.zeros(self.num_classes)
        
        # Start after 50 samples to build a basic posterior
        for i, val in enumerate(jodis):
            if i >= 50:
                probs = self.get_probabilities()
                top_preds = np.argsort(probs)[-top_k:]
                
                if val in top_preds:
                    hits += 1
                total += 1
            
            self.update(val)
            
        accuracy = hits / total if total > 0 else 0
        baseline = top_k / self.num_classes
        return {
            "accuracy": round(accuracy, 4),
            "baseline": baseline,
            "edge": round(accuracy - baseline, 4),
            "total_samples": total
        }

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    engine = BayesianEngine()
    print("--- Bayesian Predictor Quick Backtest ---")
    results = engine.quick_backtest(df, top_k=10)
    print(f"Top-10 Accuracy: {results['accuracy']:.2%} (Baseline: {results['baseline']:.2%})")
    print(f"Edge: {results['edge']:.2%}")
