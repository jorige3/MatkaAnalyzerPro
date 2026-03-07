"""
Transition Matrix Engine (Markov Chain Analysis)
-----------------------------------------------
Calculates the probability of Jodi B appearing given that Jodi A appeared 
in the previous draw. This identifies sequential dependencies.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class TransitionEngine:
    """
    Analyzes first-order Markov transitions between Jodis.
    """

    def __init__(self, num_classes: int = 100):
        self.num_classes = num_classes
        self.transition_matrix = np.zeros((num_classes, num_classes))

    def train(self, df: pd.DataFrame):
        """Builds the transition matrix from historical data."""
        # Ensure numeric jodis
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        
        for i in range(len(jodis) - 1):
            state_from = jodis[i]
            state_to = jodis[i+1]
            if 0 <= state_from < self.num_classes and 0 <= state_to < self.num_classes:
                self.transition_matrix[state_from, state_to] += 1
        
        # Normalize rows to get probabilities
        row_sums = self.transition_matrix.sum(axis=1)
        # Avoid division by zero
        self.transition_matrix = np.divide(self.transition_matrix, row_sums[:, np.newaxis], 
                                         where=row_sums[:, np.newaxis] != 0)

    def predict_next(self, last_jodi: int) -> Dict[int, float]:
        """Returns the probability distribution for the next Jodi given the last one."""
        if 0 <= last_jodi < self.num_classes:
            probs = self.transition_matrix[last_jodi]
            return {i: probs[i] for i in range(self.num_classes) if probs[i] > 0}
        return {}

    def get_top_transitions(self, top_n: int = 10) -> list:
        """Finds the strongest transitions overall (A -> B)."""
        flat_indices = np.argsort(self.transition_matrix.ravel())[::-1][:top_n]
        results = []
        for idx in flat_indices:
            from_state, to_state = divmod(idx, self.num_classes)
            prob = self.transition_matrix[from_state, to_state]
            if prob > 0:
                results.append((from_state, to_state, prob))
        return results

    def quick_backtest(self, df: pd.DataFrame, top_k: int = 5) -> Dict[str, float]:
        """Tests if the top K predicted transitions hit above random baseline."""
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total = 0
        
        # Simple walk-forward: train on all data before 'i'
        # For speed in quick_backtest, we'll use a expanding window but update matrix incrementally
        current_matrix = np.zeros((self.num_classes, self.num_classes))
        
        for i in range(100, len(jodis) - 1):
            # Update matrix with the transition that just happened (i-1 to i)
            s_from, s_to = jodis[i-1], jodis[i]
            current_matrix[s_from, s_to] += 1
            
            # Predict for i+1
            row = current_matrix[s_to]
            if row.sum() > 0:
                probs = row / row.sum()
                top_preds = np.argsort(probs)[-top_k:]
                
                if jodis[i+1] in top_preds:
                    hits += 1
                total += 1
        
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
    
    engine = TransitionEngine()
    print("--- Transition Matrix Quick Backtest ---")
    results = engine.quick_backtest(df, top_k=10)
    print(f"Top-10 Accuracy: {results['accuracy']:.2%} (Baseline: {results['baseline']:.2%})")
    print(f"Edge: {results['edge']:.2%}")
