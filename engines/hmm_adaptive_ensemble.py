"""
HMM-Adaptive Ensemble Predictor
-------------------------------
Hypothesis: Latent states from a 3-state HMM provide a more accurate 
trigger for weight-switching than rolling entropy. 
States mapped:
- State 0: 'STABLE' -> Focus on Bayesian Frequency
- State 1: 'SEQUENTIAL' -> Focus on Markov Transitions
- State 2: 'VOLATILE' -> Focus on Momentum/Random
"""
import pandas as pd
import numpy as np
from hmmlearn import hmm
from scipy.stats import norm
from engines.frequency import FrequencyEngine
from engines.transition_matrix import TransitionEngine
from engines.bayesian_predictor import BayesianEngine
from engines.momentum import MomentumEngine
from research.tracker import log_experiment
from typing import Dict, List, Tuple, Any

class HMMAdaptiveEnsemble:
    # State-to-Weight Mapping
    STATE_WEIGHTS = {
        0: {"bayesian": 0.50, "transition": 0.20, "momentum": 0.20, "baseline": 0.10}, # Stable
        1: {"transition": 0.50, "bayesian": 0.20, "momentum": 0.20, "baseline": 0.10}, # Sequential
        2: {"momentum": 0.40, "bayesian": 0.30, "transition": 0.10, "baseline": 0.20}, # Volatile
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        
        # Engines
        self.fe = FrequencyEngine()
        self.te = TransitionEngine()
        self.be = BayesianEngine()
        self.me = MomentumEngine()
        
        # HMM setup
        self.hmm_model = hmm.CategoricalHMM(n_components=3, n_iter=100, random_state=42)
        
    def _prepare_models(self):
        # Train HMM on all history
        X = np.array(self.jodis).reshape(-1, 1)
        self.hmm_model.fit(X)
        
        # Train Markov
        self.te.train(self.df)
        
        # Bayesian
        for j in self.jodis:
            self.be.update(j)

    def predict_next(self, top_n: int = 10) -> Tuple[List[Tuple[str, float]], int]:
        # 1. Decode current state using Viterbi
        X = np.array(self.jodis).reshape(-1, 1)
        states = self.hmm_model.predict(X)
        current_state = states[-1]
        
        weights = self.STATE_WEIGHTS[current_state]

        # 2. Get scores from all engines
        freq_scores = self.fe.run(self.df)
        last_jodi = self.jodis[-1]
        markov_probs = self.te.predict_next(last_jodi)
        bayesian_probs = self.be.get_probabilities()
        momentum_scores = self.me.run(self.df)

        final_scores = {}
        for j in range(100):
            j_str = str(j).zfill(2)
            
            s_freq = freq_scores.get(j_str, 0) / 100.0
            s_markov = markov_probs.get(j, 0)
            s_bayesian = bayesian_probs[j]
            s_momentum = momentum_scores.get(j_str, 50) / 200.0
            
            score = (
                weights.get("transition", 0) * s_markov +
                weights.get("bayesian", 0) * s_bayesian +
                weights.get("momentum", 0) * s_momentum +
                weights.get("baseline", 0) * 0.01 # Flat prior
            )
            final_scores[j_str] = round(score, 4)
            
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n], current_state

    @staticmethod
    def run_backtest(df: pd.DataFrame, days: int = 200, top_k: int = 10):
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total = 0
        
        print(f"Running HMM-Adaptive backtest for {days} days...")
        
        for i in range(len(jodis) - days, len(jodis) - 1):
            sub_df = df.iloc[:i+1]
            actual_next = str(jodis[i+1]).zfill(2)
            
            # Setup and predict
            hae = HMMAdaptiveEnsemble(sub_df)
            hae._prepare_models()
            preds, state = hae.predict_next(top_n=top_k)
            
            if actual_next in [p[0] for p in preds]:
                hits += 1
            total += 1
            
        hit_rate = hits / total if total > 0 else 0
        baseline = top_k / 100.0
        
        std_error = np.sqrt((baseline * (1 - baseline)) / total)
        z_score = (hit_rate - baseline) / std_error if std_error > 0 else 0
        p_value = 1 - norm.cdf(z_score)

        print("\n" + "="*40)
        print("## HMM-ADAPTIVE ENSEMBLE BACKTEST")
        print("="*40)
        print(f"Days Tested: {total}")
        print(f"Top-{top_k} Hit Rate: {hit_rate:.2%}")
        print(f"Baseline: {baseline:.2%}")
        print(f"Z-Score: {z_score:.4f}")
        print(f"P-Value: {p_value:.6f}")
        
        log_experiment("hmm_adaptive_ensemble", {
            "hit_rate": hit_rate,
            "z_score": z_score,
            "samples": total,
            "top_k": top_k
        })

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    HMMAdaptiveEnsemble.run_backtest(df, days=150, top_k=10)
