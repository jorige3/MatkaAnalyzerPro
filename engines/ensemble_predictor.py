"""
Ensemble Predictor Engine
-------------------------
Combines multiple weak signals into a single, robust confidence score.
Weights:
- Transition Prob (Markov): 0.30
- Frequency Bias: 0.20
- Bayesian Prob: 0.20
- Momentum Score: 0.15
- Regime Weight: 0.15
"""
import pandas as pd
import numpy as np
from engines.frequency import FrequencyEngine
from engines.transition_matrix import TransitionEngine
from engines.bayesian_predictor import BayesianEngine
from engines.momentum import MomentumEngine
from engines.regime_detector import RegimeDetector
from research.tracker import log_experiment
from typing import Dict, List, Tuple, Any

class EnsemblePredictor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.fe = FrequencyEngine()
        self.te = TransitionEngine()
        self.be = BayesianEngine()
        self.me = MomentumEngine()
        self.rd = RegimeDetector()
        
        # Pre-train/setup
        self.te.train(df)

    def predict_next(self, top_n: int = 10) -> List[Tuple[str, float]]:
        # 1. Get component scores
        freq_scores = self.fe.run(self.df)
        last_jodi = int(self.df.iloc[-1]['Jodi'])
        markov_probs = self.te.predict_next(last_jodi)
        bayesian_probs = self.be.get_probabilities()
        momentum_scores = self.me.run(self.df)
        regime = self.rd.detect_regime(self.df)
        
        # Adjust momentum by regime
        regime_weight = 1.0
        if regime == "PATTERNED":
            regime_weight = 1.2
        elif regime == "RANDOM":
            regime_weight = 0.8

        final_scores = {}
        for j in range(100):
            j_str = str(j).zfill(2)
            
            s_freq = freq_scores.get(j_str, 0) / 100.0
            s_markov = markov_probs.get(j, 0)
            s_bayesian = bayesian_probs[j]
            s_momentum = momentum_scores.get(j_str, 50) / 200.0
            
            # Weighted average
            total_score = (
                0.30 * s_markov +
                0.20 * s_freq +
                0.20 * s_bayesian +
                0.15 * s_momentum +
                0.15 * (regime_weight * 0.5) # simple contribution from regime
            )
            final_scores[j_str] = round(total_score, 4)
            
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def quick_backtest(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Walking backtest for the ensemble.
        """
        jodis = pd.to_numeric(self.df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total = 0
        
        # Step through last 200 days
        for i in range(len(jodis)-200, len(jodis)-1):
            sub_df = self.df.iloc[:i+1]
            ep = EnsemblePredictor(sub_df)
            preds = [j for j, s in ep.predict_next(top_n=top_k)]
            
            actual = str(jodis[i+1]).zfill(2)
            if actual in preds:
                hits += 1
            total += 1
            
        hit_rate = hits / total if total > 0 else 0
        log_experiment("ensemble_predictor", {"hit_rate": hit_rate, "top_k": top_k, "samples": total})
        return {"hit_rate": hit_rate, "total": total}

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    ensemble = EnsemblePredictor(df)
    print("--- Ensemble Predictor Backtest (Last 200 Days) ---")
    results = ensemble.quick_backtest(top_k=10)
    print(f"Ensemble Hit Rate: {results['hit_rate']:.2%} (Baseline: 10.00%)")
