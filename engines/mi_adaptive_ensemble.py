"""
MI-Adaptive Ensemble Predictor
------------------------------
Combines HMM regime detection with Information Gain (MI) feature selection.
Final weights are a product of regime-specific importance and historical 
predictive value (Mutual Information).
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from hmmlearn import hmm
from engines.frequency import FrequencyEngine
from engines.transition_matrix import TransitionEngine
from engines.bayesian_predictor import BayesianEngine
from engines.momentum import MomentumEngine
from research.information_gain_selector import InformationGainSelector
from research.tracker import log_experiment
from typing import Dict, List, Tuple, Any

class MIAdaptiveEnsemble:
    # Baseline weights for HMM states
    HMM_WEIGHTS = {
        0: {"bayesian": 0.50, "transition": 0.20, "momentum": 0.20, "frequency": 0.10}, # Stable
        1: {"transition": 0.50, "bayesian": 0.20, "momentum": 0.20, "frequency": 0.10}, # Sequential
        2: {"momentum": 0.40, "bayesian": 0.30, "transition": 0.10, "frequency": 0.20}, # Volatile
    }

    def __init__(self, df: pd.DataFrame, mi_weights: Dict[str, float]):
        self.df = df
        self.mi_weights = mi_weights
        self.jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        
        # Engines
        self.fe = FrequencyEngine()
        self.te = TransitionEngine()
        self.be = BayesianEngine()
        self.me = MomentumEngine()
        
        # Pre-train HMM
        self.hmm_model = hmm.CategoricalHMM(n_components=3, n_iter=50, random_state=42)
        X = np.array(self.jodis).reshape(-1, 1)
        self.hmm_model.fit(X)
        
        # Train others
        self.te.train(df)
        for j in self.jodis: self.be.update(j)

    def predict_next(self, top_n: int = 10) -> Tuple[List[Tuple[str, float]], int]:
        # 1. HMM State
        X = np.array(self.jodis).reshape(-1, 1)
        states = self.hmm_model.predict(X)
        current_state = states[-1]
        hmm_w = self.HMM_WEIGHTS[current_state]

        # 2. Get scores
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
            
            # Combine: HMM_Weight * Score * MI_Weight
            # Logic: We use HMM to define the 'type' of signal we want, 
            # and MI to verify if that engine is actually working.
            score = (
                hmm_w.get("transition", 0) * s_markov * self.mi_weights.get("transition", 1.0) +
                hmm_w.get("bayesian", 0) * s_bayesian * self.mi_weights.get("bayesian", 1.0) +
                hmm_w.get("momentum", 0) * s_momentum * self.mi_weights.get("momentum", 1.0) +
                hmm_w.get("frequency", 0) * s_freq * self.mi_weights.get("frequency", 1.0)
            )
            final_scores[j_str] = round(score, 6)
            
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n], current_state

def run_backtest_window(df: pd.DataFrame, window_days: int, top_k: int = 10):
    jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
    if len(jodis) < window_days + 150:
        return None

    hits = 0
    total = 0
    selector = InformationGainSelector(window_size=100)
    
    # Track historical correct scores for MI
    engine_history = {
        "transition": [], "bayesian": [], "momentum": [], "frequency": []
    }
    outcomes = []

    # Iterate walk-forward
    for i in range(len(jodis) - window_days, len(jodis) - 1):
        sub_df = df.iloc[:i+1]
        actual_next = jodis[i+1]
        
        # 1. Update Engine History (needed for MI calculation)
        # We simulate what each engine gave to the 'correct' Jodi yesterday
        # This is a proxy for how well the engine is aligning with reality
        fe = FrequencyEngine()
        te = TransitionEngine()
        be = BayesianEngine()
        me = MomentumEngine()
        
        te.train(sub_df)
        for j in jodis[:i+1]: be.update(j)
        
        f_s = fe.run(sub_df).get(str(jodis[i]).zfill(2), 0) / 100.0
        t_s = te.predict_next(jodis[i-1]).get(jodis[i], 0)
        b_s = be.get_probabilities()[jodis[i]]
        m_s = me.run(sub_df).get(str(jodis[i]).zfill(2), 50) / 200.0
        
        engine_history["frequency"].append(f_s)
        engine_history["transition"].append(t_s)
        engine_history["bayesian"].append(b_s)
        engine_history["momentum"].append(m_s)
        outcomes.append(jodis[i])

        # 2. Compute MI Weights
        mi_w = selector.compute_mi_weights(engine_history, outcomes)
        
        # 3. Predict
        ensemble = MIAdaptiveEnsemble(sub_df, mi_w)
        preds, _ = ensemble.predict_next(top_n=top_k)
        
        if str(actual_next).zfill(2) in [p[0] for p in preds]:
            hits += 1
        total += 1

    return hits, total

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    windows = [150, 300, 500]
    results_map = {}

    print("## MI-ADAPTIVE ENSEMBLE BACKTEST\n")
    for w in windows:
        res = run_backtest_window(df, w)
        if res:
            hits, total = res
            hr = hits / total
            results_map[w] = hr
            print(f"WINDOW {w} days: {hr:.2%}")
        else:
            print(f"WINDOW {w} days: Not enough data")

    # Stats for the primary window (150)
    w150 = results_map.get(150, 0)
    baseline = 0.10
    n = 150
    std_err = np.sqrt((baseline * (1 - baseline)) / n)
    z = (w150 - baseline) / std_err
    p = 1 - norm.cdf(z)

    print(f"\nSummary (150 days):")
    print(f"Top-10 Hit Rate: {w150:.2%}")
    print(f"Baseline: {baseline:.2%}")
    print(f"Z-Score: {z:.4f}")
    print(f"P-Value: {p:.6f}")

    log_experiment("mi_adaptive_ensemble", {"hit_rate_150": w150, "p_value": p, "z_score": z})
