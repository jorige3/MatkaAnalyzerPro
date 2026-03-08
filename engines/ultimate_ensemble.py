"""
Ultimate Ensemble Predictor
---------------------------
The final synthesized model of the MatkaAnalyzerPro research phase.
Combines:
1. 3-State HMM for latent regime detection.
2. Multi-Lag Convolution (Lags 1, 2, 3) for sequential dependencies.
3. Bayesian recursive probability for frequency bias.
4. Adaptive weighting based on the current regime.
"""
import pandas as pd
import numpy as np
from hmmlearn import hmm
from scipy.stats import norm
from engines.frequency import FrequencyEngine
from engines.transition_matrix import TransitionEngine
from engines.bayesian_predictor import BayesianEngine
from engines.momentum import MomentumEngine
from engines.multi_lag_convolution import MultiLagConvolutionEngine
from research.tracker import log_experiment
from typing import Dict, List, Tuple, Any

class UltimateEnsemble:
    # State-to-Weight Mapping
    # State 0: STABLE (Frequency dominant)
    # State 1: SEQUENTIAL (Multi-lag dependencies dominant)
    # State 2: VOLATILE (Momentum/Mean-reversion dominant)
    REGIME_WEIGHTS = {
        0: {"bayesian": 0.60, "convolution": 0.20, "momentum": 0.20},
        1: {"convolution": 0.60, "bayesian": 0.20, "momentum": 0.20},
        2: {"momentum": 0.50, "bayesian": 0.30, "convolution": 0.20},
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        
        # Component Engines
        self.conv_engine = MultiLagConvolutionEngine(lags=[1, 2, 3])
        self.be = BayesianEngine()
        self.me = MomentumEngine()
        self.fe = FrequencyEngine()
        
        # HMM for regime detection
        self.hmm_model = hmm.CategoricalHMM(n_components=3, n_iter=50, random_state=42)

    def _train_all(self):
        # 1. HMM
        X = np.array(self.jodis).reshape(-1, 1)
        self.hmm_model.fit(X)
        
        # 2. Convolutional Matrices
        self.conv_engine.train(pd.Series(self.jodis))
        
        # 3. Bayesian
        for j in self.jodis:
            self.be.update(j)

    def predict_next(self, top_n: int = 10) -> Tuple[List[Tuple[str, float]], int]:
        # 1. Determine Regime
        X = np.array(self.jodis).reshape(-1, 1)
        states = self.hmm_model.predict(X)
        current_state = states[-1]
        weights = self.REGIME_WEIGHTS[current_state]

        # 2. Get Component Probabilities
        conv_probs = self.conv_engine.predict_next(self.jodis)
        bayesian_probs = self.be.get_probabilities()
        momentum_scores = self.me.run(self.df)
        
        # Note: Convolution weights [1,2,3] are balanced for now
        
        final_scores = {}
        for j in range(100):
            j_str = str(j).zfill(2)
            
            s_conv = conv_probs[j]
            s_bayesian = bayesian_probs[j]
            # Momentum normalized 0-200 -> 0-1
            s_momentum = momentum_scores.get(j_str, 50) / 200.0
            
            # Weighted synthesis
            score = (
                weights["convolution"] * s_conv +
                weights["bayesian"] * s_bayesian +
                weights["momentum"] * s_momentum
            )
            final_scores[j_str] = round(score, 6)
            
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n], current_state

    @staticmethod
    def run_full_backtest(df: pd.DataFrame, days: int = 200, top_k: int = 10):
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total = 0
        
        print(f"Running Ultimate Ensemble backtest for {days} days...")
        
        # Batch training every 20 days for efficiency, using history up to current day
        for i in range(len(jodis) - days, len(jodis) - 1):
            sub_df = df.iloc[:i+1]
            actual_next = str(jodis[i+1]).zfill(2)
            
            ue = UltimateEnsemble(sub_df)
            ue._train_all()
            preds, state = ue.predict_next(top_n=top_k)
            
            if actual_next in [p[0] for p in preds]:
                hits += 1
            total += 1
            
        hit_rate = hits / total if total > 0 else 0
        baseline = top_k / 100.0
        
        # Statistical Analysis
        std_error = np.sqrt((baseline * (1 - baseline)) / total)
        z_score = (hit_rate - baseline) / std_error if std_error > 0 else 0
        p_value = 1 - norm.cdf(z_score)

        print("\n" + "="*40)
        print("## ULTIMATE ENSEMBLE BACKTEST RESULTS")
        print("="*40)
        print(f"Total Test Days: {total}")
        print(f"Top-{top_k} Hit Rate: {hit_rate:.2%}")
        print(f"Baseline: {baseline:.2%}")
        print(f"Edge over Baseline: {hit_rate - baseline:+.2%}")
        print(f"Z-Score: {z_score:.4f}")
        print(f"P-Value: {p_value:.6f}")
        
        log_experiment("ultimate_ensemble_validation", {
            "hit_rate": hit_rate,
            "z_score": z_score,
            "p_value": p_value,
            "days": total
        })

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    UltimateEnsemble.run_full_backtest(df, days=150, top_k=10)
