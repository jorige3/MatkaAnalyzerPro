"""
Permutation Importance Test
---------------------------
Measures the 'predictive weight' of each engine by shuffling its signals 
across the timeline. Engines that cause the largest hit-rate drop 
upon shuffling are the primary signal carriers.
"""
import pandas as pd
import numpy as np
from engines.mi_adaptive_ensemble import MIAdaptiveEnsemble, run_backtest_window
from research.information_gain_selector import InformationGainSelector
from engines.frequency import FrequencyEngine
from engines.transition_matrix import TransitionEngine
from engines.bayesian_predictor import BayesianEngine
from engines.momentum import MomentumEngine
from research.tracker import log_experiment
from typing import Dict, List, Any

class PermutationTester:
    def __init__(self, df: pd.DataFrame, window_days: int = 150):
        self.df = df
        self.window_days = window_days
        self.engines = ["transition", "bayesian", "momentum", "frequency"]

    def run_permutation_backtest(self, shuffle_engine: str = None) -> float:
        """
        Runs the MI-Adaptive backtest. If shuffle_engine is provided, 
        its scores are randomized across the history.
        """
        jodis = pd.to_numeric(self.df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total = 0
        selector = InformationGainSelector(window_size=100)
        
        engine_history = {k: [] for k in self.engines}
        outcomes = []

        # Walk-forward
        for i in range(len(jodis) - self.window_days, len(jodis) - 1):
            sub_df = self.df.iloc[:i+1]
            
            # Generate real scores
            fe, te, be, me = FrequencyEngine(), TransitionEngine(), BayesianEngine(), MomentumEngine()
            te.train(sub_df)
            for j in jodis[:i+1]: be.update(j)
            
            f_s = fe.run(sub_df).get(str(jodis[i]).zfill(2), 0) / 100.0
            t_s = te.predict_next(jodis[i-1]).get(jodis[i], 0)
            b_s = be.get_probabilities()[jodis[i]]
            m_s = me.run(sub_df).get(str(jodis[i]).zfill(2), 50) / 200.0
            
            # Record history
            engine_history["frequency"].append(f_s)
            engine_history["transition"].append(t_s)
            engine_history["bayesian"].append(b_s)
            engine_history["momentum"].append(m_s)
            outcomes.append(jodis[i])

            # Apply permutation if requested
            current_mi_history = {k: list(v) for k, v in engine_history.items()}
            if shuffle_engine and len(current_mi_history[shuffle_engine]) > 1:
                # Shuffle the history of the target engine to destroy its temporal link
                np.random.shuffle(current_mi_history[shuffle_engine])

            mi_w = selector.compute_mi_weights(current_mi_history, outcomes)
            ensemble = MIAdaptiveEnsemble(sub_df, mi_w)
            preds, _ = ensemble.predict_next(top_n=10)
            
            if str(jodis[i+1]).zfill(2) in [p[0] for p in preds]:
                hits += 1
            total += 1

        return hits / total if total > 0 else 0

    def execute_test(self, iterations: int = 5):
        print("--- Establishing Baseline ---")
        baseline_hr = self.run_permutation_backtest(shuffle_engine=None)
        print(f"Baseline Hit Rate: {baseline_hr:.2%}\n")

        results = []
        for engine in self.engines:
            print(f"Testing Importance: {engine}...")
            iters = []
            for _ in range(iterations):
                hr = self.run_permutation_backtest(shuffle_engine=engine)
                iters.append(hr)
            
            avg_hr = np.mean(iters)
            drop = baseline_hr - avg_hr
            results.append({
                "Engine": engine,
                "Hit Rate After Shuffle": f"{avg_hr:.2%}",
                "Performance Drop": f"{drop:.2%}",
                "raw_drop": drop
            })

        # Sort by drop (highest drop = most important)
        results = sorted(results, key=lambda x: x['raw_drop'], reverse=True)
        
        print("\nPERMUTATION IMPORTANCE RESULTS")
        print(f"{'Engine':<20} | {'Hit Rate (Shuffled)':<20} | {'Drop'}")
        print("-" * 65)
        for r in results:
            print(f"{r['Engine']:<20} | {r['Hit Rate After Shuffle']:<20} | {r['Performance Drop']}")

        log_experiment("permutation_importance", results)
        
        top_engine = results[0]['Engine']
        print(f"\nCONCLUSION: The '{top_engine}' engine is the primary signal carrier.")
        if results[0]['raw_drop'] < 0.01:
            print("WARNING: No single engine shows strong dominance. Predictive edge may be fragile.")
        else:
            print(f"STABILITY: The ensemble depends significantly on {top_engine}.")

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    tester = PermutationTester(df, window_days=150)
    # Using 3 iterations for research speed; increase for final validation
    tester.execute_test(iterations=3)
