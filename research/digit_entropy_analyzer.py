"""
Digit Entropy & Joint Distribution Analyzer
-------------------------------------------
Hypothesis: The randomness of Jodi draws is constrained. We analyze the 
Joint Entropy H(Tens, Units) and Conditional Entropy H(Units | Tens) 
to find gaps in the digit-pair distribution.
"""
import pandas as pd
import numpy as np
from scipy.stats import norm, ks_2samp
from research.tracker import log_experiment
from typing import Dict, Any, Tuple

class DigitEntropyAnalyzer:
    def __init__(self):
        self.num_digits = 10

    def _get_digits(self, series: pd.Series) -> pd.DataFrame:
        s = series.astype(str).str.zfill(2)
        df = pd.DataFrame()
        df['tens'] = s.str[0].astype(int)
        df['units'] = s.str[1].astype(int)
        return df

    def calculate_entropy(self, df_digits: pd.DataFrame) -> Dict[str, float]:
        # Joint Probability P(T, U)
        joint_counts = df_digits.groupby(['tens', 'units']).size()
        p_joint = joint_counts / len(df_digits)
        h_joint = -np.sum(p_joint * np.log2(p_joint))

        # Marginal Probabilities
        p_tens = df_digits['tens'].value_counts(normalize=True)
        h_tens = -np.sum(p_tens * np.log2(p_tens))

        p_units = df_digits['units'].value_counts(normalize=True)
        h_units = -np.sum(p_units * np.log2(p_units))

        # Conditional Entropy H(U|T) = H(T, U) - H(T)
        h_cond = h_joint - h_tens

        # Mutual Information I(T; U) = H(U) - H(U|T)
        mi = h_units - h_cond

        return {
            "joint_entropy": round(h_joint, 4),
            "tens_entropy": round(h_tens, 4),
            "units_entropy": round(h_units, 4),
            "conditional_entropy": round(h_cond, 4),
            "mutual_information": round(mi, 4),
            "theoretical_max_joint": round(np.log2(100), 4)
        }

    def quick_backtest(self, df: pd.DataFrame, top_k: int = 10) -> Dict[str, Any]:
        """
        Backtest Strategy: Under-represented Mean Reversion.
        Identify digit pairs (Jodis) that have appeared significantly less than 
        expected given the current marginal distributions of tens and units.
        """
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total_days = 0
        
        # Warm up
        for i in range(200, len(jodis) - 1):
            sub_df = pd.Series(jodis[:i+1])
            digits = self._get_digits(sub_df)
            
            # 1. Calculate Expected Frequency for each of 100 Jodis
            # E(T, U) = P(T) * P(U) * Total_Samples
            p_tens = digits['tens'].value_counts(normalize=True).reindex(range(10), fill_value=0.01)
            p_units = digits['units'].value_counts(normalize=True).reindex(range(10), fill_value=0.01)
            
            observed_counts = digits.groupby(['tens', 'units']).size().reindex(
                pd.MultiIndex.from_product([range(10), range(10)]), fill_value=0
            )
            
            # 2. Find "Gaps" (Observed < Expected)
            gaps = {}
            for t in range(10):
                for u in range(10):
                    expected = p_tens[t] * p_units[u] * len(digits)
                    observed = observed_counts.loc[(t, u)]
                    # The gap is the deviation from digit-independence expectation
                    gaps[f"{t}{u}"] = expected - observed
            
            # 3. Predict Top K Jodis with largest gaps
            preds = sorted(gaps.keys(), key=lambda x: gaps[x], reverse=True)[:top_k]
            preds_int = [int(p) for p in preds]
            
            actual_next = jodis[i+1]
            if actual_next in preds_int:
                hits += 1
            total_days += 1
            
        hit_rate = hits / total_days if total_days > 0 else 0
        baseline = top_k / 100.0
        
        # Significance
        std_error = np.sqrt((baseline * (1 - baseline)) / total_days) if total_days > 0 else 0
        z_score = (hit_rate - baseline) / std_error if std_error > 0 else 0
        p_value = 1 - norm.cdf(z_score)

        results = {
            "hit_rate": round(hit_rate, 4),
            "baseline": round(baseline, 4),
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "total_days": total_days
        }
        
        log_experiment("digit_entropy_reversion", results)
        return results

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    dea = DigitEntropyAnalyzer()
    print("--- Digit Entropy Analysis ---")
    jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int)
    stats = dea.calculate_entropy(dea._get_digits(jodis))
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n--- Digit-Pair Gap Reversion Backtest (Top-10) ---")
    bt = dea.quick_backtest(df)
    print(f"Hit Rate: {bt['hit_rate']:.2%} (Baseline: {bt['baseline']:.2%})")
    print(f"Z-Score: {bt['z_score']}")
    print(f"P-Value: {bt['p_value']}")
