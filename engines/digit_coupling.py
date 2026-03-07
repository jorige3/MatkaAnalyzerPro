"""
Digit Coupling Engine
---------------------
Hypothesis: The tens digit and units digit are not independent.
This module analyzes the co-occurrence (coupling) of digits within a Jodi.
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, norm
from sklearn.metrics import mutual_info_score
from typing import Dict, Any

class DigitCouplingAnalyzer:
    """
    Analyzes statistical dependencies between the two digits of a Jodi.
    """
    def __init__(self):
        pass

    def _decompose(self, series: pd.Series) -> pd.DataFrame:
        # Convert to 2-digit strings then split
        s = series.astype(str).str.zfill(2)
        df = pd.DataFrame()
        df['tens'] = s.str[0].astype(int)
        df['units'] = s.str[1].astype(int)
        return df

    def run_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int)
        digits = self._decompose(jodis)
        
        # 1. Contingency Table
        contingency = pd.crosstab(digits['tens'], digits['units'])
        
        # 2. Chi-Square Test of Independence
        # Null Hypothesis: Tens and Units are independent
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        
        # 3. Mutual Information (Normalized)
        mi = mutual_info_score(digits['tens'], digits['units'])
        
        # 4. Correlation
        corr = digits['tens'].corr(digits['units'])
        
        return {
            "chi2_stat": chi2,
            "p_value": p_val,
            "mutual_information": mi,
            "pearson_correlation": corr,
            "top_couplings": self._get_top_couplings(digits)
        }

    def _get_top_couplings(self, digits_df: pd.DataFrame, top_n: int = 10) -> list:
        counts = digits_df.groupby(['tens', 'units']).size().reset_index(name='count')
        top = counts.sort_values('count', ascending=False).head(top_n)
        return [f"{row['tens']}{row['units']}" for _, row in top.iterrows()]

    def quick_backtest(self, df: pd.DataFrame, top_k: int = 10) -> Dict[str, Any]:
        """
        Walk-forward backtest using top-K historical digit couplings.
        """
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total = 0
        
        for i in range(100, len(jodis) - 1):
            historical_jodis = pd.Series(jodis[:i+1])
            digits = self._decompose(historical_jodis)
            
            # Identify most frequent couplings in history
            top_couplings = self._get_top_couplings(digits, top_n=top_k)
            top_couplings_int = [int(c) for c in top_couplings]
            
            actual_next = jodis[i+1]
            if actual_next in top_couplings_int:
                hits += 1
            total += 1
            
        hit_rate = hits / total if total > 0 else 0
        expected_prob = top_k / 100.0
        
        # Statistical Significance
        std_error = np.sqrt((expected_prob * (1 - expected_prob)) / total)
        z_score = (hit_rate - expected_prob) / std_error if std_error > 0 else 0
        p_value = 1 - norm.cdf(z_score)

        return {
            "hit_rate": round(hit_rate, 4),
            "expected_prob": expected_prob,
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "total_samples": total,
            "hits": hits
        }

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    analyzer = DigitCouplingAnalyzer()
    print("--- Digit Coupling Analysis Metrics ---")
    stats = analyzer.run_analysis(df)
    print(f"Chi-Square Independence P-Value: {stats['p_value']:.6f}")
    print(f"Pearson Correlation (Tens vs Units): {stats['pearson_correlation']:.4f}")
    print(f"Mutual Information: {stats['mutual_information']:.4f}")
    
    print("\n--- Digit Coupling Backtest (Top-10) ---")
    bt = analyzer.quick_backtest(df, top_k=10)
    print(f"Hit Rate: {bt['hit_rate']:.2%} (Baseline: {bt['expected_prob']:.2%})")
    print(f"Z-Score: {bt['z_score']}")
    print(f"P-Value: {bt['p_value']}")
