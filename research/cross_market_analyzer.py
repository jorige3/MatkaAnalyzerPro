"""
Cross-Market Analyzer
---------------------
Hypothesis: Different Matka markets (Sridevi vs Kalyan) are not independent.
This module looks for correlations and lead-lag relationships.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, norm
from typing import Dict, Any

class CrossMarketAnalyzer:
    """
    Analyzes correlations between two different Matka markets.
    """
    def __init__(self, file1: str = 'data/sridevi.csv', file2: str = 'data/kalyan.csv'):
        self.file1 = file1
        self.file2 = file2

    def load_and_merge(self) -> pd.DataFrame:
        df1 = pd.read_csv(self.file1)
        df2 = pd.read_csv(self.file2)
        
        df1['Date'] = pd.to_datetime(df1['Date'])
        df2['Date'] = pd.to_datetime(df2['Date'])
        
        df1 = df1.rename(columns={'Jodi': 'Jodi_Sridevi'})
        df2 = df2.rename(columns={'Jodi': 'Jodi_Kalyan'})
        
        merged = pd.merge(df1, df2, on='Date', how='inner').sort_values('Date')
        return merged

    def run_analysis(self) -> Dict[str, Any]:
        df = self.load_and_merge()
        if df.empty:
            return {"error": "No overlapping data found."}
        
        # 1. Pearson Correlation
        corr, p_val = pearsonr(df['Jodi_Sridevi'], df['Jodi_Kalyan'])
        
        # 2. Lagged Correlation (Sridevi(t) vs Kalyan(t-1))
        # Does yesterday's Kalyan predict today's Sridevi?
        df['Jodi_Kalyan_Lag1'] = df['Jodi_Kalyan'].shift(1)
        df_lag = df.dropna()
        corr_lag, p_val_lag = pearsonr(df_lag['Jodi_Sridevi'], df_lag['Jodi_Kalyan_Lag1'])
        
        return {
            "correlation_same_day": round(corr, 4),
            "p_value_same_day": round(p_val, 6),
            "correlation_lag1": round(corr_lag, 4),
            "p_value_lag1": round(p_val_lag, 6),
            "overlapping_days": len(df)
        }

    def quick_backtest(self, top_k: int = 10) -> Dict[str, Any]:
        """
        Backtest: Use Kalyan's result from yesterday to predict Sridevi's result today.
        Hypothesis: If Jodi X appeared in Kalyan yesterday, it's more likely to 
        appear in Sridevi today (Echo Effect).
        """
        df = self.load_and_merge()
        if len(df) < 100:
            return {"error": "Insufficient overlapping data."}
            
        hits = 0
        total = 0
        
        # This is a very simple "Echo" strategy
        for i in range(1, len(df)):
            yesterday_kalyan = df.iloc[i-1]['Jodi_Kalyan']
            today_sridevi = df.iloc[i]['Jodi_Sridevi']
            
            # Prediction is just the single Jodi from Kalyan yesterday
            if yesterday_kalyan == today_sridevi:
                hits += 1
            total += 1
            
        hit_rate = hits / total if total > 0 else 0
        expected_prob = 1 / 100.0 # Predicting 1 Jodi
        
        std_error = np.sqrt((expected_prob * (1 - expected_prob)) / total)
        z_score = (hit_rate - expected_prob) / std_error if std_error > 0 else 0
        p_value = 1 - norm.cdf(z_score)

        return {
            "strategy": "Kalyan(t-1) Echo in Sridevi(t)",
            "hit_rate": round(hit_rate, 4),
            "expected_prob": expected_prob,
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "total_samples": total,
            "hits": hits
        }

if __name__ == "__main__":
    analyzer = CrossMarketAnalyzer()
    print("--- Cross-Market Analysis (Sridevi vs Kalyan) ---")
    stats = analyzer.run_analysis()
    if "error" in stats:
        print(stats["error"])
    else:
        print(f"Same-day Correlation: {stats['correlation_same_day']} (p={stats['p_value_same_day']})")
        print(f"Lag-1 Correlation: {stats['correlation_lag1']} (p={stats['p_value_lag1']})")
        print(f"Overlapping Days: {stats['overlapping_days']}")
        
        print("\n--- Echo Strategy Backtest ---")
        bt = analyzer.quick_backtest()
        print(f"Hit Rate: {bt['hit_rate']:.2%} (Baseline: {bt['expected_prob']:.2%})")
        print(f"Z-Score: {bt['z_score']}")
        print(f"P-Value: {bt['p_value']}")
