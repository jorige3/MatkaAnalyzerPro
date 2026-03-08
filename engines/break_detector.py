"""
Structural Break Detector Engine
--------------------------------
Identifies sudden shifts in the statistical distribution of draws.
Uses a CUSUM-based approach to detect when the recent 'mean behavior'
deviates significantly from the long-term historical baseline.
"""
import pandas as pd
import numpy as np
from research.tracker import log_experiment
from typing import Dict, List, Any

class BreakDetector:
    """
    Detects structural breaks (change-points) in the Jodi sequence.
    """
    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift = drift # Slack variable to prevent triggers from small noise

    def detect_breaks(self, df: pd.DataFrame) -> List[int]:
        """
        Returns indices where a structural break was detected.
        We monitor the 'Jodi' value itself as a proxy for the mean of the process.
        """
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().values
        # Standardize for CUSUM
        mu = np.mean(jodis)
        std = np.std(jodis)
        if std == 0: return []
        
        x = (jodis - mu) / std
        
        # Cumulative Sums for upward and downward shifts
        gp, gn = np.zeros(len(x)), np.zeros(len(x))
        breaks = []
        
        for i in range(1, len(x)):
            gp[i] = max(0, gp[i-1] + x[i] - self.drift)
            gn[i] = max(0, gn[i-1] - x[i] - self.drift)
            
            if gp[i] > self.threshold or gn[i] > self.threshold:
                breaks.append(i)
                # Reset sums after a break is detected
                gp[i] = 0
                gn[i] = 0
                
        return breaks

    def quick_backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the frequency of detected breaks and their relation to regimes.
        """
        breaks = self.detect_breaks(df)
        total_days = len(df)
        break_frequency = len(breaks) / total_days if total_days > 0 else 0
        
        # Check if breaks precede high-volatility (Random) periods
        results = {
            "total_breaks_detected": len(breaks),
            "break_frequency_rate": round(break_frequency, 4),
            "average_days_between_breaks": round(total_days / len(breaks), 2) if breaks else 0
        }
        
        log_experiment("structural_break_detection", results)
        return results

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    bd = BreakDetector(threshold=4.0, drift=0.1)
    print("--- Structural Break Detection Analysis ---")
    stats = bd.quick_backtest(df)
    print(f"Total Breaks Found: {stats['total_breaks_detected']}")
    print(f"Average Stability Period: {stats['average_days_between_breaks']} days")
    
    breaks = bd.detect_breaks(df)
    if breaks:
        last_break_date = df.iloc[breaks[-1]]['Date']
        print(f"Latest Detected Break: {last_break_date}")
