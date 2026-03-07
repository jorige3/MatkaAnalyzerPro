"""
Regime Detector Engine
----------------------
Detects 'regime changes' in the data, identifying when the system 
shifts from high entropy (random) to low entropy (patterned) behavior.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from engines.entropy import EntropyEngine

class RegimeDetector:
    """
    Detects if the current system is in a 'Patterned' or 'Random' regime
    based on rolling entropy and volatility of occurrences.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.entropy_engine = EntropyEngine()

    def detect_regime(self, df: pd.DataFrame) -> str:
        """Determines the current regime: 'PATTERNED' | 'RANDOM'."""
        if len(df) < self.window_size:
            return "UNKNOWN"
        
        # Calculate entropy of recent window
        recent_window = df.iloc[-self.window_size:]
        entropy_res = self.entropy_engine.run(recent_window)
        entropy_score = entropy_res.get("overall_entropy_score", 100)

        # Lower entropy means more patterns (lower uncertainty)
        # 0 is max patterns, 100 is max randomness
        if entropy_score < 40:
            return "PATTERNED"
        elif entropy_score > 70:
            return "RANDOM"
        else:
            return "TRANSITIONAL"

    def get_rolling_regimes(self, df: pd.DataFrame) -> pd.Series:
        """Returns a series of regimes over time."""
        regimes = []
        for i in range(self.window_size, len(df)):
            regime = self.detect_regime(df.iloc[:i])
            regimes.append(regime)
        return pd.Series(regimes, index=df.index[self.window_size:])

    def quick_backtest(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyzes if 'PATTERNED' regimes actually lead to better prediction accuracy."""
        # This is more of an analysis than a predictor
        regimes = self.get_rolling_regimes(df)
        
        counts = regimes.value_counts()
        return {
            "regime_distribution": counts.to_dict(),
            "patterned_percentage": round((counts.get('PATTERNED', 0) / len(regimes)) * 100, 2)
        }

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    detector = RegimeDetector()
    print("--- Regime Detection Analysis ---")
    results = detector.quick_backtest(df)
    print(f"Regime Distribution: {results['regime_distribution']}")
    print(f"Patterned Regime found in {results['patterned_percentage']}% of the time.")
