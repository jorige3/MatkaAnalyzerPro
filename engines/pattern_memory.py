"""
Pattern Memory Engine
---------------------
Hypothesis: Specific sequences of Jodis (e.g., last 3-5 draws) may repeat 
historically. This engine looks for exact matches of the recent sequence 
in historical data and identifies what followed.
"""
import pandas as pd
import numpy as np
from research.tracker import log_experiment
from typing import Dict, List, Any

class PatternMemoryEngine:
    def __init__(self, sequence_length: int = 3):
        self.sequence_length = sequence_length

    def find_historical_matches(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Looks for the most recent sequence in history and returns the 
        distribution of Jodis that followed those matches.
        """
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        if len(jodis) <= self.sequence_length:
            return {}

        target_seq = jodis[-self.sequence_length:]
        
        matches = []
        # Search history (excluding the current tail)
        for i in range(len(jodis) - self.sequence_length - 1):
            if jodis[i : i+self.sequence_length] == target_seq:
                # Found match! Get the next one.
                matches.append(jodis[i + self.sequence_length])
        
        if not matches:
            return {}

        counts = pd.Series(matches).value_counts()
        probs = counts / len(matches)
        return probs.to_dict()

    def quick_backtest(self, df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        jodis = pd.to_numeric(df['Jodi'], errors='coerce').dropna().astype(int).tolist()
        hits = 0
        total = 0
        
        # Test on the second half of data
        for i in range(len(jodis)//2, len(jodis) - 1):
            sub_df = df.iloc[:i+1]
            probs = self.find_historical_matches(sub_df)
            
            if not probs:
                continue
                
            # Take top K predictions from memory
            preds = sorted(probs.keys(), key=lambda x: probs[x], reverse=True)[:top_k]
            preds_str = [str(p).zfill(2) for p in preds]
            
            actual = str(jodis[i+1]).zfill(2)
            if actual in preds_str:
                hits += 1
            total += 1
            
        hit_rate = hits / total if total > 0 else 0
        log_experiment("pattern_memory", {"hit_rate": hit_rate, "total_samples": total, "seq_len": self.sequence_length})
        return {"hit_rate": hit_rate, "total": total}

if __name__ == "__main__":
    from data.data_loader import DataLoader
    from config import DATA_FILE, SCHEMA_FILE
    loader = DataLoader(DATA_FILE, SCHEMA_FILE)
    df = loader.load_data()
    
    # Try length 2 for more matches
    pme = PatternMemoryEngine(sequence_length=2)
    print("--- Pattern Memory Backtest (Seq Len: 2) ---")
    results = pme.quick_backtest(df, top_k=5)
    print(f"Pattern Memory Hit Rate: {results['hit_rate']:.2%} (Samples: {results['total']})")
