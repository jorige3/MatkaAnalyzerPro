"""
Core Analysis Module for Matka Analyzer Pro
-------------------------------------------
Provides functions for loading data, computing global bias statistics,
and performing deep-dive analysis on specific Jodis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from data.data_loader import DataLoader
from config import DATA_FILE, SCHEMA_FILE

def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    """
    Loads and normalizes Matka data from a CSV file.
    """
    loader = DataLoader(path, SCHEMA_FILE)
    return loader.load_data()

def compute_bias_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes global bias statistics for all Jodis in the dataset.
    Returns top Jodis, high/low bias indicators, entropy, and chi-square results.
    """
    from engines.statistical_analysis import StatisticalAnalyzer
    from engines.frequency import FrequencyEngine
    
    analyzer = StatisticalAnalyzer(df, target_col='Jodi')
    fe = FrequencyEngine()
    
    frequency = fe.run(df)
    entropy_val, max_entropy = analyzer.shannon_entropy()
    chi_stat, p_value = analyzer.chi_square_test()
    
    # Simple top Jodis by frequency
    top_jodis = sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "top_jodis": top_jodis,
        "entropy": entropy_val,
        "max_entropy": max_entropy,
        "chi_stat": chi_stat,
        "p_value": p_value
    }

def analyze_jodi(df: pd.DataFrame, jodi: str, rolling_window: int = 30) -> Dict[str, Any]:
    """
    Performs a statistical deep-dive for a specific Jodi.
    
    Returns: count, Z-score, frequency_score, last_date, rolling_freq, etc.
    """
    jodi_str = str(jodi).zfill(2)
    
    # 1. Frequency Calculations
    total_records = len(df)
    actual_count = df['Jodi'].eq(jodi_str).sum()
    expected_count = total_records / 100.0
    
    # 2. Z-Score
    std_dev = np.sqrt(total_records * 0.01 * 0.99)
    z_score = (actual_count - expected_count) / std_dev if std_dev > 0 else 0.0
    
    # 3. Frequency Score (0-100)
    freq_score = min((actual_count / expected_count) * 50.0, 100.0) if expected_count > 0 else 0.0
    
    # 4. Recency
    last_occ = df[df['Jodi'] == jodi_str]['Date'].max()
    days_since = (df['Date'].max() - last_occ).days if pd.notnull(last_occ) else None
    
    # 5. Rolling Analysis
    df_copy = df.copy()
    df_copy['is_target'] = df_copy['Jodi'].eq(jodi_str).astype(int)
    df_copy['rolling_freq'] = df_copy['is_target'].rolling(window=rolling_window).sum()
    current_rolling = df_copy['rolling_freq'].iloc[-1]
    
    return {
        "jodi": jodi_str,
        "count": actual_count,
        "expected": expected_count,
        "z_score": z_score,
        "frequency_score": freq_score,
        "last_occurrence": last_occ,
        "days_since": days_since,
        "current_rolling_freq": current_rolling,
        "total_records": total_records,
        "rolling_data": df_copy[['Date', 'rolling_freq', 'is_target']]
    }

def backtest_strategy(df: pd.DataFrame, jodis: list, window: int = 30) -> float:
    """
    Calculates the historical hit rate for a list of Jodis over the dataset.
    """
    df_copy = df.copy()
    df_copy["hit"] = df_copy["Jodi"].isin(jodis).astype(int)
    win_rate = df_copy["hit"].mean()
    return win_rate
