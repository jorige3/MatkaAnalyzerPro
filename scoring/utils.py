"""
Utility functions for scoring and data validation.
"""
from typing import Set
import pandas as pd


def validate_df(df: pd.DataFrame, required_cols: Set[str] = None) -> pd.DataFrame:
    """Shared validation for Sridevi data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    required_cols : Set[str], optional
        A set of column names that must be present in the DataFrame, by default {"Date", "Jodi"}.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with the 'Date' column converted to datetime objects.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    if required_cols is None:
        required_cols = {"Date", "Jodi"}

    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    return data
