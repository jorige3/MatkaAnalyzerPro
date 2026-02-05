# data/schema.py

import pandas as pd


class DataSchemaValidator:
    """
    Validates and normalizes historical Matka data.
    """

    REQUIRED_COLUMNS = {"Date", "Jodi"}

    @staticmethod
    def validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and normalizes the input DataFrame to ensure it conforms to the expected schema.
        This method performs several cleaning steps:
        - Checks for required columns ('Date', 'Jodi').
        - Converts 'Date' column to datetime objects, dropping invalid entries.
        - Normalizes 'Jodi' column to a two-digit string format (e.g., '05').
        - Filters out non-numeric or out-of-range Jodi values (00-99).
        - Sorts the data chronologically by 'Date'.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to be validated and normalized.

        Returns
        -------
        pd.DataFrame
            The validated, cleaned, and normalized DataFrame.

        Raises
        ------
        ValueError
            If required columns are missing or if no valid rows remain after normalization.
        """
        # --- Column check ---
        if not DataSchemaValidator.REQUIRED_COLUMNS.issubset(df.columns):
            raise ValueError(
                f"CSV must contain columns: {DataSchemaValidator.REQUIRED_COLUMNS}"
            )

        data = df.copy()

        # --- Normalize Date ---
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"])

        # --- Normalize Jodi ---
        data["Jodi"] = data["Jodi"].astype(str).str.strip()

        # Keep only numeric jodis
        data = data[data["Jodi"].str.isdigit()]

        # Pad single digits → 2 digits
        data["Jodi"] = data["Jodi"].str.zfill(2)

        # Enforce range 00–99
        data = data[data["Jodi"].astype(int).between(0, 99)]

        # --- Sort chronologically ---
        data = data.sort_values("Date").reset_index(drop=True)

        if data.empty:
            raise ValueError("No valid rows after normalization")

        return data
