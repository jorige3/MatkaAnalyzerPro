# data/schema.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataSchemaValidator:
    """
    Validates and normalizes historical Matka data.
    Ensures data integrity before analytical engines process it.
    """

    REQUIRED_COLUMNS = {"Date", "Jodi"}

    @staticmethod
    def validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and normalizes the input DataFrame.
        
        Steps:
        1. Column Presence Check
        2. Date Normalization & Row Removal (for invalid dates)
        3. Jodi Normalization (Padding, Digit Check, Range Check)
        4. Deduplication
        5. Chronological Sorting
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")

        # --- 1. Column Check ---
        missing = DataSchemaValidator.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        data = df.copy()
        initial_count = len(data)

        # --- 2. Normalize Date ---
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        invalid_dates = data["Date"].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Dropped {invalid_dates} rows due to invalid date formats.")
        data = data.dropna(subset=["Date"])

        # --- 3. Normalize Jodi ---
        # Drop NaN jodis first
        data = data.dropna(subset=["Jodi"])
        
        # Convert to numeric, then to int, then to zero-padded string
        # This handles cases where Jodi might be read as float (e.g. 42.0)
        try:
            data["Jodi"] = pd.to_numeric(data["Jodi"], errors="coerce")
            invalid_jodi_count = data["Jodi"].isna().sum()
            if invalid_jodi_count > 0:
                logger.warning(f"Dropped {invalid_jodi_count} rows with non-numeric Jodi values.")
            data = data.dropna(subset=["Jodi"])
            
            data["Jodi"] = data["Jodi"].astype(int).astype(str).str.zfill(2)
        except Exception as e:
            logger.error(f"Error during Jodi normalization: {e}")
            raise ValueError(f"Failed to normalize Jodi column: {e}")
        
        # Range check 00-99
        data["Jodi_int"] = data["Jodi"].astype(int)
        in_range = data["Jodi_int"].between(0, 99)
        out_of_range_count = (~in_range).sum()
        if out_of_range_count > 0:
            logger.warning(f"Dropped {out_of_range_count} rows with Jodi values outside 00-99.")
        data = data[in_range]
        data = data.drop(columns=["Jodi_int"])

        # --- 4. Deduplication ---
        # We generally expect one result per date for a specific market
        duplicate_count = data.duplicated(subset=["Date"], keep='last').sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate dates. Keeping only the latest entry for each.")
            data = data.drop_duplicates(subset=["Date"], keep='last')

        # --- 5. Final Sort & Clean ---
        data = data.sort_values("Date").reset_index(drop=True)
        
        final_count = len(data)
        logger.info(f"Data validation complete. Rows: {initial_count} -> {final_count}")

        if data.empty:
            raise ValueError("No valid data rows remain after normalization and cleaning.")

        return data
