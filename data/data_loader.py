# data/data_loader.py

import pandas as pd
from data.schema import DataSchemaValidator

class DataLoader:
    """
    Handles loading and initial validation of historical Matka data from a CSV file.
    Ensures data conforms to a predefined schema before further processing.
    """
    def __init__(self, file_path: str, schema_path: str):
        """
        Initializes the DataLoader with paths to the data file and schema definition.

        Parameters
        ----------
        file_path : str
            The absolute or relative path to the CSV data file.
        schema_path : str
            The absolute or relative path to the data schema definition (e.g., 'data/schema.py').
            Currently not directly used for loading, but kept for consistency.
        """
        self.file_path = file_path
        self.schema_path = schema_path # Not directly used here, but kept for consistency if schema loading logic changes

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the specified CSV file and validates it against the schema.

        Returns
        -------
        pd.DataFrame
            The validated and normalized DataFrame.

        Raises
        ------
        FileNotFoundError
            If the data file does not exist.
        ValueError
            If data validation fails.
        """
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.file_path}")

        df = DataSchemaValidator.validate_and_normalize(df)
        return df
