# config.py

# Data file paths
DATA_FILE = "data/sridevi.csv"
SCHEMA_FILE = "data/schema.py"

# Confidence Engine Weights (for documentation and potential external tuning)
# These weights are used in scoring/confidence.py
CONFIDENCE_WEIGHTS = {
    "frequency": 0.35,
    "cycles": 0.30,
    "digits": 0.20,
    "momentum": 0.15,
}

# Backtesting parameters
MIN_HISTORY_DAYS = 30
TOP_N_PREDICTIONS = 10

# Tagging thresholds for ConfidenceEngine
HIGH_FREQ_THRESHOLD = 75
LOW_FREQ_THRESHOLD = 25
STRONG_DIGIT_THRESHOLD = 70
WEAK_DIGIT_THRESHOLD = 30
HIGH_MOMENTUM_THRESHOLD = 75
LOW_MOMENTUM_THRESHOLD = 25
BALANCED_SIGNAL_LOWER = 40
BALANCED_SIGNAL_UPPER = 60
MODERATE_SIGNAL_LOWER = 60 # Scores above this but not HIGH_MOMENTUM etc.

# Disclaimer for CLI and UI
DISCLAIMER = (
    "Disclaimer: This tool provides historical data analysis and pattern "
    "recognition for educational purposes only. It does NOT offer "
    "gambling advice, predictions, or guaranteed outcomes. "
    "All results are observations of past data and should not be "
    "interpreted as future performance indicators. "
    "Please use responsibly and for analytical insights only."
)