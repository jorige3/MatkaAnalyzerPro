import logging
import pandas as pd
from data.data_loader import DataLoader
from simulation.paper_test import PaperBacktest
from engines.frequency import FrequencyEngine
from engines.cycles import CycleEngine
from engines.digits import DigitEngine
from engines.momentum import MomentumEngine
from engines.entropy import EntropyEngine
from scoring.confidence import ConfidenceEngine
from config import DATA_FILE, SCHEMA_FILE, DISCLAIMER, MIN_HISTORY_DAYS, TOP_N_PREDICTIONS


def run_engines(df: pd.DataFrame) -> dict:
    """
    Executes all analysis engines (Frequency, Cycles, Digits, Momentum, Entropy)
    and the ConfidenceEngine on the provided DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing historical Matka data.

    Returns
    -------
    dict
        A dictionary containing the results from each engine:
        - 'frequency': Results from FrequencyEngine.
        - 'cycles': Results from CycleEngine.
        - 'digits': Jodi-specific digit scores from DigitEngine.
        - 'individual_digit_strength': Individual digit strengths from DigitEngine.
        - 'momentum': Results from MomentumEngine.
        - 'entropy': Overall entropy score from EntropyEngine.
        - 'confidence': Results from ConfidenceEngine (ranked Jodis with scores and tags).
    """
    logging.info("Running analysis engines...")
    results = {}

    freq_engine = FrequencyEngine(window_days=30)
    cycle_engine = CycleEngine()
    digit_engine = DigitEngine()
    momentum_engine = MomentumEngine()
    entropy_engine = EntropyEngine()
    conf_engine = ConfidenceEngine()

    frequency = freq_engine.run(df)
    cycles = cycle_engine.run(df)
    digits_output = digit_engine.run(df)
    digits = digits_output["jodi_scores"]
    momentum = momentum_engine.run(df)
    entropy = entropy_engine.run(df)

    confidence = conf_engine.run(frequency, cycles, digits, momentum, sample_size=len(df), top_n=TOP_N_PREDICTIONS)

    results["frequency"] = frequency
    results["cycles"] = cycles
    results["digits"] = digits
    results["individual_digit_strength"] = digits_output["individual_digit_strength"]
    results["momentum"] = momentum
    results["entropy"] = entropy
    results["confidence"] = confidence

    logging.info("Analysis engines finished.")
    return results


def summarize(results: dict, top_n: int = TOP_N_PREDICTIONS):
    """
    Prints a summary of the analysis results to the console.

    Includes:
    - Top Confidence Alignments (Jodis with highest confidence scores and their tags).
    - Summary of Frequency Analysis (top Jodis by frequency).
    - Summary of Cycle Analysis (counts of 'DUE' and 'EXHAUSTED' cycles).
    - Summary of Digit Strength Analysis (top Jodis by digit score).
    - Summary of Momentum Analysis (top Jodis by momentum score).
    - Overall Entropy Score.

    Parameters
    ----------
    results : dict
        The dictionary containing all analysis results from run_engines.
    top_n : int, optional
        The number of top Jodis to display for each summary section, by default TOP_N_PREDICTIONS.
    """
    confidence = results.get("confidence", [])
    if confidence:
        logging.info("\nTop Confidence Alignments:")
        for jodi, score, tags in confidence:
            logging.info(
                f"  Jodi {jodi} → "
                f"Score: {score} | "
                f"Tags: {', '.join(tags)}"
            )


    logging.info("\n=== Matka Analyzer Pro : Analysis Summary ===\n")

    # Frequency summary
    freq = results.get("frequency", {})
    if freq:
        logging.info("Top Frequency Jodis (Last 30 Days):")
        for jodi, score in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            logging.info(f"  Jodi {jodi} → Frequency Score: {score}")
        logging.info("")

    # Cycle summary
    cycles = results.get("cycles", {})
    if cycles:
        due = [j for j, v in cycles.items() if v["status"] == "DUE"]
        exhausted = [j for j, v in cycles.items() if v["status"] == "EXHAUSTED"]

        logging.info(f"DUE Zone Count       : {len(due)}")
        logging.info(f"EXHAUSTED Zone Count : {len(exhausted)}")

    logging.info("\nNote: All outputs are historical observations only.\n")

    # Digit summary
    digits = results.get("digits", {})
    if digits:
        logging.info("\nTop Digit-Strength Jodis:")
        for j, info in sorted(
            digits.items(),
            key=lambda x: x[1]["digit_score"],
            reverse=True
        )[:top_n]:
            logging.info(
                f"  Jodi {j} → Digit Score: {info['digit_score']} "
                f"(digits {info['tens_digit']},{info['unit_digit']})"
            )

    # Momentum summary
    momentum = results.get("momentum", {})
    if momentum:
        logging.info("\nTop Momentum Jodis:")
        for j, score in sorted(
            momentum.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]:
            logging.info(f"  Jodi {j} → Momentum Score: {score}")

    # Entropy summary
    entropy_score = results.get("entropy", {}).get("overall_entropy_score")
    if entropy_score is not None:
        logging.info(f"\nOverall Entropy Score: {entropy_score}")

def main():
    """
    Orchestrates the entire Matka Analyzer Pro analysis process.
    This includes:
    1. Loading historical data.
    2. Running various analysis engines (Frequency, Cycles, Digits, Momentum, Entropy).
    3. Calculating confidence scores.
    4. Performing a paper backtest simulation.
    5. Summarizing and printing the results to the console.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(DISCLAIMER)
    logging.info("Starting Matka Analyzer Pro...")
    data_loader = DataLoader(file_path=DATA_FILE, schema_path=SCHEMA_FILE)
    df = data_loader.load_data()
    logging.info(f"Loaded {len(df)} records from {DATA_FILE}")
    results = run_engines(df)

    backtester = PaperBacktest(DATA_FILE, min_history_days=MIN_HISTORY_DAYS)
    backtest_stats = backtester.run(top_n=TOP_N_PREDICTIONS)

    logging.info("\n=== Paper Backtest Results ===")
    for k, v in backtest_stats.items():
        logging.info(f"{k}: {v}")

    summarize(results)
    logging.info(f"\nData Confidence Factor Applied (rows={len(df)})")
    logging.info("\nAnalysis Complete. Use 'streamlit run ui/dashboard.py' to visualize.")


if __name__ == "__main__":
    main()
