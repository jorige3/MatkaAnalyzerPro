import pandas as pd
from data.data_loader import DataLoader
from simulation.paper_test import PaperBacktest
from engines.frequency import FrequencyEngine
from engines.cycles import CycleEngine
from engines.digits import DigitEngine
from engines.momentum import MomentumEngine
from engines.entropy import EntropyEngine # New import
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
    results = {}

    freq_engine = FrequencyEngine(window_days=30)
    cycle_engine = CycleEngine()
    digit_engine = DigitEngine()
    momentum_engine = MomentumEngine()
    entropy_engine = EntropyEngine() # New instantiation
    conf_engine = ConfidenceEngine()

    frequency = freq_engine.run(df)
    cycles = cycle_engine.run(df)
    digits_output = digit_engine.run(df)
    digits = digits_output["jodi_scores"]
    momentum = momentum_engine.run(df)
    entropy = entropy_engine.run(df) # New engine run

    confidence = conf_engine.run(frequency, cycles, digits, momentum, sample_size=len(df))

    results["frequency"] = frequency
    results["cycles"] = cycles
    results["digits"] = digits
    results["individual_digit_strength"] = digits_output["individual_digit_strength"]
    results["momentum"] = momentum
    results["entropy"] = entropy # Add entropy to results
    results["confidence"] = confidence

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
        print("\nTop Confidence Alignments:")
        for jodi, score, tags in confidence:
            print(
                f"  Jodi {jodi} → "
                f"Score: {score} | "
                f"Tags: {', '.join(tags)}"
            )


    print("\n=== Matka Analyzer Pro : Analysis Summary ===\n")

    # Frequency summary
    freq = results.get("frequency", {})
    if freq:
        print("Top Frequency Jodis (Last 30 Days):")
        for jodi, score in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            print(f"  Jodi {jodi} → Frequency Score: {score}")
        print()

    # Cycle summary
    cycles = results.get("cycles", {})
    if cycles:
        due = [j for j, v in cycles.items() if v["status"] == "DUE"]
        exhausted = [j for j, v in cycles.items() if v["status"] == "EXHAUSTED"]

        print(f"DUE Zone Count       : {len(due)}")
        print(f"EXHAUSTED Zone Count : {len(exhausted)}")

    print("\nNote: All outputs are historical observations only.\n")

    # Digit summary
    digits = results.get("digits", {})
    if digits:
        print("\nTop Digit-Strength Jodis:")
        for j, info in sorted(
            digits.items(),
            key=lambda x: x[1]["digit_score"],
            reverse=True
        )[:top_n]:
            print(
                f"  Jodi {j} → Digit Score: {info['digit_score']} "
                f"(digits {info['tens_digit']},{info['unit_digit']})"
            )

    # Momentum summary
    momentum = results.get("momentum", {})
    if momentum:
        print("\nTop Momentum Jodis:")
        for j, score in sorted(
            momentum.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]:
            print(f"  Jodi {j} → Momentum Score: {score}")

    # Entropy summary
    entropy_score = results.get("entropy", {}).get("overall_entropy_score")
    if entropy_score is not None:
        print(f"\nOverall Entropy Score: {entropy_score}")

def main():
    """
    Orchestrates the entire Matka Analyzer Pro analysis process.
    This includes:
    1. Loading historical data.
    2. Running various analysis engines (Frequency, Cycles, Digits, Momentum).
    3. Calculating confidence scores.
    4. Performing a paper backtest simulation.
    5. Summarizing and printing the results to the console.
    """
    data_loader = DataLoader(file_path=DATA_FILE, schema_path=SCHEMA_FILE)
    df = data_loader.load_data()
    print(f"Loaded {len(df)} records from {DATA_FILE}")
    results = run_engines(df)

    conf_engine = ConfidenceEngine()
    final_scores = conf_engine.run(
        frequency=results["frequency"],
        cycles=results["cycles"],
        digits=results["digits"],
        momentum=results["momentum"],
        sample_size=len(df),
        top_n=TOP_N_PREDICTIONS
    )
    results["confidence"] = final_scores

    backtester = PaperBacktest(DATA_FILE, min_history_days=MIN_HISTORY_DAYS)
    backtest_stats = backtester.run(top_n=TOP_N_PREDICTIONS)

    print("\n=== Paper Backtest Results ===")
    for k, v in backtest_stats.items():
        print(f"{k}: {v}")

    summarize(results)
    print(f"\n{DISCLAIMER}\n")

    # Print summary
    print(f"\nData Confidence Factor Applied (rows={len(df)})")
    print("\nAnalysis Complete. Use 'streamlit run ui/dashboard.py' to visualize.")


if __name__ == "__main__":
    # Global disclaimer for CLI
    print(DISCLAIMER)
    main()
