"""
Provides statistical tools to assess the significance of a model's performance
against a random baseline. This includes the Binomial Test and a Monte Carlo simulation.
"""

import numpy as np
from scipy.stats import binomtest, percentileofscore
from typing import Dict, Any

def run_binomial_test(hits: int, total_trials: int, num_classes: int = 100) -> float:
    """
    Performs a one-sided Binomial Test to determine if the model's accuracy is
    statistically significant compared to random chance.

    Null Hypothesis (H₀): The model's hit rate is less than or equal to random chance (p <= 1/100).
    Alternative Hypothesis (H₁): The model's hit rate is greater than random chance (p > 1/100).

    Args:
        hits (int): The number of correct predictions (successes).
        total_trials (int): The total number of predictions made.
        num_classes (int): The number of possible outcomes (e.g., 100 for jodis 00-99).

    Returns:
        float: The calculated p-value from the test.
    """
    if total_trials == 0:
        print("Binomial Test: Cannot be performed with zero total trials.")
        return 1.0

    p_random = 1.0 / num_classes
    
    # We use 'greater' because we're testing if our model is *better* than random.
    result = binomtest(k=hits, n=total_trials, p=p_random, alternative='greater')
    
    p_value = result.pvalue

    print("\n--- One-Sided Binomial Test ---")
    print(f"Total Predictions: {total_trials}, Hits: {hits}")
    print(f"Model Accuracy: {hits/total_trials:.2%}, Random Chance: {p_random:.2%}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: The model's performance is statistically significant (p < 0.05).")
        print("Conclusion: We can reject the null hypothesis; the model is likely better than random chance.")
    else:
        print("Result: The model's performance is NOT statistically significant (p >= 0.05).")
        print("Conclusion: We cannot reject the null hypothesis; the observed accuracy could be due to random chance.")
    
    return p_value

def run_monte_carlo_simulation(
    model_accuracy: float,
    total_trials: int,
    num_classes: int = 100,
    num_simulations: int = 20000
) -> float:
    """
    Simulates thousands of random models to see where the actual model's
    performance ranks.

    Args:
        model_accuracy (float): The accuracy achieved by the actual model.
        total_trials (int): The total number of predictions made.
        num_classes (int): The number of possible outcomes.
        num_simulations (int): The number of random models to simulate.

    Returns:
        float: The percentile rank of the model's accuracy compared to random models.
    """
    if total_trials == 0:
        print("\nMonte Carlo Simulation: Cannot be performed with zero total trials.")
        return 0.0

    p_random = 1.0 / num_classes
    
    # Simulate the number of hits for thousands of purely random models
    random_hits = np.random.binomial(n=total_trials, p=p_random, size=num_simulations)
    random_accuracies = random_hits / total_trials

    # Calculate where our model's accuracy falls in this distribution
    percentile = percentileofscore(random_accuracies, model_accuracy)

    print("\n--- Monte Carlo Simulation ---")
    print(f"Simulated {num_simulations} random models over {total_trials} predictions.")
    print(f"Random models' 95th percentile accuracy: {np.percentile(random_accuracies, 95):.2%}")
    print(f"Your model's accuracy: {model_accuracy:.2%}")
    print(f"Percentile Rank: Your model's accuracy is at the {percentile:.2f}th percentile.")

    if percentile > 95:
        print("Conclusion: Your model performed better than 95% of simulated random models.")
    else:
        print("Conclusion: Your model's performance is within the range expected from random chance.")
        
    return percentile

def run_significance_analysis(walk_forward_results: Dict[str, Any], num_classes: int = 100):
    """
    A wrapper function to run all statistical significance tests on the
    results from a walk-forward validation.

    Args:
        walk_forward_results (Dict): The results dictionary from `walk_forward_validation`.
                                     Expected keys: 'accuracy', 'total_samples'.
        num_classes (int): The number of possible outcomes for the prediction task.
    """
    if not walk_forward_results or 'accuracy' not in walk_forward_results or 'total_samples' not in walk_forward_results:
        print("\nSignificance Analysis: Invalid or empty results from walk-forward validation. Aborting.")
        return

    model_accuracy = walk_forward_results['accuracy']
    # The number of hits can be calculated from accuracy and total samples
    total_trials = walk_forward_results['total_samples']
    hits = int(round(model_accuracy * total_trials))

    print("\n=============================================")
    print("=   STATISTICAL SIGNIFICANCE ANALYSIS   =")
    print("=============================================")

    run_binomial_test(hits=hits, total_trials=total_trials, num_classes=num_classes)
    
    run_monte_carlo_simulation(
        model_accuracy=model_accuracy,
        total_trials=total_trials,
        num_classes=num_classes
    )

    print("\n=============================================")
