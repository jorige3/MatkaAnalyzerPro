# tests/test_confidence.py
import pytest
from scoring.confidence import ConfidenceEngine

@pytest.fixture
def confidence_engine():
    """Returns a ConfidenceEngine instance."""
    return ConfidenceEngine()

def test_data_confidence_factor(confidence_engine):
    """
    Tests the data_confidence_factor method with various sample sizes.
    """
    assert confidence_engine.data_confidence_factor(10) == 0.40
    assert confidence_engine.data_confidence_factor(30) == 0.60
    assert confidence_engine.data_confidence_factor(60) == 0.75
    assert confidence_engine.data_confidence_factor(120) == 0.90
    assert confidence_engine.data_confidence_factor(250) == 1.00
    assert confidence_engine.data_confidence_factor(500) == 1.00

def test_run_confidence_engine(confidence_engine):
    """
    Tests the run method of the ConfidenceEngine with sample data.
    """
    frequency = {'01': 80, '02': 20}
    cycles = {'01': {'cycle_score': 90, 'status': 'DUE'}, '02': {'cycle_score': 10, 'status': 'EXHAUSTED'}}
    digits = {'01': {'digit_score': 70}, '02': {'digit_score': 30}}
    momentum = {'01': 150, '02': 50}
    sample_size = 150

    results = confidence_engine.run(frequency, cycles, digits, momentum, sample_size)

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0][0] == '01'  # Jodi '01' should have the highest score
    assert results[1][0] == '02'
    assert 0 <= results[0][1] <= 100  # Score should be between 0 and 100
    assert isinstance(results[0][2], list)  # Tags should be a list
