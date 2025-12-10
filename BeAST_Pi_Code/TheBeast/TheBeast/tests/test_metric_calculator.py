import pytest
import sys
import os
from pathlib import Path
import numpy as np

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from beast.processing.metric_calculator import MetricCalculator, Zone

@pytest.fixture
def calculator():
    return MetricCalculator()

@pytest.fixture
def normal_features():
    """Features representing a normal/relaxed state"""
    return {
        'left_theta_mean': 10.0,
        'right_theta_mean': 10.0,
        'left_alpha_mean': 20.0,  # High alpha = relaxed
        'right_alpha_mean': 20.0,
        'left_hrv': 80.0,         # High HRV = relaxed
        'right_hrv': 80.0,
        'left_hr': 65.0,
        'right_hr': 65.0
    }

@pytest.fixture
def stressed_features():
    """Features representing a stressed/high load state"""
    return {
        'left_theta_mean': 30.0,  # High theta = working memory load
        'right_theta_mean': 30.0,
        'left_alpha_mean': 5.0,   # Low alpha = focused/stressed
        'right_alpha_mean': 5.0,
        'left_hrv': 20.0,         # Low HRV = stress
        'right_hrv': 20.0,
        'left_hr': 95.0,
        'right_hr': 95.0
    }

def test_cognitive_load_calculation(calculator, normal_features, stressed_features):
    """Test cognitive load calculation for different states"""
    # Test normal state
    normal_result = calculator._calc_cognitive_load(normal_features)
    assert normal_result.name == 'cognitive_load'
    # Should be low load
    assert normal_result.value < 50
    assert normal_result.zone == Zone.GREEN
    
    # Test stressed state
    stressed_result = calculator._calc_cognitive_load(stressed_features)
    # Should be high load
    assert stressed_result.value > 50
    assert stressed_result.zone in [Zone.YELLOW, Zone.ORANGE, Zone.RED]

def test_tiredness_calculation(calculator, normal_features):
    """Test tiredness calculation"""
    # Modify features to simulate tiredness (High Alpha, Low HR)
    tired_features = normal_features.copy()
    tired_features['left_alpha_mean'] = 40.0
    tired_features['right_alpha_mean'] = 40.0
    tired_features['left_hr'] = 55.0
    tired_features['right_hr'] = 55.0
    
    result = calculator._calc_tiredness(tired_features)
    assert result.name == 'tiredness'
    # Should be high tiredness
    assert result.value > 50

def test_zone_classification(calculator):
    """Test zone classification logic"""
    # Standard metric (lower is better)
    assert calculator._classify_zone(30, 'cognitive_load') == Zone.GREEN
    assert calculator._classify_zone(50, 'cognitive_load') == Zone.YELLOW
    assert calculator._classify_zone(70, 'cognitive_load') == Zone.ORANGE
    assert calculator._classify_zone(90, 'cognitive_load') == Zone.RED
    
    # Inverted metric (higher is better) - e.g. attention_focus [80, 60, 40]
    assert calculator._classify_zone(90, 'attention_focus') == Zone.GREEN
    assert calculator._classify_zone(70, 'attention_focus') == Zone.YELLOW
    assert calculator._classify_zone(50, 'attention_focus') == Zone.ORANGE
    assert calculator._classify_zone(30, 'attention_focus') == Zone.RED

def test_calculate_all_metrics(calculator, normal_features):
    """Test that all metrics are calculated and returned"""
    results = calculator.calculate_all_metrics(normal_features)
    
    expected_metrics = [
        'cognitive_load', 'tiredness', 'fatigue', 'attention_focus',
        'stress_index', 'neurovascular_coupling_index', 
        'metabolic_stress_index', 'compensation_cognitive_load',
        'fatigue_severity_score', 'attention_capacity'
    ]
    
    for metric in expected_metrics:
        assert metric in results
        assert results[metric].name == metric
        assert isinstance(results[metric].value, float)
        assert isinstance(results[metric].zone, Zone)

def test_history_tracking(calculator, normal_features):
    """Test that metric history is updated"""
    # Initial state
    assert len(calculator.metric_history['cognitive_load']) == 0
    
    # Calculate once
    calculator.calculate_all_metrics(normal_features)
    assert len(calculator.metric_history['cognitive_load']) == 1
    
    # Calculate again
    calculator.calculate_all_metrics(normal_features)
    assert len(calculator.metric_history['cognitive_load']) == 2
