import pytest
import sys
import os
from pathlib import Path
import numpy as np

# Add the parent directory to sys.path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from beast_sensordata.sensor_fusion import BeASTSensorFusion

@pytest.fixture
def sensor_fusion():
    return BeASTSensorFusion()

@pytest.fixture
def sample_sensor_data():
    return [
        {
            "time": [2023, 10, 27, 10, 0, 0],
            "side": "left",
            "devices": [
                {"name": "ppg", "status": "ok", "freq": 100, "data": [75, 98]}
            ]
        },
        {
            "time": [2023, 10, 27, 10, 0, 0],
            "side": "right",
            "devices": [
                {"name": "ppg", "status": "ok", "freq": 100, "data": [78, 99]}
            ]
        },
        {
            "time": [2023, 10, 27, 10, 0, 1],
            "side": "left",
            "devices": [
                {"name": "ppg", "status": "ok", "freq": 100, "data": [76, 98]}
            ]
        },
        {
            "time": [2023, 10, 27, 10, 0, 1],
            "side": "right",
            "devices": [
                {"name": "ppg", "status": "ok", "freq": 100, "data": [79, 99]}
            ]
        }
    ]

def test_extract_sensor_by_type(sensor_fusion, sample_sensor_data):
    """Test extracting sensor data by type (ppg)"""
    result = sensor_fusion.extract_sensor_by_type(sample_sensor_data, "ppg")
    
    assert len(result['left']) == 2
    assert len(result['right']) == 2
    assert result['left'][0]['data'] == [75, 98]
    assert result['right'][0]['data'] == [78, 99]

def test_combine_bilateral_data(sensor_fusion, sample_sensor_data):
    """Test combining left and right ear data by timestamp"""
    ppg_data = sensor_fusion.extract_sensor_by_type(sample_sensor_data, "ppg")
    combined = sensor_fusion.combine_bilateral_data(ppg_data['left'], ppg_data['right'])
    
    assert len(combined) == 2
    
    # First timestamp has both left and right
    assert combined[0]['left'] is not None
    assert combined[0]['right'] is not None
    assert combined[0]['time'] == [2023, 10, 27, 10, 0, 0]
    
    # Second timestamp has both left and right
    assert combined[1]['left'] is not None
    assert combined[1]['right'] is not None
    assert combined[1]['time'] == [2023, 10, 27, 10, 0, 1]

def test_extract_ppg_features(sensor_fusion, sample_sensor_data):
    """Test extracting features from PPG data"""
    ppg_data = sensor_fusion.extract_sensor_by_type(sample_sensor_data, "ppg")
    combined = sensor_fusion.combine_bilateral_data(ppg_data['left'], ppg_data['right'])
    features = sensor_fusion.extract_ppg_features(combined)
    
    # Check if keys exist
    assert 'left_hr_mean' in features
    assert 'right_hr_mean' in features
    
    # Check calculations
    # Left HR: [75, 76] -> mean 75.5
    assert features['left_hr_mean'] == 75.5
    # Right HR: [78, 79] -> mean 78.5
    assert features['right_hr_mean'] == 78.5

def test_empty_data(sensor_fusion):
    """Test handling of empty data"""
    empty_data = []
    result = sensor_fusion.extract_sensor_by_type(empty_data, "ppg")
    assert result['left'] == []
    assert result['right'] == []
    
    combined = sensor_fusion.combine_bilateral_data([], [])
    assert combined == []
    
    features = sensor_fusion.extract_ppg_features([])
    assert features == {}
