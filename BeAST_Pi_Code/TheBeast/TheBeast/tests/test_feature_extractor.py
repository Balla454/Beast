import pytest
import sys
import os
from pathlib import Path
import numpy as np
from collections import deque

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from beast.processing.feature_extractor import FeatureExtractor

@pytest.fixture
def extractor():
    return FeatureExtractor(sample_rate=100, window_size=50)

def test_initialization(extractor):
    """Test that buffers are initialized correctly"""
    assert isinstance(extractor.buffers['left_ppg_ir'], deque)
    assert len(extractor.buffers['left_eeg']) == 8
    assert extractor.buffers['left_ppg_ir'].maxlen == 50

def test_add_sample(extractor):
    """Test adding a single sample to buffers"""
    sample = {
        'left': {
            'ppg_ir': 1000,
            'ppg_red': 2000,
            'eeg': [1, 2, 3, 4, 5, 6, 7, 8],
            'imu_accel': [0.1, 0.2, 0.3],
            'temperature': 36.5
        },
        'right': {
            'ppg_ir': 1100
        }
    }
    
    extractor.add_sample(sample)
    
    assert len(extractor.buffers['left_ppg_ir']) == 1
    assert extractor.buffers['left_ppg_ir'][0] == 1000
    assert len(extractor.buffers['left_eeg'][0]) == 1
    assert extractor.buffers['left_eeg'][0][0] == 1
    assert len(extractor.buffers['left_accel'][0]) == 1
    assert extractor.buffers['left_accel'][0][0] == 0.1
    assert len(extractor.buffers['left_temp']) == 1
    
    assert len(extractor.buffers['right_ppg_ir']) == 1
    assert extractor.buffers['right_ppg_ir'][0] == 1100

def test_extract_features_empty(extractor):
    """Test extracting features when buffers are empty"""
    features = extractor.extract_features()
    assert isinstance(features, dict)
    # Should be mostly empty or zero, but return a dict
    assert len(features) == 0

def test_extract_features_populated(extractor):
    """Test extracting features with populated buffers"""
    # Fill buffers with sine wave for PPG to simulate heart rate
    t = np.linspace(0, 5, 100) # 5 seconds at 20Hz (approx)
    # 1 Hz sine wave = 60 BPM
    signal = np.sin(2 * np.pi * 1.0 * t) + 10 
    
    # Manually populate buffers to bypass add_sample loop for speed
    extractor.buffers['left_ppg_ir'].extend(signal)
    extractor.buffers['right_ppg_ir'].extend(signal)
    
    # Populate EEG with random noise
    for i in range(8):
        extractor.buffers['left_eeg'][i].extend(np.random.random(100))
        extractor.buffers['right_eeg'][i].extend(np.random.random(100))
        
    # Populate Temp
    extractor.buffers['left_temp'].extend([36.5] * 100)
    
    # Mock internal methods that require complex signal processing
    # We don't want to test scipy signal processing here, just the flow
    extractor._extract_hr_features = lambda x: (60.0, 50.0) # Mock return 60 BPM, 50ms HRV
    extractor._estimate_spo2 = lambda x, y: 98.0
    extractor._extract_band_powers = lambda x: {'alpha': 10.0, 'beta': 5.0, 'theta': 8.0}
    
    features = extractor.extract_features()
    
    assert features['left_hr'] == 60.0
    assert features['left_hrv'] == 50.0
    assert features['left_alpha_mean'] == 10.0
    assert features['left_temp_mean'] == 36.5
    
    # Check bilateral features
    assert 'hr_bilateral_diff' in features
    assert features['hr_bilateral_diff'] == 0.0 # 60 - 60

def test_buffer_overflow(extractor):
    """Test that buffers respect maxlen"""
    # Maxlen is 50
    for i in range(100):
        extractor.add_sample({'left': {'ppg_ir': i}})
        
    assert len(extractor.buffers['left_ppg_ir']) == 50
    assert extractor.buffers['left_ppg_ir'][-1] == 99
    assert extractor.buffers['left_ppg_ir'][0] == 50
