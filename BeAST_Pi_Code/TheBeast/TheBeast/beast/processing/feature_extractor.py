#!/usr/bin/env python3
"""
BeAST Feature Extractor
=======================
Extracts physiological and cognitive features from raw sensor data.
Based on sensor_fusion.py patterns.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger('BeAST.Features')


class FeatureExtractor:
    """
    Extract features from BeAST sensor data.
    
    Features extracted:
    - Heart rate and HRV from PPG
    - EEG band powers (delta, theta, alpha, beta, gamma)
    - Motion features from IMU
    - Temperature stability
    - Bioimpedance changes
    """
    
    # EEG frequency bands
    EEG_BANDS = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 50.0)
    }
    
    def __init__(self, sample_rate: int = 250, window_size: int = 250):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Sensor sample rate in Hz
            window_size: Analysis window size in samples
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        
        # Buffers for each sensor type
        self.buffers = {
            'left_ppg_ir': deque(maxlen=window_size),
            'left_ppg_red': deque(maxlen=window_size),
            'right_ppg_ir': deque(maxlen=window_size),
            'right_ppg_red': deque(maxlen=window_size),
            'left_eeg': [deque(maxlen=window_size) for _ in range(8)],
            'right_eeg': [deque(maxlen=window_size) for _ in range(8)],
            'left_accel': [deque(maxlen=window_size) for _ in range(3)],
            'right_accel': [deque(maxlen=window_size) for _ in range(3)],
            'left_gyro': [deque(maxlen=window_size) for _ in range(3)],
            'right_gyro': [deque(maxlen=window_size) for _ in range(3)],
            'left_temp': deque(maxlen=window_size),
            'right_temp': deque(maxlen=window_size),
            'left_bioz': deque(maxlen=window_size),
            'right_bioz': deque(maxlen=window_size)
        }
        
        # Peak detection state
        self.last_peaks = {'left': [], 'right': []}
        
    def add_sample(self, data: Dict):
        """
        Add a sensor sample to buffers.
        
        Args:
            data: Raw sensor data dict with 'left' and 'right' keys
        """
        for side in ['left', 'right']:
            if side not in data:
                continue
                
            side_data = data[side]
            
            # PPG
            if 'ppg_ir' in side_data:
                self.buffers[f'{side}_ppg_ir'].append(side_data['ppg_ir'])
            if 'ppg_red' in side_data:
                self.buffers[f'{side}_ppg_red'].append(side_data['ppg_red'])
                
            # EEG
            if 'eeg' in side_data:
                for i, val in enumerate(side_data['eeg'][:8]):
                    self.buffers[f'{side}_eeg'][i].append(val)
                    
            # IMU
            if 'imu_accel' in side_data:
                for i, val in enumerate(side_data['imu_accel'][:3]):
                    self.buffers[f'{side}_accel'][i].append(val)
            if 'imu_gyro' in side_data:
                for i, val in enumerate(side_data['imu_gyro'][:3]):
                    self.buffers[f'{side}_gyro'][i].append(val)
                    
            # Temperature
            if 'temperature' in side_data:
                self.buffers[f'{side}_temp'].append(side_data['temperature'])
                
            # Bioimpedance
            if 'bioimpedance' in side_data:
                self.buffers[f'{side}_bioz'].append(side_data['bioimpedance'])
                
    def extract_features(self) -> Dict[str, float]:
        """
        Extract all features from current buffer state.
        
        Returns:
            Dict of feature name -> value
        """
        features = {}
        
        # PPG features (heart rate, SpO2 proxy)
        for side in ['left', 'right']:
            ppg_ir = np.array(self.buffers[f'{side}_ppg_ir'])
            ppg_red = np.array(self.buffers[f'{side}_ppg_red'])
            
            if len(ppg_ir) >= self.window_size // 2:
                hr, hrv = self._extract_hr_features(ppg_ir)
                features[f'{side}_hr'] = hr
                features[f'{side}_hrv'] = hrv
                
                if len(ppg_red) >= len(ppg_ir) // 2:
                    spo2 = self._estimate_spo2(ppg_ir, ppg_red)
                    features[f'{side}_spo2'] = spo2
                    
        # EEG features (band powers)
        for side in ['left', 'right']:
            for ch in range(8):
                eeg = np.array(self.buffers[f'{side}_eeg'][ch])
                if len(eeg) >= self.window_size // 2:
                    band_powers = self._extract_band_powers(eeg)
                    for band, power in band_powers.items():
                        features[f'{side}_eeg_ch{ch}_{band}'] = power
                        
            # Aggregate EEG features
            if f'{side}_eeg_ch0_alpha' in features:
                features[f'{side}_alpha_mean'] = np.mean([
                    features.get(f'{side}_eeg_ch{i}_alpha', 0) for i in range(8)
                ])
                features[f'{side}_beta_mean'] = np.mean([
                    features.get(f'{side}_eeg_ch{i}_beta', 0) for i in range(8)
                ])
                features[f'{side}_theta_mean'] = np.mean([
                    features.get(f'{side}_eeg_ch{i}_theta', 0) for i in range(8)
                ])
                
        # Motion features
        for side in ['left', 'right']:
            accel = np.array([list(self.buffers[f'{side}_accel'][i]) for i in range(3)])
            if accel.shape[1] >= 10:
                features[f'{side}_motion_magnitude'] = np.mean(np.linalg.norm(accel, axis=0))
                features[f'{side}_motion_variance'] = np.mean(np.var(accel, axis=1))
                
        # Temperature features
        for side in ['left', 'right']:
            temp = np.array(self.buffers[f'{side}_temp'])
            if len(temp) >= 10:
                features[f'{side}_temp_mean'] = np.mean(temp)
                features[f'{side}_temp_std'] = np.std(temp)
                
        # Bioimpedance features
        for side in ['left', 'right']:
            bioz = np.array(self.buffers[f'{side}_bioz'])
            if len(bioz) >= 10:
                features[f'{side}_bioz_mean'] = np.mean(bioz)
                features[f'{side}_bioz_std'] = np.std(bioz)
                
        # Bilateral comparison features
        if 'left_hr' in features and 'right_hr' in features:
            features['hr_bilateral_diff'] = abs(features['left_hr'] - features['right_hr'])
            
        if 'left_alpha_mean' in features and 'right_alpha_mean' in features:
            features['alpha_bilateral_ratio'] = (
                features['left_alpha_mean'] / (features['right_alpha_mean'] + 1e-6)
            )
            
        return features
        
    def _extract_hr_features(self, ppg: np.ndarray) -> Tuple[float, float]:
        """
        Extract heart rate and HRV from PPG signal.
        
        Returns:
            (heart_rate_bpm, hrv_ms)
        """
        if len(ppg) < 50:
            return 0.0, 0.0
            
        # Bandpass filter (0.5-4 Hz)
        ppg_filtered = self._bandpass_filter(ppg, 0.5, 4.0)
        
        # Find peaks
        peaks = self._find_peaks(ppg_filtered)
        
        if len(peaks) < 2:
            return 0.0, 0.0
            
        # Calculate RR intervals
        rr_intervals = np.diff(peaks) / self.sample_rate * 1000  # ms
        
        # Filter physiological range (300-2000ms = 30-200 bpm)
        rr_valid = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
        
        if len(rr_valid) < 2:
            return 0.0, 0.0
            
        # Heart rate from mean RR
        hr = 60000 / np.mean(rr_valid)
        
        # HRV as RMSSD
        rr_diff = np.diff(rr_valid)
        hrv = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0.0
        
        return float(hr), float(hrv)
        
    def _estimate_spo2(self, ppg_ir: np.ndarray, ppg_red: np.ndarray) -> float:
        """
        Estimate SpO2 from dual-wavelength PPG.
        
        Returns:
            Estimated SpO2 percentage
        """
        # Simple ratio of ratios method
        # R = (AC_red / DC_red) / (AC_ir / DC_ir)
        
        ac_ir = np.std(ppg_ir)
        dc_ir = np.mean(ppg_ir)
        ac_red = np.std(ppg_red)
        dc_red = np.mean(ppg_red)
        
        if dc_ir < 1 or dc_red < 1:
            return 0.0
            
        r = (ac_red / dc_red) / (ac_ir / dc_ir + 1e-6)
        
        # Empirical calibration curve
        # SpO2 = 110 - 25 * R (simplified)
        spo2 = 110 - 25 * r
        
        # Clamp to valid range
        return float(np.clip(spo2, 70, 100))
        
    def _extract_band_powers(self, eeg: np.ndarray) -> Dict[str, float]:
        """
        Extract EEG band powers using FFT.
        
        Returns:
            Dict of band name -> power
        """
        if len(eeg) < 32:
            return {band: 0.0 for band in self.EEG_BANDS}
            
        # Compute FFT
        n = len(eeg)
        fft_vals = np.abs(np.fft.rfft(eeg))
        freqs = np.fft.rfftfreq(n, 1 / self.sample_rate)
        
        # Calculate band powers
        powers = {}
        for band, (low, high) in self.EEG_BANDS.items():
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                powers[band] = float(np.mean(fft_vals[mask] ** 2))
            else:
                powers[band] = 0.0
                
        return powers
        
    def _bandpass_filter(self, signal: np.ndarray, low: float, high: float) -> np.ndarray:
        """Simple bandpass filter using FFT"""
        n = len(signal)
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, 1 / self.sample_rate)
        
        # Zero out frequencies outside band
        fft[(freqs < low) | (freqs > high)] = 0
        
        return np.fft.irfft(fft, n)
        
    def _find_peaks(self, signal: np.ndarray, min_distance: int = 50) -> np.ndarray:
        """Find peaks in signal"""
        peaks = []
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
                    
        return np.array(peaks)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing FeatureExtractor...")
    
    extractor = FeatureExtractor(sample_rate=250, window_size=250)
    
    # Generate fake data
    import random
    import math
    
    for t in range(500):  # 2 seconds of data
        t_sec = t / 250
        data = {
            'left': {
                'ppg_ir': 50000 + 5000 * math.sin(2 * math.pi * 1.2 * t_sec) + random.gauss(0, 500),
                'ppg_red': 40000 + 4000 * math.sin(2 * math.pi * 1.2 * t_sec) + random.gauss(0, 400),
                'eeg': [random.gauss(0, 10) + 5 * math.sin(2 * math.pi * 10 * t_sec) for _ in range(8)],
                'imu_accel': [random.gauss(0, 0.1), random.gauss(0, 0.1), 9.8],
                'imu_gyro': [random.gauss(0, 1) for _ in range(3)],
                'temperature': 36.5,
                'bioimpedance': 500
            },
            'right': {
                'ppg_ir': 51000 + 5100 * math.sin(2 * math.pi * 1.2 * t_sec) + random.gauss(0, 510),
                'ppg_red': 41000 + 4100 * math.sin(2 * math.pi * 1.2 * t_sec) + random.gauss(0, 410),
                'eeg': [random.gauss(0, 10) + 5 * math.sin(2 * math.pi * 10 * t_sec + 0.1) for _ in range(8)],
                'imu_accel': [random.gauss(0, 0.1), random.gauss(0, 0.1), 9.8],
                'imu_gyro': [random.gauss(0, 1) for _ in range(3)],
                'temperature': 36.6,
                'bioimpedance': 505
            }
        }
        extractor.add_sample(data)
        
    # Extract features
    features = extractor.extract_features()
    
    print(f"\nExtracted {len(features)} features:")
    for name, value in sorted(features.items())[:20]:
        print(f"  {name}: {value:.2f}")
        
    print("\nTest complete!")
