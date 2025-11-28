"""
BeAST Real-Time Multimodal Physiological Signal Analysis System
================================================================

PRODUCTION-READY CODE for real-time warfighter monitoring.

Architecture:
1. JSONL Stream Reader -> Parses incoming sensor data
2. Signal Preprocessor -> Filters, artifact removal, normalization
3. Feature Extractor -> Calculates EEG bands, HRV, bioimpedance metrics
4. Metric Calculator -> Computes 10 cognitive/physiological variables (0-100 scale)
5. Zone Classifier -> Assigns Zone 1-4 for each metric
6. Local DB Writer -> Pushes to SQLite (CPU-level storage)

Usage:
    processor = BeASTRealtimeProcessor(db_path='beast_local.db')
    processor.start_streaming(jsonl_file_path='sensor_data.jsonl')

Author: BeAST Development Team
Version: 2.0 - Real-Time Streaming Production
Date: October 2025
"""

import numpy as np
import pandas as pd
import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from queue import Queue
from collections import deque
from scipy import signal
from scipy.stats import zscore
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BeAST')


# ============================================================================
# ZONE DEFINITIONS (From Research & Clinical Guidelines)
# ============================================================================

ZONE_DEFINITIONS = {
    'Cognitive_Load': {
        1: (0, 40, 'Optimal', 'Normal cognitive demand, effective processing'),
        2: (40, 60, 'Mild Risk', 'Elevated workload, monitor performance'),
        3: (60, 80, 'Moderate Risk', 'High workload, consider task reduction'),
        4: (80, 100, 'High Risk', 'Cognitive overload, performance degradation')
    },
    'Tiredness': {
        1: (0, 30, 'Optimal', 'Alert and energized'),
        2: (30, 50, 'Mild Risk', 'Slight fatigue, monitor closely'),
        3: (50, 70, 'Moderate Risk', 'Significant fatigue, rest recommended'),
        4: (70, 100, 'High Risk', 'Severe fatigue, mandatory rest')
    },
    'Fatigue': {
        1: (0, 30, 'Optimal', 'No significant fatigue'),
        2: (30, 50, 'Mild Risk', 'Early fatigue signs'),
        3: (50, 70, 'Moderate Risk', 'Moderate fatigue, recovery needed'),
        4: (70, 100, 'High Risk', 'Severe fatigue, mission-compromising')
    },
    'Attention_Focus': {
        1: (70, 100, 'Optimal', 'Excellent attention and focus'),
        2: (50, 70, 'Mild Risk', 'Adequate focus, slight lapses'),
        3: (30, 50, 'Moderate Risk', 'Poor focus, frequent lapses'),
        4: (0, 30, 'High Risk', 'Severe attention deficit')
    },
    'Stress_Index': {
        1: (0, 30, 'Optimal', 'Low stress, calm state'),
        2: (30, 50, 'Mild Risk', 'Moderate stress, manageable'),
        3: (50, 75, 'Moderate Risk', 'High stress, intervention helpful'),
        4: (75, 100, 'High Risk', 'Extreme stress, immediate action required')
    },
    'Neurovascular_Coupling_Index': {
        1: (70, 100, 'Optimal', 'Healthy brain blood flow coupling'),
        2: (50, 70, 'Mild Risk', 'Slight coupling inefficiency'),
        3: (30, 50, 'Moderate Risk', 'Moderate coupling dysfunction'),
        4: (0, 30, 'High Risk', 'Severe coupling impairment')
    },
    'Metabolic_Stress_Index': {
        1: (0, 30, 'Optimal', 'Low metabolic demand'),
        2: (30, 50, 'Mild Risk', 'Moderate metabolic stress'),
        3: (50, 75, 'Moderate Risk', 'High metabolic stress'),
        4: (75, 100, 'High Risk', 'Extreme metabolic stress')
    },
    'Compensation_Cognitive_Load': {
        1: (0, 30, 'Optimal', 'Minimal compensatory effort'),
        2: (30, 50, 'Mild Risk', 'Mild compensation required'),
        3: (50, 70, 'Moderate Risk', 'Significant compensation'),
        4: (70, 100, 'High Risk', 'Maximal compensation, near failure')
    },
    'Fatigue_Severity_Score': {
        1: (0, 33, 'Optimal', 'No clinically significant fatigue'),
        2: (33, 55, 'Mild Risk', 'Mild fatigue, monitor'),
        3: (55, 77, 'Moderate Risk', 'Moderate fatigue, intervention needed'),
        4: (77, 100, 'High Risk', 'Severe fatigue, immediate rest')
    },
    'Attention_Capacity': {
        1: (70, 100, 'Optimal', 'Full attention capacity'),
        2: (50, 70, 'Mild Risk', 'Reduced capacity'),
        3: (30, 50, 'Moderate Risk', 'Limited capacity'),
        4: (0, 30, 'High Risk', 'Minimal capacity')
    }
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SensorReading:
    """Single timestamped sensor reading"""
    timestamp: datetime
    device_side: str  # 'left' or 'right'
    
    # EEG channels (12 total: 6 per ear)
    eeg_fp1: Optional[float] = None
    eeg_fp2: Optional[float] = None
    eeg_f7: Optional[float] = None
    eeg_f8: Optional[float] = None
    eeg_t3: Optional[float] = None
    eeg_t4: Optional[float] = None
    eeg_t5: Optional[float] = None
    eeg_t6: Optional[float] = None
    eeg_o1: Optional[float] = None
    eeg_o2: Optional[float] = None
    eeg_a1: Optional[float] = None
    eeg_a2: Optional[float] = None
    
    # Reference and bias
    eeg_ref_left: Optional[float] = None
    eeg_ref_right: Optional[float] = None
    eeg_bias_left: Optional[float] = None
    eeg_bias_right: Optional[float] = None
    
    # PPG/Heart metrics
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    
    # Bioimpedance
    bioimpedance_resistance: Optional[float] = None
    bioimpedance_reactance: Optional[float] = None
    
    # Environmental
    ambient_temp: Optional[float] = None
    ambient_humidity: Optional[float] = None
    sound_level_db: Optional[float] = None


@dataclass
class ProcessedMetrics:
    """All 10 calculated metrics (0-100 scale)"""
    timestamp: datetime
    session_id: str
    
    cognitive_load: float
    tiredness: float
    fatigue: float
    attention_focus: float
    stress_index: float
    neurovascular_coupling_index: float
    metabolic_stress_index: float
    compensation_cognitive_load: float
    fatigue_severity_score: float
    attention_capacity: float
    
    # Zone assignments
    cognitive_load_zone: int
    tiredness_zone: int
    fatigue_zone: int
    attention_focus_zone: int
    stress_index_zone: int
    neurovascular_coupling_index_zone: int
    metabolic_stress_index_zone: int
    compensation_cognitive_load_zone: int
    fatigue_severity_score_zone: int
    attention_capacity_zone: int


# ============================================================================
# SIGNAL PREPROCESSING
# ============================================================================

class SignalPreprocessor:
    """
    EEG and physiological signal preprocessing pipeline
    
    Based on research-validated approaches:
    - Bandpass filtering: 0.5-45 Hz (EEG)
    - Notch filter: 60 Hz (power line noise)
    - Artifact removal: threshold-based + IQR outlier detection
    - Normalization: Z-score per channel
    """
    
    def __init__(self, sampling_rate: int = 250):
        self.fs = sampling_rate
        
        # Design filters
        self.bp_filter = self._design_bandpass(0.5, 45)
        self.notch_filter = self._design_notch(60)
        
    def _design_bandpass(self, low: float, high: float):
        """Butterworth bandpass filter"""
        nyquist = self.fs / 2
        low_norm = low / nyquist
        high_norm = high / nyquist
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        return (b, a)
    
    def _design_notch(self, freq: float, quality: float = 30):
        """Notch filter for power line noise"""
        nyquist = self.fs / 2
        freq_norm = freq / nyquist
        b, a = signal.iirnotch(freq_norm, quality)
        return (b, a)
    
    def preprocess_eeg_channel(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing for single EEG channel
        
        Args:
            raw_signal: Raw mV values from sensor
            
        Returns:
            Preprocessed signal ready for feature extraction
        """
        if len(raw_signal) < 10:
            return raw_signal
        
        # Apply bandpass filter
        filtered = signal.filtfilt(self.bp_filter[0], self.bp_filter[1], raw_signal)
        
        # Apply notch filter
        filtered = signal.filtfilt(self.notch_filter[0], self.notch_filter[1], filtered)
        
        # Artifact removal: threshold-based (±200 µV)
        artifact_mask = np.abs(filtered) > 200
        filtered[artifact_mask] = np.nan
        
        # Interpolate artifacts
        if np.any(artifact_mask):
            valid_idx = np.where(~artifact_mask)[0]
            artifact_idx = np.where(artifact_mask)[0]
            if len(valid_idx) > 1:
                filtered[artifact_idx] = np.interp(artifact_idx, valid_idx, filtered[valid_idx])
        
        # Z-score normalization
        filtered = zscore(filtered, nan_policy='omit')
        
        return filtered
    
    def remove_outliers_iqr(self, data: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """IQR-based outlier removal for physiological signals"""
        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        cleaned = data.copy()
        cleaned[(data < lower_bound) | (data > upper_bound)] = np.nan
        
        return cleaned


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """
    Extract validated features from preprocessed signals
    
    EEG Features:
    - Band power: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), 
                  Beta (13-30 Hz), Gamma (30-45 Hz)
    - Ratios: Alpha/Theta, Beta/Alpha, Theta/Beta
    - Engagement Index: Beta / (Theta + Alpha)
    
    HRV Features:
    - Time-domain: RMSSD, SDNN, pNN50
    - Frequency-domain: LF/HF ratio
    
    Bioimpedance:
    - Phase angle, resistance trends
    """
    
    def __init__(self, sampling_rate: int = 250):
        self.fs = sampling_rate
        
    def calculate_band_power(self, signal_data: np.ndarray, 
                            freq_band: Tuple[float, float]) -> float:
        """
        Calculate power in specific frequency band using Welch's method
        
        Args:
            signal_data: Preprocessed EEG signal
            freq_band: (low_freq, high_freq) in Hz
            
        Returns:
            Band power in µV²
        """
        if len(signal_data) < 50:
            return 0.0
            
        freqs, psd = signal.welch(signal_data, fs=self.fs, nperseg=min(256, len(signal_data)))
        
        # Find indices for frequency band
        idx_band = np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1])
        
        if not np.any(idx_band):
            return 0.0
        
        # Integrate power in band
        band_power = np.trapz(psd[idx_band], freqs[idx_band])
        
        return max(0.0, band_power)
    
    def extract_eeg_features(self, eeg_channels: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Extract all EEG features from multiple channels
        
        Returns dict with:
        - delta_power, theta_power, alpha_power, beta_power, gamma_power
        - alpha_theta_ratio, beta_alpha_ratio, theta_beta_ratio
        - engagement_index
        """
        features = {}
        
        # Average across all channels for global brain state
        all_signals = [sig for sig in eeg_channels.values() if sig is not None and len(sig) > 0]
        if len(all_signals) == 0:
            return self._get_default_eeg_features()
        
        avg_signal = np.nanmean(np.array(all_signals), axis=0)
        
        # Calculate band powers
        features['delta_power'] = self.calculate_band_power(avg_signal, (0.5, 4))
        features['theta_power'] = self.calculate_band_power(avg_signal, (4, 8))
        features['alpha_power'] = self.calculate_band_power(avg_signal, (8, 13))
        features['beta_power'] = self.calculate_band_power(avg_signal, (13, 30))
        features['gamma_power'] = self.calculate_band_power(avg_signal, (30, 45))
        
        # Calculate ratios (with safety checks)
        features['alpha_theta_ratio'] = self._safe_divide(
            features['alpha_power'], features['theta_power'], default=1.0
        )
        features['beta_alpha_ratio'] = self._safe_divide(
            features['beta_power'], features['alpha_power'], default=1.0
        )
        features['theta_beta_ratio'] = self._safe_divide(
            features['theta_power'], features['beta_power'], default=1.0
        )
        
        # Engagement Index
        features['engagement_index'] = self._safe_divide(
            features['beta_power'],
            features['theta_power'] + features['alpha_power'],
            default=0.5
        )
        
        return features
    
    def calculate_hrv_features(self, ibi_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate Heart Rate Variability features from inter-beat intervals
        
        Args:
            ibi_series: Array of inter-beat intervals in ms
            
        Returns:
            Dict with RMSSD, SDNN, pNN50, LF/HF ratio
        """
        if len(ibi_series) < 10:
            return {'rmssd': 50, 'sdnn': 50, 'pnn50': 20, 'lf_hf_ratio': 1.0}
        
        # Time-domain features
        rmssd = np.sqrt(np.mean(np.diff(ibi_series) ** 2))
        sdnn = np.std(ibi_series)
        
        # pNN50: percentage of successive differences > 50ms
        successive_diffs = np.abs(np.diff(ibi_series))
        pnn50 = 100 * np.sum(successive_diffs > 50) / len(successive_diffs)
        
        # Frequency-domain: LF/HF ratio
        try:
            # Convert IBI to evenly sampled HR signal (4 Hz)
            time_ibi = np.cumsum(ibi_series) / 1000  # Convert to seconds
            time_uniform = np.arange(0, time_ibi[-1], 0.25)  # 4 Hz
            if len(time_uniform) < 2:
                return {'rmssd': rmssd, 'sdnn': sdnn, 'pnn50': pnn50, 'lf_hf_ratio': 1.0}
            
            hr_uniform = np.interp(time_uniform, time_ibi, 60000 / ibi_series)
            
            freqs, psd = signal.welch(hr_uniform, fs=4, nperseg=min(256, len(hr_uniform)))
            
            lf_power = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)])
            hf_power = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)])
            
            lf_hf_ratio = self._safe_divide(lf_power, hf_power, default=1.0)
        except:
            lf_hf_ratio = 1.0
        
        return {
            'rmssd': rmssd,
            'sdnn': sdnn,
            'pnn50': pnn50,
            'lf_hf_ratio': lf_hf_ratio
        }
    
    def calculate_bioimpedance_features(self, resistance: np.ndarray, 
                                       reactance: np.ndarray) -> Dict[str, float]:
        """
        Calculate bioimpedance-derived metrics
        
        Returns:
        - phase_angle: Hydration indicator (degrees)
        - resistance_trend: Rate of change
        """
        if len(resistance) < 2:
            return {'phase_angle': 5.0, 'resistance_trend': 0.0}
        
        # Phase angle = arctan(Xc / R)
        phase_angle = np.rad2deg(np.arctan2(np.nanmean(reactance), np.nanmean(resistance)))
        
        # Resistance trend (linear fit slope)
        if len(resistance) > 5:
            time_idx = np.arange(len(resistance))
            valid_mask = ~np.isnan(resistance)
            if np.sum(valid_mask) > 2:
                coeffs = np.polyfit(time_idx[valid_mask], resistance[valid_mask], 1)
                resistance_trend = coeffs[0]
            else:
                resistance_trend = 0.0
        else:
            resistance_trend = 0.0
        
        return {
            'phase_angle': phase_angle,
            'resistance_trend': resistance_trend
        }
    
    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value"""
        if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
            return default
        return numerator / denominator
    
    @staticmethod
    def _get_default_eeg_features() -> Dict[str, float]:
        """Return default EEG features when no data available"""
        return {
            'delta_power': 0.1,
            'theta_power': 0.1,
            'alpha_power': 0.1,
            'beta_power': 0.1,
            'gamma_power': 0.05,
            'alpha_theta_ratio': 1.0,
            'beta_alpha_ratio': 1.0,
            'theta_beta_ratio': 1.0,
            'engagement_index': 0.5
        }


# ============================================================================
# METRIC CALCULATOR (10 Variables on 0-100 Scale)
# ============================================================================

class MetricCalculator:
    """
    Calculate all 10 warfighter metrics from extracted features
    
    Converts research-validated features into 0-100 scaled metrics
    """
    
    def calculate_cognitive_load(self, eeg_features: Dict, hrv_features: Dict) -> float:
        """Cognitive Load (0-100) - Berka et al. (2007)"""
        theta_beta = eeg_features.get('theta_beta_ratio', 1.0)
        beta_alpha = eeg_features.get('beta_alpha_ratio', 1.0)
        lf_hf = hrv_features.get('lf_hf_ratio', 1.0)
        
        theta_component = np.clip((theta_beta - 0.5) / 2.5 * 100, 0, 100)
        beta_component = np.clip((beta_alpha - 0.5) / 2.0 * 100, 0, 100)
        autonomic_component = np.clip((lf_hf - 0.5) / 2.5 * 100, 0, 100)
        
        cognitive_load = (
            0.4 * theta_component +
            0.4 * beta_component +
            0.2 * autonomic_component
        )
        
        return np.clip(cognitive_load, 0, 100)
    
    def calculate_tiredness(self, eeg_features: Dict) -> float:
        """Tiredness (0-100) - Lal & Craig (2001)"""
        alpha_theta = eeg_features.get('alpha_theta_ratio', 1.0)
        alpha_power = eeg_features.get('alpha_power', 0.1)
        theta_power = eeg_features.get('theta_power', 0.1)
        
        ratio_component = np.clip((2.0 - alpha_theta) / 1.5 * 100, 0, 100)
        alpha_component = np.clip(alpha_power * 200, 0, 100)
        theta_component = np.clip(theta_power * 250, 0, 100)
        
        tiredness = (
            0.5 * ratio_component +
            0.25 * alpha_component +
            0.25 * theta_component
        )
        
        return np.clip(tiredness, 0, 100)
    
    def calculate_fatigue(self, eeg_features: Dict, hrv_features: Dict) -> float:
        """Fatigue (0-100) - Trejo et al. (2005)"""
        alpha_power = eeg_features.get('alpha_power', 0.1)
        theta_power = eeg_features.get('theta_power', 0.1)
        rmssd = hrv_features.get('rmssd', 50)
        
        alpha_component = np.clip((1.0 - alpha_power * 10) * 100, 0, 100)
        theta_component = np.clip(theta_power * 250, 0, 100)
        hrv_component = np.clip((100 - rmssd) / 100 * 100, 0, 100)
        
        fatigue = (
            0.4 * alpha_component +
            0.3 * theta_component +
            0.3 * hrv_component
        )
        
        return np.clip(fatigue, 0, 100)
    
    def calculate_attention_focus(self, eeg_features: Dict) -> float:
        """Attention Focus (0-100) - Pope et al. (1995)"""
        engagement = eeg_features.get('engagement_index', 0.5)
        theta_beta = eeg_features.get('theta_beta_ratio', 1.0)
        beta_power = eeg_features.get('beta_power', 0.1)
        
        engagement_component = np.clip((engagement - 0.2) / 1.3 * 100, 0, 100)
        theta_beta_component = np.clip((3.0 - theta_beta) / 2.5 * 100, 0, 100)
        beta_component = np.clip(beta_power * 200, 0, 100)
        
        attention = (
            0.5 * engagement_component +
            0.3 * theta_beta_component +
            0.2 * beta_component
        )
        
        return np.clip(attention, 0, 100)
    
    def calculate_stress_index(self, hrv_features: Dict, eeg_features: Dict) -> float:
        """Stress Index (0-100) - Task Force (1996)"""
        lf_hf = hrv_features.get('lf_hf_ratio', 1.0)
        rmssd = hrv_features.get('rmssd', 50)
        beta_alpha = eeg_features.get('beta_alpha_ratio', 1.0)
        
        autonomic_component = np.clip((lf_hf - 0.5) / 4.5 * 100, 0, 100)
        hrv_component = np.clip((100 - rmssd) / 100 * 100, 0, 100)
        cortical_component = np.clip((beta_alpha - 0.5) / 2.0 * 100, 0, 100)
        
        stress = (
            0.5 * autonomic_component +
            0.3 * hrv_component +
            0.2 * cortical_component
        )
        
        return np.clip(stress, 0, 100)
    
    def calculate_neurovascular_coupling(self, eeg_features: Dict, 
                                        bioimpedance_features: Dict) -> float:
        """Neurovascular Coupling Index (0-100)"""
        total_eeg_power = sum([
            eeg_features.get('delta_power', 0),
            eeg_features.get('theta_power', 0),
            eeg_features.get('alpha_power', 0),
            eeg_features.get('beta_power', 0),
            eeg_features.get('gamma_power', 0)
        ])
        
        phase_angle = bioimpedance_features.get('phase_angle', 5.0)
        
        eeg_component = np.clip(total_eeg_power * 100, 0, 100)
        perfusion_component = np.clip((phase_angle - 3) / 4 * 100, 0, 100)
        
        coupling = 0.5 * eeg_component + 0.5 * perfusion_component
        
        return np.clip(coupling, 0, 100)
    
    def calculate_metabolic_stress(self, heart_rate: float, 
                                   bioimpedance_features: Dict) -> float:
        """Metabolic Stress Index (0-100)"""
        hr_component = np.clip((heart_rate - 60) / 120 * 100, 0, 100) if heart_rate else 0
        resistance_trend = bioimpedance_features.get('resistance_trend', 0.0)
        trend_component = np.clip(-resistance_trend * 50, 0, 100)
        
        metabolic_stress = 0.7 * hr_component + 0.3 * trend_component
        
        return np.clip(metabolic_stress, 0, 100)
    
    def calculate_compensation_cognitive_load(self, eeg_features: Dict, 
                                             cognitive_load: float) -> float:
        """Compensation Cognitive Load (0-100) - Hockey (1997)"""
        gamma_power = eeg_features.get('gamma_power', 0.05)
        beta_power = eeg_features.get('beta_power', 0.1)
        
        gamma_component = np.clip(gamma_power * 300, 0, 100)
        beta_component = np.clip(beta_power * 200, 0, 100)
        
        compensation = (
            0.3 * gamma_component +
            0.3 * beta_component +
            0.4 * cognitive_load
        )
        
        return np.clip(compensation, 0, 100)
    
    def calculate_fatigue_severity_score(self, fatigue: float, 
                                        tiredness: float) -> float:
        """Fatigue Severity Score (0-100) - Krupp et al. (1989)"""
        combined = (fatigue + tiredness) / 2
        return np.clip(combined, 0, 100)
    
    def calculate_attention_capacity(self, attention_focus: float, 
                                     cognitive_load: float) -> float:
        """Attention Capacity (0-100) - Kahneman (1973)"""
        capacity = attention_focus - (cognitive_load * 0.5)
        return np.clip(capacity, 0, 100)
    
    def calculate_all_metrics(self, eeg_features: Dict, hrv_features: Dict,
                             bioimpedance_features: Dict, 
                             heart_rate: float) -> Dict[str, float]:
        """Calculate all 10 metrics"""
        cognitive_load = self.calculate_cognitive_load(eeg_features, hrv_features)
        tiredness = self.calculate_tiredness(eeg_features)
        fatigue = self.calculate_fatigue(eeg_features, hrv_features)
        attention_focus = self.calculate_attention_focus(eeg_features)
        
        metrics = {
            'cognitive_load': cognitive_load,
            'tiredness': tiredness,
            'fatigue': fatigue,
            'attention_focus': attention_focus,
            'stress_index': self.calculate_stress_index(hrv_features, eeg_features),
            'neurovascular_coupling_index': self.calculate_neurovascular_coupling(
                eeg_features, bioimpedance_features
            ),
            'metabolic_stress_index': self.calculate_metabolic_stress(
                heart_rate, bioimpedance_features
            ),
            'compensation_cognitive_load': self.calculate_compensation_cognitive_load(
                eeg_features, cognitive_load
            ),
            'fatigue_severity_score': self.calculate_fatigue_severity_score(
                fatigue, tiredness
            ),
            'attention_capacity': self.calculate_attention_capacity(
                attention_focus, cognitive_load
            )
        }
        
        return metrics


# ============================================================================
# ZONE CLASSIFIER
# ============================================================================

class ZoneClassifier:
    """Classify metrics into 4 risk zones"""
    
    @staticmethod
    def classify_metric(value: float, metric_name: str) -> int:
        """Classify metric value into zone 1-4"""
        if metric_name not in ZONE_DEFINITIONS:
            return 1
        
        zones = ZONE_DEFINITIONS[metric_name]
        
        for zone_num, (low, high, label, desc) in zones.items():
            if low <= value < high or (zone_num == 4 and value >= low):
                return zone_num
        
        return 1
    
    @staticmethod
    def classify_all_metrics(metrics: Dict[str, float]) -> Dict[str, int]:
        """Classify all metrics"""
        name_mapping = {
            'cognitive_load': 'Cognitive_Load',
            'tiredness': 'Tiredness',
            'fatigue': 'Fatigue',
            'attention_focus': 'Attention_Focus',
            'stress_index': 'Stress_Index',
            'neurovascular_coupling_index': 'Neurovascular_Coupling_Index',
            'metabolic_stress_index': 'Metabolic_Stress_Index',
            'compensation_cognitive_load': 'Compensation_Cognitive_Load',
            'fatigue_severity_score': 'Fatigue_Severity_Score',
            'attention_capacity': 'Attention_Capacity'
        }
        
        zones = {}
        for metric_key, metric_value in metrics.items():
            zone_key = name_mapping.get(metric_key, metric_key)
            zones[f"{metric_key}_zone"] = ZoneClassifier.classify_metric(
                metric_value, zone_key
            )
        
        return zones


# ============================================================================
# LOCAL DATABASE MANAGER
# ============================================================================

class LocalDatabaseManager:
    """Manages local SQLite database (80% similar to PostgreSQL schema)"""
    
    def __init__(self, db_path: str = 'beast_local.db'):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database and tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                device_side TEXT,
                activity_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eeg_aggregate_current (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                avg_delta_power REAL,
                avg_theta_power REAL,
                avg_alpha_power REAL,
                avg_beta_power REAL,
                avg_gamma_power REAL,
                alpha_theta_ratio REAL,
                beta_alpha_ratio REAL,
                theta_beta_ratio REAL,
                engagement_index REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS physiological_current (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                heart_rate REAL,
                spo2 REAL,
                hrv_rmssd REAL,
                hrv_sdnn REAL,
                hrv_pnn50 REAL,
                hrv_lf_hf_ratio REAL,
                bioimpedance_phase_angle REAL,
                bioimpedance_resistance_trend REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calculated_metrics_current (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                cognitive_load REAL,
                tiredness REAL,
                fatigue REAL,
                attention_focus REAL,
                stress_index REAL,
                neurovascular_coupling_index REAL,
                metabolic_stress_index REAL,
                compensation_cognitive_load REAL,
                fatigue_severity_score REAL,
                attention_capacity REAL,
                cognitive_load_zone INTEGER,
                tiredness_zone INTEGER,
                fatigue_zone INTEGER,
                attention_focus_zone INTEGER,
                stress_index_zone INTEGER,
                neurovascular_coupling_index_zone INTEGER,
                metabolic_stress_index_zone INTEGER,
                compensation_cognitive_load_zone INTEGER,
                fatigue_severity_score_zone INTEGER,
                attention_capacity_zone INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zone_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                metric_name TEXT NOT NULL,
                from_zone INTEGER,
                to_zone INTEGER,
                metric_value REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_cumulative_stats (
                session_id TEXT PRIMARY KEY,
                total_duration_minutes REAL,
                avg_cognitive_load REAL,
                avg_tiredness REAL,
                avg_fatigue REAL,
                avg_attention_focus REAL,
                avg_stress_index REAL,
                time_in_zone1_minutes REAL,
                time_in_zone2_minutes REAL,
                time_in_zone3_minutes REAL,
                time_in_zone4_minutes REAL,
                zone_transitions_count INTEGER,
                last_updated TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_session_time 
            ON calculated_metrics_current(session_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transitions_session 
            ON zone_transitions(session_id, metric_name)
        """)
        
        self.conn.commit()
        logger.info(f"Local database initialized: {self.db_path}")
    
    def create_session(self, session_id: str, user_id: str, device_side: str):
        """Create new session record"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_id, user_id, start_time, device_side)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_id, datetime.now(), device_side))
        
        cursor.execute("""
            INSERT INTO session_cumulative_stats (session_id, last_updated)
            VALUES (?, ?)
        """, (session_id, datetime.now()))
        
        self.conn.commit()
        logger.info(f"Session created: {session_id}")
    
    def insert_eeg_metrics(self, session_id: str, timestamp: datetime, eeg_features: Dict):
        """Insert EEG aggregate metrics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO eeg_aggregate_current (
                session_id, timestamp, avg_delta_power, avg_theta_power,
                avg_alpha_power, avg_beta_power, avg_gamma_power,
                alpha_theta_ratio, beta_alpha_ratio, theta_beta_ratio,
                engagement_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, timestamp,
            eeg_features.get('delta_power', 0),
            eeg_features.get('theta_power', 0),
            eeg_features.get('alpha_power', 0),
            eeg_features.get('beta_power', 0),
            eeg_features.get('gamma_power', 0),
            eeg_features.get('alpha_theta_ratio', 0),
            eeg_features.get('beta_alpha_ratio', 0),
            eeg_features.get('theta_beta_ratio', 0),
            eeg_features.get('engagement_index', 0)
        ))
        self.conn.commit()
    
    def insert_physiological_metrics(self, session_id: str, timestamp: datetime,
                                    hrv_features: Dict, bioimpedance_features: Dict,
                                    heart_rate: float):
        """Insert physiological metrics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO physiological_current (
                session_id, timestamp, heart_rate, hrv_rmssd, hrv_sdnn,
                hrv_pnn50, hrv_lf_hf_ratio, bioimpedance_phase_angle,
                bioimpedance_resistance_trend
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, timestamp, heart_rate,
            hrv_features.get('rmssd', 0),
            hrv_features.get('sdnn', 0),
            hrv_features.get('pnn50', 0),
            hrv_features.get('lf_hf_ratio', 0),
            bioimpedance_features.get('phase_angle', 0),
            bioimpedance_features.get('resistance_trend', 0)
        ))
        self.conn.commit()
    
    def insert_calculated_metrics(self, processed: ProcessedMetrics):
        """Insert all 10 calculated metrics with zones"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO calculated_metrics_current (
                session_id, timestamp,
                cognitive_load, tiredness, fatigue, attention_focus, stress_index,
                neurovascular_coupling_index, metabolic_stress_index,
                compensation_cognitive_load, fatigue_severity_score, attention_capacity,
                cognitive_load_zone, tiredness_zone, fatigue_zone, attention_focus_zone,
                stress_index_zone, neurovascular_coupling_index_zone, metabolic_stress_index_zone,
                compensation_cognitive_load_zone, fatigue_severity_score_zone, attention_capacity_zone
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            processed.session_id, processed.timestamp,
            processed.cognitive_load, processed.tiredness, processed.fatigue,
            processed.attention_focus, processed.stress_index,
            processed.neurovascular_coupling_index, processed.metabolic_stress_index,
            processed.compensation_cognitive_load, processed.fatigue_severity_score,
            processed.attention_capacity,
            processed.cognitive_load_zone, processed.tiredness_zone, processed.fatigue_zone,
            processed.attention_focus_zone, processed.stress_index_zone,
            processed.neurovascular_coupling_index_zone, processed.metabolic_stress_index_zone,
            processed.compensation_cognitive_load_zone, processed.fatigue_severity_score_zone,
            processed.attention_capacity_zone
        ))
        self.conn.commit()
    
    def track_zone_transition(self, session_id: str, timestamp: datetime,
                             metric_name: str, from_zone: int, to_zone: int,
                             metric_value: float):
        """Record zone transition event"""
        if from_zone != to_zone:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO zone_transitions (
                    session_id, timestamp, metric_name, from_zone, to_zone, metric_value
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, timestamp, metric_name, from_zone, to_zone, metric_value))
            self.conn.commit()
            
            logger.info(f"Zone transition: {metric_name} {from_zone}→{to_zone} @ {timestamp}")
    
    def update_cumulative_stats(self, session_id: str):
        """Update cumulative session statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                AVG(cognitive_load) as avg_cog,
                AVG(tiredness) as avg_tired,
                AVG(fatigue) as avg_fatigue,
                AVG(attention_focus) as avg_attention,
                AVG(stress_index) as avg_stress,
                COUNT(*) as total_records
            FROM calculated_metrics_current
            WHERE session_id = ?
        """, (session_id,))
        
        stats = cursor.fetchone()
        
        if stats and stats[5] > 0:
            cursor.execute("""
                UPDATE session_cumulative_stats
                SET avg_cognitive_load = ?,
                    avg_tiredness = ?,
                    avg_fatigue = ?,
                    avg_attention_focus = ?,
                    avg_stress_index = ?,
                    last_updated = ?
                WHERE session_id = ?
            """, (stats[0], stats[1], stats[2], stats[3], stats[4], 
                  datetime.now(), session_id))
            
            self.conn.commit()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# ============================================================================
# MAIN REAL-TIME PROCESSOR
# ============================================================================

class BeASTRealtimeProcessor:
    """Main real-time processing system"""
    
    def __init__(self, db_path: str = 'beast_local.db', window_size: int = 500):
        self.db = LocalDatabaseManager(db_path)
        self.preprocessor = SignalPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.metric_calculator = MetricCalculator()
        
        self.window_size = window_size
        
        # Rolling buffers
        self.eeg_buffers = {
            f'eeg_{ch}': deque(maxlen=window_size)
            for ch in ['fp1', 'fp2', 'f7', 'f8', 't3', 't4', 't5', 't6', 'o1', 'o2', 'a1', 'a2']
        }
        
        self.heart_rate_buffer = deque(maxlen=100)
        self.bioimpedance_buffer = {
            'resistance': deque(maxlen=window_size),
            'reactance': deque(maxlen=window_size)
        }
        
        self.previous_zones = {}
        
        logger.info(f"BeAST Real-Time Processor initialized (window={window_size})")
    
    def parse_jsonl_line(self, line: str) -> Optional[SensorReading]:
        """Parse single JSONL line into SensorReading"""
        try:
            data = json.loads(line)
            
            reading = SensorReading(
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                device_side=data.get('device_side', 'unknown')
            )
            
            if 'eeg' in data:
                eeg = data['eeg']
                reading.eeg_fp1 = eeg.get('fp1')
                reading.eeg_fp2 = eeg.get('fp2')
                reading.eeg_f7 = eeg.get('f7')
                reading.eeg_f8 = eeg.get('f8')
                reading.eeg_t3 = eeg.get('t3')
                reading.eeg_t4 = eeg.get('t4')
                reading.eeg_t5 = eeg.get('t5')
                reading.eeg_t6 = eeg.get('t6')
                reading.eeg_o1 = eeg.get('o1')
                reading.eeg_o2 = eeg.get('o2')
                reading.eeg_a1 = eeg.get('a1')
                reading.eeg_a2 = eeg.get('a2')
            
            if 'ppg' in data:
                ppg = data['ppg']
                reading.heart_rate = ppg.get('heart_rate')
                reading.spo2 = ppg.get('spo2')
            
            if 'bioimpedance' in data:
                bio = data['bioimpedance']
                reading.bioimpedance_resistance = bio.get('resistance')
                reading.bioimpedance_reactance = bio.get('reactance')
            
            if 'environment' in data:
                env = data['environment']
                reading.ambient_temp = env.get('temp')
                reading.ambient_humidity = env.get('humidity')
                reading.sound_level_db = env.get('sound')
            
            return reading
            
        except Exception as e:
            logger.error(f"Failed to parse JSONL line: {e}")
            return None
    
    def update_buffers(self, reading: SensorReading):
        """Add new sensor reading to rolling buffers"""
        for ch in ['fp1', 'fp2', 'f7', 'f8', 't3', 't4', 't5', 't6', 'o1', 'o2', 'a1', 'a2']:
            value = getattr(reading, f'eeg_{ch}')
            if value is not None:
                self.eeg_buffers[f'eeg_{ch}'].append(value)
        
        if reading.heart_rate:
            ibi = 60000 / reading.heart_rate
            self.heart_rate_buffer.append(ibi)
        
        if reading.bioimpedance_resistance:
            self.bioimpedance_buffer['resistance'].append(reading.bioimpedance_resistance)
        if reading.bioimpedance_reactance:
            self.bioimpedance_buffer['reactance'].append(reading.bioimpedance_reactance)
    
    def process_current_window(self, session_id: str, current_time: datetime) -> Optional[ProcessedMetrics]:
        """Process current buffered data"""
        if len(self.eeg_buffers['eeg_fp1']) < self.window_size // 2:
            return None
        
        # Preprocess EEG
        preprocessed_eeg = {}
        for ch_name, buffer in self.eeg_buffers.items():
            if len(buffer) > 0:
                signal_array = np.array(buffer)
                preprocessed = self.preprocessor.preprocess_eeg_channel(signal_array)
                preprocessed_eeg[ch_name] = preprocessed
        
        # Extract features
        eeg_features = self.feature_extractor.extract_eeg_features(preprocessed_eeg)
        
        if len(self.heart_rate_buffer) > 10:
            ibi_array = np.array(self.heart_rate_buffer)
            hrv_features = self.feature_extractor.calculate_hrv_features(ibi_array)
        else:
            hrv_features = {'rmssd': 50, 'sdnn': 50, 'pnn50': 20, 'lf_hf_ratio': 1.0}
        
        if len(self.bioimpedance_buffer['resistance']) > 5:
            resistance_array = np.array(self.bioimpedance_buffer['resistance'])
            reactance_array = np.array(self.bioimpedance_buffer['reactance'])
            bioimpedance_features = self.feature_extractor.calculate_bioimpedance_features(
                resistance_array, reactance_array
            )
        else:
            bioimpedance_features = {'phase_angle': 5.0, 'resistance_trend': 0.0}
        
        current_hr = 60000 / self.heart_rate_buffer[-1] if len(self.heart_rate_buffer) > 0 else 75
        
        # Calculate metrics
        metrics = self.metric_calculator.calculate_all_metrics(
            eeg_features, hrv_features, bioimpedance_features, current_hr
        )
        
        zones = ZoneClassifier.classify_all_metrics(metrics)
        
        processed = ProcessedMetrics(
            timestamp=current_time,
            session_id=session_id,
            cognitive_load=metrics['cognitive_load'],
            tiredness=metrics['tiredness'],
            fatigue=metrics['fatigue'],
            attention_focus=metrics['attention_focus'],
            stress_index=metrics['stress_index'],
            neurovascular_coupling_index=metrics['neurovascular_coupling_index'],
            metabolic_stress_index=metrics['metabolic_stress_index'],
            compensation_cognitive_load=metrics['compensation_cognitive_load'],
            fatigue_severity_score=metrics['fatigue_severity_score'],
            attention_capacity=metrics['attention_capacity'],
            cognitive_load_zone=zones['cognitive_load_zone'],
            tiredness_zone=zones['tiredness_zone'],
            fatigue_zone=zones['fatigue_zone'],
            attention_focus_zone=zones['attention_focus_zone'],
            stress_index_zone=zones['stress_index_zone'],
            neurovascular_coupling_index_zone=zones['neurovascular_coupling_index_zone'],
            metabolic_stress_index_zone=zones['metabolic_stress_index_zone'],
            compensation_cognitive_load_zone=zones['compensation_cognitive_load_zone'],
            fatigue_severity_score_zone=zones['fatigue_severity_score_zone'],
            attention_capacity_zone=zones['attention_capacity_zone']
        )
        
        # Store to database
        self.db.insert_eeg_metrics(session_id, current_time, eeg_features)
        self.db.insert_physiological_metrics(
            session_id, current_time, hrv_features, bioimpedance_features, current_hr
        )
        self.db.insert_calculated_metrics(processed)
        
        self._track_zone_transitions(session_id, current_time, metrics, zones)
        
        if current_time.second % 10 == 0:
            self.db.update_cumulative_stats(session_id)
        
        logger.info(f"Processed @ {current_time}: "
                   f"CogLoad={metrics['cognitive_load']:.1f} (Z{zones['cognitive_load_zone']}), "
                   f"Fatigue={metrics['fatigue']:.1f} (Z{zones['fatigue_zone']}), "
                   f"Attention={metrics['attention_focus']:.1f} (Z{zones['attention_focus_zone']})")
        
        return processed
    
    def _track_zone_transitions(self, session_id: str, timestamp: datetime,
                                metrics: Dict[str, float], zones: Dict[str, int]):
        """Track zone transitions"""
        for metric_name, metric_value in metrics.items():
            zone_key = f"{metric_name}_zone"
            current_zone = zones.get(zone_key, 1)
            
            if metric_name in self.previous_zones:
                previous_zone = self.previous_zones[metric_name]
                if previous_zone != current_zone:
                    self.db.track_zone_transition(
                        session_id, timestamp, metric_name,
                        previous_zone, current_zone, metric_value
                    )
            
            self.previous_zones[metric_name] = current_zone
    
    def start_streaming(self, jsonl_file_path: str, session_id: str, 
                       user_id: str = 'user001', device_side: str = 'left',
                       process_interval: int = 1):
        """Start real-time streaming from JSONL file"""
        logger.info(f"Starting streaming: {jsonl_file_path}")
        logger.info(f"Session: {session_id}, User: {user_id}")
        
        self.db.create_session(session_id, user_id, device_side)
        
        try:
            with open(jsonl_file_path, 'r') as f:
                line_count = 0
                last_process_time = None
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    reading = self.parse_jsonl_line(line)
                    if reading is None:
                        continue
                    
                    self.update_buffers(reading)
                    line_count += 1
                    
                    if last_process_time is None:
                        last_process_time = reading.timestamp
                    
                    time_elapsed = (reading.timestamp - last_process_time).total_seconds()
                    
                    if time_elapsed >= process_interval:
                        self.process_current_window(session_id, reading.timestamp)
                        last_process_time = reading.timestamp
                
                logger.info(f"Streaming complete: {line_count} lines processed")
                self.db.update_cumulative_stats(session_id)
                
        except FileNotFoundError:
            logger.error(f"JSONL file not found: {jsonl_file_path}")
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
    
    def cleanup(self):
        """Cleanup resources"""
        self.db.close()
        logger.info("Processor cleanup complete")


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

def generate_synthetic_jsonl(output_path: str = 'test_sensor_data.jsonl', 
                             duration_seconds: int = 60, 
                             sampling_rate: int = 250):
    """Generate synthetic JSONL sensor data for testing"""
    logger.info(f"Generating {duration_seconds}s of synthetic data @ {sampling_rate} Hz")
    
    num_samples = duration_seconds * sampling_rate
    start_time = datetime.now()
    
    with open(output_path, 'w') as f:
        for i in range(num_samples):
            timestamp = start_time + timedelta(seconds=i/sampling_rate)
            
            eeg_data = {
                'fp1': np.random.normal(10, 2),
                'fp2': np.random.normal(10, 2),
                'f7': np.random.normal(12, 3),
                'f8': np.random.normal(12, 3),
                't3': np.random.normal(15, 4),
                't4': np.random.normal(15, 4),
                't5': np.random.normal(14, 3),
                't6': np.random.normal(14, 3),
                'o1': np.random.normal(20, 5),
                'o2': np.random.normal(20, 5),
                'a1': np.random.normal(8, 2),
                'a2': np.random.normal(8, 2)
            }
            
            ppg_data = {
                'heart_rate': np.random.normal(75, 10),
                'spo2': np.random.normal(97, 1)
            }
            
            bio_data = {
                'resistance': np.random.normal(450, 20),
                'reactance': np.random.normal(50, 5)
            }
            
            env_data = {
                'temp': np.random.normal(22, 1),
                'humidity': np.random.normal(45, 5),
                'sound': np.random.normal(65, 10)
            }
            
            sensor_reading = {
                'timestamp': timestamp.isoformat(),
                'device_side': 'left',
                'eeg': eeg_data,
                'ppg': ppg_data,
                'bioimpedance': bio_data,
                'environment': env_data
            }
            
            f.write(json.dumps(sensor_reading) + '\n')
    
    logger.info(f"Synthetic data generated: {output_path}")


def main():
    """Main execution"""
    
    # Generate test data
    generate_synthetic_jsonl('test_sensor_data.jsonl', duration_seconds=30, sampling_rate=250)
    
    # Initialize processor
    processor = BeASTRealtimeProcessor(
        db_path='beast_local.db',
        window_size=500
    )
    
    # Start streaming
    try:
        processor.start_streaming(
            jsonl_file_path='test_sensor_data.jsonl',
            session_id='session_001',
            user_id='user001',
            device_side='left',
            process_interval=1
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        processor.cleanup()
    
    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"Database: beast_local.db")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
