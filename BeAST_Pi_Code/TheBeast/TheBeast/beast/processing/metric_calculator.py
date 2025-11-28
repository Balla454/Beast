#!/usr/bin/env python3
"""
BeAST Metric Calculator
=======================
Calculates 10 physiological/cognitive metrics from extracted features.
Maps to the metrics defined in beast_realtime_processor.py.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger('BeAST.Metrics')


class Zone(Enum):
    """Risk zone classification"""
    GREEN = 1   # Normal/Optimal
    YELLOW = 2  # Caution/Monitor
    ORANGE = 3  # Warning/Elevated
    RED = 4     # Critical/Immediate attention


@dataclass
class MetricResult:
    """Result of a metric calculation"""
    name: str
    value: float
    zone: Zone
    confidence: float
    description: str


class MetricCalculator:
    """
    Calculate cognitive and physiological metrics.
    
    Metrics:
    1. cognitive_load - Mental workload level
    2. tiredness - Physical/mental tiredness
    3. fatigue - Accumulated fatigue
    4. attention_focus - Current attention level
    5. stress_index - Physiological stress
    6. neurovascular_coupling_index - Brain blood flow efficiency
    7. metabolic_stress_index - Metabolic strain
    8. compensation_cognitive_load - Compensatory effort
    9. fatigue_severity_score - Fatigue severity
    10. attention_capacity - Available attention resources
    """
    
    # Zone thresholds for each metric [green_max, yellow_max, orange_max]
    # Values above orange_max are red
    ZONE_THRESHOLDS = {
        'cognitive_load': [40, 60, 80],
        'tiredness': [30, 50, 70],
        'fatigue': [25, 50, 75],
        'attention_focus': [80, 60, 40],  # Inverted: higher is better
        'stress_index': [30, 50, 70],
        'neurovascular_coupling_index': [80, 60, 40],  # Inverted
        'metabolic_stress_index': [30, 50, 70],
        'compensation_cognitive_load': [30, 50, 70],
        'fatigue_severity_score': [25, 50, 75],
        'attention_capacity': [80, 60, 40]  # Inverted
    }
    
    def __init__(self):
        """Initialize metric calculator"""
        # History for trend analysis
        self.metric_history: Dict[str, List[float]] = {
            name: [] for name in self.ZONE_THRESHOLDS
        }
        self.max_history = 100
        
    def calculate_all_metrics(self, features: Dict[str, float]) -> Dict[str, MetricResult]:
        """
        Calculate all metrics from features.
        
        Args:
            features: Dict of extracted features
            
        Returns:
            Dict of metric name -> MetricResult
        """
        results = {}
        
        # Calculate each metric
        results['cognitive_load'] = self._calc_cognitive_load(features)
        results['tiredness'] = self._calc_tiredness(features)
        results['fatigue'] = self._calc_fatigue(features)
        results['attention_focus'] = self._calc_attention_focus(features)
        results['stress_index'] = self._calc_stress_index(features)
        results['neurovascular_coupling_index'] = self._calc_nvc_index(features)
        results['metabolic_stress_index'] = self._calc_metabolic_stress(features)
        results['compensation_cognitive_load'] = self._calc_compensation_load(features)
        results['fatigue_severity_score'] = self._calc_fatigue_severity(features)
        results['attention_capacity'] = self._calc_attention_capacity(features)
        
        # Update history
        for name, result in results.items():
            self.metric_history[name].append(result.value)
            if len(self.metric_history[name]) > self.max_history:
                self.metric_history[name].pop(0)
                
        return results
        
    def _classify_zone(self, value: float, metric_name: str) -> Zone:
        """Classify value into zone"""
        thresholds = self.ZONE_THRESHOLDS.get(metric_name, [40, 60, 80])
        
        # Check if metric is inverted (higher is better)
        inverted = metric_name in ['attention_focus', 'neurovascular_coupling_index', 'attention_capacity']
        
        if inverted:
            if value >= thresholds[0]:
                return Zone.GREEN
            elif value >= thresholds[1]:
                return Zone.YELLOW
            elif value >= thresholds[2]:
                return Zone.ORANGE
            else:
                return Zone.RED
        else:
            if value <= thresholds[0]:
                return Zone.GREEN
            elif value <= thresholds[1]:
                return Zone.YELLOW
            elif value <= thresholds[2]:
                return Zone.ORANGE
            else:
                return Zone.RED
                
    def _calc_cognitive_load(self, features: Dict[str, float]) -> MetricResult:
        """
        Calculate cognitive load from EEG theta/alpha ratio and HRV.
        
        High theta/alpha + low HRV = high cognitive load
        """
        # Get EEG features
        theta = features.get('left_theta_mean', 0) + features.get('right_theta_mean', 0)
        alpha = features.get('left_alpha_mean', 1) + features.get('right_alpha_mean', 1)
        
        theta_alpha_ratio = theta / (alpha + 1e-6)
        
        # Get HRV (lower HRV = higher load)
        hrv = (features.get('left_hrv', 50) + features.get('right_hrv', 50)) / 2
        hrv_factor = 1 - min(hrv / 100, 1)  # Normalize and invert
        
        # Combine
        value = (theta_alpha_ratio * 30 + hrv_factor * 70)
        value = np.clip(value, 0, 100)
        
        return MetricResult(
            name='cognitive_load',
            value=float(value),
            zone=self._classify_zone(value, 'cognitive_load'),
            confidence=0.8 if alpha > 0.1 else 0.4,
            description='Mental workload level based on EEG patterns and heart rate variability'
        )
        
    def _calc_tiredness(self, features: Dict[str, float]) -> MetricResult:
        """
        Calculate tiredness from alpha power and blink rate proxy.
        
        High alpha + slow HR variability = tired
        """
        alpha = features.get('left_alpha_mean', 0) + features.get('right_alpha_mean', 0)
        hr = (features.get('left_hr', 70) + features.get('right_hr', 70)) / 2
        
        # Alpha increase with fatigue
        alpha_factor = min(alpha / 20, 1)
        
        # Lower HR often with tiredness
        hr_factor = 1 - min((hr - 50) / 50, 1) if hr > 50 else 1
        
        value = (alpha_factor * 60 + hr_factor * 40)
        value = np.clip(value * 100, 0, 100)
        
        return MetricResult(
            name='tiredness',
            value=float(value),
            zone=self._classify_zone(value, 'tiredness'),
            confidence=0.7,
            description='Current level of physical and mental tiredness'
        )
        
    def _calc_fatigue(self, features: Dict[str, float]) -> MetricResult:
        """
        Calculate accumulated fatigue from trends and current state.
        """
        # Current tiredness
        tiredness = self._calc_tiredness(features).value
        
        # Trend from history
        history = self.metric_history['tiredness']
        if len(history) >= 5:
            trend = (history[-1] - history[-5]) / 5
            fatigue_accumulation = max(trend * 10, 0)
        else:
            fatigue_accumulation = 0
            
        value = tiredness * 0.7 + fatigue_accumulation * 30
        value = np.clip(value, 0, 100)
        
        return MetricResult(
            name='fatigue',
            value=float(value),
            zone=self._classify_zone(value, 'fatigue'),
            confidence=0.6,
            description='Accumulated fatigue over time'
        )
        
    def _calc_attention_focus(self, features: Dict[str, float]) -> MetricResult:
        """
        Calculate attention focus from beta/theta ratio.
        
        High beta, low theta = focused attention
        """
        beta = features.get('left_beta_mean', 0) + features.get('right_beta_mean', 0)
        theta = features.get('left_theta_mean', 1) + features.get('right_theta_mean', 1)
        
        beta_theta_ratio = beta / (theta + 1e-6)
        
        # Motion affects attention
        motion = features.get('left_motion_variance', 0) + features.get('right_motion_variance', 0)
        motion_penalty = min(motion * 10, 30)
        
        value = min(beta_theta_ratio * 20, 100) - motion_penalty
        value = np.clip(value, 0, 100)
        
        return MetricResult(
            name='attention_focus',
            value=float(value),
            zone=self._classify_zone(value, 'attention_focus'),
            confidence=0.75,
            description='Current level of focused attention'
        )
        
    def _calc_stress_index(self, features: Dict[str, float]) -> MetricResult:
        """
        Calculate stress index from HRV, HR, and EEG beta.
        
        Low HRV + high HR + high beta = stressed
        """
        hr = (features.get('left_hr', 70) + features.get('right_hr', 70)) / 2
        hrv = (features.get('left_hrv', 50) + features.get('right_hrv', 50)) / 2
        beta = features.get('left_beta_mean', 0) + features.get('right_beta_mean', 0)
        
        # HR component (higher HR = more stress)
        hr_stress = np.clip((hr - 60) / 60, 0, 1)
        
        # HRV component (lower HRV = more stress)
        hrv_stress = 1 - np.clip(hrv / 100, 0, 1)
        
        # Beta component
        beta_stress = np.clip(beta / 10, 0, 1)
        
        value = (hr_stress * 40 + hrv_stress * 40 + beta_stress * 20) * 100
        value = np.clip(value, 0, 100)
        
        return MetricResult(
            name='stress_index',
            value=float(value),
            zone=self._classify_zone(value, 'stress_index'),
            confidence=0.8,
            description='Physiological stress level'
        )
        
    def _calc_nvc_index(self, features: Dict[str, float]) -> MetricResult:
        """
        Calculate neurovascular coupling index.
        
        Good coupling = correlated EEG activity and blood flow changes
        """
        # Use bilateral coherence as proxy
        alpha_ratio = features.get('alpha_bilateral_ratio', 1)
        hr_diff = features.get('hr_bilateral_diff', 0)
        
        # Symmetry suggests good coupling
        alpha_symmetry = 1 - abs(alpha_ratio - 1)
        hr_symmetry = 1 - min(hr_diff / 20, 1)
        
        # SpO2 stability
        spo2_left = features.get('left_spo2', 98)
        spo2_right = features.get('right_spo2', 98)
        spo2_factor = min((spo2_left + spo2_right) / 2 - 90, 10) / 10
        
        value = (alpha_symmetry * 30 + hr_symmetry * 30 + spo2_factor * 40) * 100
        value = np.clip(value, 0, 100)
        
        return MetricResult(
            name='neurovascular_coupling_index',
            value=float(value),
            zone=self._classify_zone(value, 'neurovascular_coupling_index'),
            confidence=0.6,
            description='Brain blood flow efficiency'
        )
        
    def _calc_metabolic_stress(self, features: Dict[str, float]) -> MetricResult:
        """
        Calculate metabolic stress from temperature and bioimpedance.
        """
        temp_left = features.get('left_temp_mean', 36.5)
        temp_right = features.get('right_temp_mean', 36.5)
        temp_avg = (temp_left + temp_right) / 2
        
        # Temperature deviation from normal
        temp_stress = abs(temp_avg - 36.5) * 20
        
        # Bioimpedance variation
        bioz_std = features.get('left_bioz_std', 0) + features.get('right_bioz_std', 0)
        bioz_stress = min(bioz_std / 5, 50)
        
        value = temp_stress + bioz_stress
        value = np.clip(value, 0, 100)
        
        return MetricResult(
            name='metabolic_stress_index',
            value=float(value),
            zone=self._classify_zone(value, 'metabolic_stress_index'),
            confidence=0.5,
            description='Metabolic strain indicator'
        )
        
    def _calc_compensation_load(self, features: Dict[str, float]) -> MetricResult:
        """
        Calculate compensatory cognitive load.
        
        Effort to maintain performance despite fatigue/stress.
        """
        cognitive = self._calc_cognitive_load(features).value
        fatigue = self._calc_fatigue(features).value
        attention = self._calc_attention_focus(features).value
        
        # High cognitive load + high fatigue + maintained attention = compensation
        if attention > 50:
            compensation = (cognitive + fatigue) / 2 * (attention / 100)
        else:
            compensation = cognitive * 0.5
            
        value = np.clip(compensation, 0, 100)
        
        return MetricResult(
            name='compensation_cognitive_load',
            value=float(value),
            zone=self._classify_zone(value, 'compensation_cognitive_load'),
            confidence=0.55,
            description='Mental effort to maintain performance'
        )
        
    def _calc_fatigue_severity(self, features: Dict[str, float]) -> MetricResult:
        """
        Calculate fatigue severity score.
        
        Combines multiple fatigue indicators.
        """
        tiredness = self._calc_tiredness(features).value
        fatigue = self._calc_fatigue(features).value
        
        # EEG slow wave activity
        theta = features.get('left_theta_mean', 0) + features.get('right_theta_mean', 0)
        delta = sum(features.get(f'left_eeg_ch{i}_delta', 0) for i in range(8)) / 8
        slow_wave = (theta + delta) / 2
        
        value = tiredness * 0.3 + fatigue * 0.4 + slow_wave * 3
        value = np.clip(value, 0, 100)
        
        return MetricResult(
            name='fatigue_severity_score',
            value=float(value),
            zone=self._classify_zone(value, 'fatigue_severity_score'),
            confidence=0.65,
            description='Overall fatigue severity'
        )
        
    def _calc_attention_capacity(self, features: Dict[str, float]) -> MetricResult:
        """
        Calculate available attention capacity.
        
        Resources available for attention tasks.
        """
        attention = self._calc_attention_focus(features).value
        fatigue = self._calc_fatigue(features).value
        cognitive = self._calc_cognitive_load(features).value
        
        # Capacity decreases with fatigue and existing load
        capacity = 100 - (fatigue * 0.4 + cognitive * 0.3)
        capacity = capacity * (attention / 100 + 0.5)  # Boost if currently focused
        
        value = np.clip(capacity, 0, 100)
        
        return MetricResult(
            name='attention_capacity',
            value=float(value),
            zone=self._classify_zone(value, 'attention_capacity'),
            confidence=0.6,
            description='Available attention resources'
        )
        
    def get_summary(self, metrics: Dict[str, MetricResult]) -> str:
        """Get human-readable summary of metrics"""
        lines = ["Current Status:"]
        
        for name, result in metrics.items():
            zone_emoji = {
                Zone.GREEN: "🟢",
                Zone.YELLOW: "🟡", 
                Zone.ORANGE: "🟠",
                Zone.RED: "🔴"
            }.get(result.zone, "⚪")
            
            lines.append(f"  {zone_emoji} {name.replace('_', ' ').title()}: {result.value:.0f}/100")
            
        return "\n".join(lines)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing MetricCalculator...")
    
    calculator = MetricCalculator()
    
    # Simulated features
    features = {
        'left_hr': 75,
        'right_hr': 74,
        'left_hrv': 45,
        'right_hrv': 47,
        'left_spo2': 98,
        'right_spo2': 97,
        'left_theta_mean': 8,
        'right_theta_mean': 7,
        'left_alpha_mean': 12,
        'right_alpha_mean': 11,
        'left_beta_mean': 6,
        'right_beta_mean': 5,
        'alpha_bilateral_ratio': 1.09,
        'hr_bilateral_diff': 1,
        'left_motion_variance': 0.05,
        'right_motion_variance': 0.04,
        'left_temp_mean': 36.5,
        'right_temp_mean': 36.6,
        'left_bioz_std': 2,
        'right_bioz_std': 2
    }
    
    # Add delta features
    for i in range(8):
        features[f'left_eeg_ch{i}_delta'] = 10
        features[f'right_eeg_ch{i}_delta'] = 9
        
    metrics = calculator.calculate_all_metrics(features)
    
    print("\n" + calculator.get_summary(metrics))
    print("\nTest complete!")
