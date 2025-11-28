"""
BeAST Sensor Fusion Module
Combines left and right ear sensor data, creates feature vectors, and performs cross-sensor analysis
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime
import scipy.stats as stats
from scipy.signal import welch, find_peaks


class BeASTSensorFusion:
    def __init__(self):
        self.feature_names = []
        self.sensor_types = ['ppg', 'eeg', 'imu', 'temp', 'bioz']
        
    def load_sensor_data(self, file_path: str) -> Dict:
        """Load BeAST sensor data from JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def extract_sensor_by_type(self, data, sensor_type: str) -> Dict:
        """Extract specific sensor data from both ears"""
        left_data = []
        right_data = []
        
        # Handle both dict with 'sensor_data' key and direct list format
        if isinstance(data, dict) and 'sensor_data' in data:
            samples = data['sensor_data']
        elif isinstance(data, list):
            samples = data
        else:
            return {'left': left_data, 'right': right_data}
        
        for sample in samples:
            if sample['side'] == 'left':
                devices = sample.get('devices', sample.get('device', []))
                for device in devices:
                    if device['name'] == sensor_type:
                        left_data.append({
                            'time': sample['time'],
                            'status': device['status'],
                            'freq': device['freq'],
                            'data': device['data']
                        })
            elif sample['side'] == 'right':
                devices = sample.get('devices', sample.get('device', []))
                for device in devices:
                    if device['name'] == sensor_type:
                        right_data.append({
                            'time': sample['time'],
                            'status': device['status'],
                            'freq': device['freq'],
                            'data': device['data']
                        })
        
        return {'left': left_data, 'right': right_data}
    
    def combine_bilateral_data(self, left_data: List, right_data: List) -> List:
        """Combine left and right ear sensor data"""
        combined = []
        
        # Align data by timestamp
        left_dict = {tuple(item['time']): item for item in left_data}
        right_dict = {tuple(item['time']): item for item in right_data}
        
        all_times = set(left_dict.keys()) | set(right_dict.keys())
        
        for time_key in sorted(all_times):
            combined_sample = {
                'time': list(time_key),
                'left': left_dict.get(time_key, None),
                'right': right_dict.get(time_key, None)
            }
            combined.append(combined_sample)
        
        return combined
    
    def extract_ppg_features(self, ppg_data: List) -> Dict:
        """Extract PPG features from combined bilateral data"""
        features = {}
        
        left_hr = []
        right_hr = []
        left_spo2 = []
        right_spo2 = []
        
        for sample in ppg_data:
            if sample['left'] and sample['left']['data']:
                left_hr.append(sample['left']['data'][0])
                left_spo2.append(sample['left']['data'][1])
            if sample['right'] and sample['right']['data']:
                right_hr.append(sample['right']['data'][0])
                right_spo2.append(sample['right']['data'][1])
        
        # Basic statistics
        if left_hr:
            features['left_hr_mean'] = np.mean(left_hr)
            features['left_hr_std'] = np.std(left_hr)
            features['left_hr_min'] = np.min(left_hr)
            features['left_hr_max'] = np.max(left_hr)
            features['left_spo2_mean'] = np.mean(left_spo2)
            features['left_spo2_std'] = np.std(left_spo2)
        
        if right_hr:
            features['right_hr_mean'] = np.mean(right_hr)
            features['right_hr_std'] = np.std(right_hr)
            features['right_hr_min'] = np.min(right_hr)
            features['right_hr_max'] = np.max(right_hr)
            features['right_spo2_mean'] = np.mean(right_spo2)
            features['right_spo2_std'] = np.std(right_spo2)
        
        # Cross-ear features
        if left_hr and right_hr:
            features['hr_bilateral_diff_mean'] = np.mean(np.array(left_hr) - np.array(right_hr))
            features['hr_bilateral_correlation'] = np.corrcoef(left_hr, right_hr)[0, 1] if len(left_hr) > 1 else 0
            features['spo2_bilateral_diff_mean'] = np.mean(np.array(left_spo2) - np.array(right_spo2))
        
        return features
    
    def extract_eeg_features(self, eeg_data: List) -> Dict:
        """Extract EEG features from combined bilateral data"""
        features = {}
        
        for sample in eeg_data:
            for side in ['left', 'right']:
                if sample[side] and sample[side]['data']:
                    # Handle both 'channels' and 'channel' keys
                    eeg_channels = sample[side]['data'].get('channels', sample[side]['data'].get('channel', []))
                    
                    for channel in eeg_channels:
                        ch_num = channel['number']
                        ch_data = np.array(channel['set'])
                        
                        # Time domain features
                        features[f'{side}_eeg_ch{ch_num}_mean'] = np.mean(ch_data)
                        features[f'{side}_eeg_ch{ch_num}_std'] = np.std(ch_data)
                        features[f'{side}_eeg_ch{ch_num}_rms'] = np.sqrt(np.mean(ch_data**2))
                        features[f'{side}_eeg_ch{ch_num}_skewness'] = stats.skew(ch_data)
                        features[f'{side}_eeg_ch{ch_num}_kurtosis'] = stats.kurtosis(ch_data)
                        
                        # Frequency domain features (if enough samples)
                        if len(ch_data) > 10:
                            freqs, psd = welch(ch_data, fs=250, nperseg=min(len(ch_data), 256))
                            features[f'{side}_eeg_ch{ch_num}_total_power'] = np.sum(psd)
                            features[f'{side}_eeg_ch{ch_num}_peak_freq'] = freqs[np.argmax(psd)]
        
        # Cross-hemisphere features
        if 'left_eeg_ch1_mean' in features and 'right_eeg_ch1_mean' in features:
            for ch in range(1, 9):
                if f'left_eeg_ch{ch}_mean' in features and f'right_eeg_ch{ch}_mean' in features:
                    features[f'eeg_ch{ch}_bilateral_diff'] = features[f'left_eeg_ch{ch}_mean'] - features[f'right_eeg_ch{ch}_mean']
        
        return features
    
    def extract_imu_features(self, imu_data: List) -> Dict:
        """Extract IMU features from combined bilateral data"""
        features = {}
        
        for sample in imu_data:
            for side in ['left', 'right']:
                if sample[side] and sample[side]['data']:
                    imu_values = sample[side]['data']
                    
                    # Handle dict format (new) vs list format (old)
                    if isinstance(imu_values, dict):
                        # New format with structured data
                        if 'accel' in imu_values:
                            accel = np.array(imu_values['accel'])
                            features[f'{side}_accel_magnitude'] = np.linalg.norm(accel)
                            features[f'{side}_accel_x'] = accel[0]
                            features[f'{side}_accel_y'] = accel[1]
                            features[f'{side}_accel_z'] = accel[2]
                        
                        if 'gyro' in imu_values:
                            gyro = np.array(imu_values['gyro'])
                            features[f'{side}_gyro_magnitude'] = np.linalg.norm(gyro)
                            features[f'{side}_gyro_x'] = gyro[0]
                            features[f'{side}_gyro_y'] = gyro[1]
                            features[f'{side}_gyro_z'] = gyro[2]
                        
                        if 'steps' in imu_values:
                            features[f'{side}_step_count'] = imu_values['steps']
                        
                        if 'activity' in imu_values:
                            features[f'{side}_activity'] = imu_values['activity']
                    
                    elif isinstance(imu_values, list):
                        # Old format with flat list
                        # Accelerometer features (first 3 values)
                        if len(imu_values) >= 3:
                            accel = np.array(imu_values[:3])
                            features[f'{side}_accel_magnitude'] = np.linalg.norm(accel)
                            features[f'{side}_accel_x'] = accel[0]
                            features[f'{side}_accel_y'] = accel[1]
                            features[f'{side}_accel_z'] = accel[2]
                        
                        # High-G accelerometer (next 3 values)
                        if len(imu_values) >= 6:
                            high_g_accel = np.array(imu_values[3:6])
                            features[f'{side}_high_g_magnitude'] = np.linalg.norm(high_g_accel)
                        
                        # Gyroscope (next 3 values)
                        if len(imu_values) >= 9:
                            gyro = np.array(imu_values[6:9])
                            features[f'{side}_gyro_magnitude'] = np.linalg.norm(gyro)
                            features[f'{side}_gyro_x'] = gyro[0]
                            features[f'{side}_gyro_y'] = gyro[1]
                            features[f'{side}_gyro_z'] = gyro[2]
                        
                        # Activity flags
                        if len(imu_values) >= 14:
                            features[f'{side}_step_count'] = imu_values[9]
                            features[f'{side}_tilt_detection'] = imu_values[10]
                            features[f'{side}_free_fall'] = imu_values[11]
                            features[f'{side}_wake_up'] = imu_values[12]
                            features[f'{side}_activity'] = imu_values[13]
        
        # Cross-ear motion features
        if 'left_accel_magnitude' in features and 'right_accel_magnitude' in features:
            features['accel_bilateral_diff'] = features['left_accel_magnitude'] - features['right_accel_magnitude']
            if 'left_gyro_magnitude' in features and 'right_gyro_magnitude' in features:
                features['gyro_bilateral_diff'] = features['left_gyro_magnitude'] - features['right_gyro_magnitude']
        
        return features
    
    def extract_temp_features(self, temp_data: List) -> Dict:
        """Extract temperature features from combined bilateral data"""
        features = {}
        
        left_body_temp = []
        right_body_temp = []
        left_ambient_temp = []
        right_ambient_temp = []
        
        for sample in temp_data:
            if sample['left'] and sample['left']['data']:
                left_body_temp.append(sample['left']['data'][0])
                left_ambient_temp.append(sample['left']['data'][1])
            if sample['right'] and sample['right']['data']:
                right_body_temp.append(sample['right']['data'][0])
                right_ambient_temp.append(sample['right']['data'][1])
        
        # Body temperature features
        if left_body_temp:
            features['left_body_temp_mean'] = np.mean(left_body_temp)
            features['left_body_temp_std'] = np.std(left_body_temp)
            features['left_ambient_temp_mean'] = np.mean(left_ambient_temp)
        
        if right_body_temp:
            features['right_body_temp_mean'] = np.mean(right_body_temp)
            features['right_body_temp_std'] = np.std(right_body_temp)
            features['right_ambient_temp_mean'] = np.mean(right_ambient_temp)
        
        # Temperature gradients
        if left_body_temp and right_body_temp:
            features['body_temp_bilateral_diff'] = np.mean(left_body_temp) - np.mean(right_body_temp)
            features['ambient_temp_bilateral_diff'] = np.mean(left_ambient_temp) - np.mean(right_ambient_temp)
        
        return features
    
    def extract_bioz_features(self, bioz_data: List) -> Dict:
        """Extract bioimpedance features from combined bilateral data"""
        features = {}
        
        left_bioz = []
        right_bioz = []
        
        for sample in bioz_data:
            if sample['left'] and sample['left']['data']:
                left_bioz.extend(sample['left']['data'])
            if sample['right'] and sample['right']['data']:
                right_bioz.extend(sample['right']['data'])
        
        # Bioimpedance features
        if left_bioz:
            features['left_bioz_mean'] = np.mean(left_bioz)
            features['left_bioz_std'] = np.std(left_bioz)
            features['left_bioz_median'] = np.median(left_bioz)
        
        if right_bioz:
            features['right_bioz_mean'] = np.mean(right_bioz)
            features['right_bioz_std'] = np.std(right_bioz)
            features['right_bioz_median'] = np.median(right_bioz)
        
        # Cross-ear bioimpedance
        if left_bioz and right_bioz:
            features['bioz_bilateral_diff'] = np.mean(left_bioz) - np.mean(right_bioz)
            features['bioz_bilateral_ratio'] = np.mean(left_bioz) / np.mean(right_bioz) if np.mean(right_bioz) != 0 else 0
        
        return features
    
    def create_cross_sensor_features(self, all_features: Dict) -> Dict:
        """Create interaction features between different sensors"""
        cross_features = {}
        
        # HR vs Temperature correlation
        if 'left_hr_mean' in all_features and 'left_body_temp_mean' in all_features:
            cross_features['hr_temp_interaction_left'] = all_features['left_hr_mean'] * all_features['left_body_temp_mean']
        
        # Motion vs EEG (motion artifacts)
        if 'left_accel_magnitude' in all_features and 'left_eeg_ch1_std' in all_features:
            cross_features['motion_eeg_artifact_left'] = all_features['left_accel_magnitude'] * all_features['left_eeg_ch1_std']
        
        # HR variability vs EEG alpha power (stress indicator)
        if 'left_hr_std' in all_features and 'left_eeg_ch1_total_power' in all_features:
            cross_features['hrv_eeg_stress_left'] = all_features['left_hr_std'] / (all_features['left_eeg_ch1_total_power'] + 1e-6)
        
        # Temperature gradient vs bioimpedance (hydration indicator)
        if 'body_temp_bilateral_diff' in all_features and 'bioz_bilateral_diff' in all_features:
            cross_features['temp_bioz_hydration'] = all_features['body_temp_bilateral_diff'] * all_features['bioz_bilateral_diff']
        
        # Bilateral asymmetry index
        bilateral_diffs = [v for k, v in all_features.items() if 'bilateral_diff' in k]
        if bilateral_diffs:
            cross_features['bilateral_asymmetry_index'] = np.mean(np.abs(bilateral_diffs))
        
        return cross_features
    
    def create_feature_vector(self, data: Dict, window_size: int = 60) -> np.ndarray:
        """Create ML-ready feature vector from sensor data"""
        all_features = {}
        
        # Extract features from each sensor type
        for sensor_type in self.sensor_types:
            sensor_data = self.extract_sensor_by_type(data, sensor_type)
            combined_data = self.combine_bilateral_data(sensor_data['left'], sensor_data['right'])
            
            if sensor_type == 'ppg':
                features = self.extract_ppg_features(combined_data)
            elif sensor_type == 'eeg':
                features = self.extract_eeg_features(combined_data)
            elif sensor_type == 'imu':
                features = self.extract_imu_features(combined_data)
            elif sensor_type == 'temp':
                features = self.extract_temp_features(combined_data)
            elif sensor_type == 'bioz':
                features = self.extract_bioz_features(combined_data)
            
            all_features.update(features)
        
        # Add cross-sensor features
        cross_features = self.create_cross_sensor_features(all_features)
        all_features.update(cross_features)
        
        # Store feature names for reference
        self.feature_names = list(all_features.keys())
        
        # Convert to numpy array
        feature_vector = np.array([all_features.get(name, 0) for name in self.feature_names])
        
        return feature_vector
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names
    
    def create_feature_dataframe(self, data: Dict) -> pd.DataFrame:
        """Create pandas DataFrame with features for easier analysis"""
        feature_vector = self.create_feature_vector(data)
        
        df = pd.DataFrame([feature_vector], columns=self.feature_names)
        return df
    
    def batch_process_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Process multiple sensor data files and create feature matrix"""
        all_features = []
        
        for file_path in file_paths:
            data = self.load_sensor_data(file_path)
            feature_vector = self.create_feature_vector(data)
            all_features.append(feature_vector)
        
        # Create DataFrame
        feature_matrix = np.array(all_features)
        df = pd.DataFrame(feature_matrix, columns=self.feature_names)
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize sensor fusion
    fusion = BeASTSensorFusion()
    
    # Load and process sensor data
    data_file = "beast_sensor_data_2025-10-12_14-30-00.json"
    
    try:
        # Load data
        sensor_data = fusion.load_sensor_data(data_file)
        
        # Create feature vector
        features = fusion.create_feature_vector(sensor_data)
        
        # Get feature names
        feature_names = fusion.get_feature_names()
        
        print(f"Generated {len(features)} features:")
        for i, name in enumerate(feature_names):
            print(f"{name}: {features[i]:.4f}")
        
        # Create DataFrame for easier analysis
        df = fusion.create_feature_dataframe(sensor_data)
        print(f"\nFeature DataFrame shape: {df.shape}")
        print("\nFirst few features:")
        print(df.head())
        
    except FileNotFoundError:
        print(f"File {data_file} not found. Please ensure the sensor data file exists.")