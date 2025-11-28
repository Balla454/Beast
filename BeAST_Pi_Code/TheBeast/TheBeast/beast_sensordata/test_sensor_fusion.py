"""
Test script for BeAST Sensor Fusion Pipeline
Demonstrates the complete sensor fusion workflow
"""

from sensor_fusion import BeASTSensorFusion
import numpy as np
import json

def create_sample_data():
    """Create sample sensor data for testing if no file exists"""
    sample_data = {
        "metadata": {
            "collection_start": [2025, 10, 12, 14, 30, 0],
            "collection_end": [2025, 10, 12, 14, 31, 0],
            "duration_minutes": 1,
            "description": "Sample sensor data for fusion testing"
        },
        "sensor_data": []
    }
    
    # Generate 10 seconds of data
    for i in range(10):
        # Left ear data
        left_sample = {
            "time": [2025, 10, 12, 14, 30, i],
            "side": "left",
            "device": [
                {
                    "name": "ppg",
                    "status": "active",
                    "freq": 100.0,
                    "data": [72 + np.random.normal(0, 2), 98 + np.random.normal(0, 1)]
                },
                {
                    "name": "temp",
                    "status": "active", 
                    "freq": 0.2,
                    "data": [98.6 + np.random.normal(0, 0.2), 72.0 + np.random.normal(0, 1)]
                },
                {
                    "name": "imu",
                    "status": "active",
                    "freq": 500.0,
                    "data": [
                        np.random.normal(0, 0.1),  # accel_x
                        np.random.normal(0, 0.1),  # accel_y
                        9.81 + np.random.normal(0, 0.1),  # accel_z
                        np.random.normal(0, 1),    # high_g_x
                        np.random.normal(0, 1),    # high_g_y
                        np.random.normal(0, 1),    # high_g_z
                        np.random.normal(0, 10),   # gyro_x
                        np.random.normal(0, 10),   # gyro_y
                        np.random.normal(0, 10),   # gyro_z
                        1000 + i,  # step_count
                        0,  # tilt
                        0,  # free_fall
                        1,  # wake_up
                        1   # activity
                    ]
                },
                {
                    "name": "bioz",
                    "status": "active",
                    "freq": 800.0,
                    "data": [130.0 + np.random.normal(0, 5)]
                },
                {
                    "name": "eeg",
                    "status": "active",
                    "freq": 250.0,
                    "data": {
                        "channel": [
                            {
                                "number": j,
                                "set": [35 + np.random.normal(0, 5) for _ in range(50)]
                            } for j in range(1, 9)
                        ]
                    }
                }
            ]
        }
        
        # Right ear data
        right_sample = {
            "time": [2025, 10, 12, 14, 30, i],
            "side": "right",
            "device": [
                {
                    "name": "ppg",
                    "status": "active",
                    "freq": 100.0,
                    "data": [73 + np.random.normal(0, 2), 97 + np.random.normal(0, 1)]
                },
                {
                    "name": "temp",
                    "status": "active",
                    "freq": 0.2,
                    "data": [98.5 + np.random.normal(0, 0.2), 72.2 + np.random.normal(0, 1)]
                },
                {
                    "name": "imu",
                    "status": "active",
                    "freq": 500.0,
                    "data": [
                        np.random.normal(0, 0.1),
                        np.random.normal(0, 0.1),
                        9.82 + np.random.normal(0, 0.1),
                        np.random.normal(0, 1),
                        np.random.normal(0, 1),
                        np.random.normal(0, 1),
                        np.random.normal(0, 10),
                        np.random.normal(0, 10),
                        np.random.normal(0, 10),
                        1001 + i,
                        0, 0, 1, 1
                    ]
                },
                {
                    "name": "bioz",
                    "status": "active",
                    "freq": 800.0,
                    "data": [128.0 + np.random.normal(0, 5)]
                },
                {
                    "name": "eeg",
                    "status": "active",
                    "freq": 250.0,
                    "data": {
                        "channel": [
                            {
                                "number": j,
                                "set": [36 + np.random.normal(0, 5) for _ in range(50)]
                            } for j in range(1, 9)
                        ]
                    }
                }
            ]
        }
        
        sample_data["sensor_data"].extend([left_sample, right_sample])
    
    return sample_data

def main():
    print("=== BeAST Sensor Fusion Pipeline Test ===\n")
    
    # Initialize sensor fusion
    fusion = BeASTSensorFusion()
    
    # Try to load existing data file, create sample if not found
    data_files = [
        "beast_sensor_data_5min_2025-10-12_14-30-00.json",
        "beast_sensor_data_2025-10-12_14-30-00.json"
    ]
    
    sensor_data = None
    for file_path in data_files:
        try:
            sensor_data = fusion.load_sensor_data(file_path)
            print(f"✓ Loaded data from: {file_path}")
            break
        except FileNotFoundError:
            continue
    
    if sensor_data is None:
        print("No existing data files found. Creating sample data...")
        sensor_data = create_sample_data()
        
        # Save sample data
        with open("sample_sensor_data.json", "w") as f:
            json.dump(sensor_data, f, indent=2)
        print("✓ Created and saved sample data to: sample_sensor_data.json")
    
    # Handle both dict and list formats
    if isinstance(sensor_data, dict):
        sample_count = len(sensor_data.get('sensor_data', []))
    else:
        sample_count = len(sensor_data)
    print(f"\nData contains {sample_count} sensor samples")
    
    # Test individual sensor extraction
    print("\n=== Testing Individual Sensor Extraction ===")
    for sensor_type in ['ppg', 'eeg', 'imu', 'temp', 'bioz']:
        sensor_data_bilateral = fusion.extract_sensor_by_type(sensor_data, sensor_type)
        left_count = len(sensor_data_bilateral['left'])
        right_count = len(sensor_data_bilateral['right'])
        print(f"{sensor_type.upper()}: {left_count} left samples, {right_count} right samples")
    
    # Test bilateral combination
    print("\n=== Testing Bilateral Data Combination ===")
    ppg_data = fusion.extract_sensor_by_type(sensor_data, 'ppg')
    combined_ppg = fusion.combine_bilateral_data(ppg_data['left'], ppg_data['right'])
    print(f"Combined PPG data: {len(combined_ppg)} time points")
    
    # Test feature extraction
    print("\n=== Testing Feature Extraction ===")
    
    # PPG features
    ppg_features = fusion.extract_ppg_features(combined_ppg)
    print(f"PPG features extracted: {len(ppg_features)}")
    for key, value in list(ppg_features.items())[:5]:
        print(f"  {key}: {value:.4f}")
    
    # EEG features
    eeg_data = fusion.extract_sensor_by_type(sensor_data, 'eeg')
    combined_eeg = fusion.combine_bilateral_data(eeg_data['left'], eeg_data['right'])
    eeg_features = fusion.extract_eeg_features(combined_eeg)
    print(f"EEG features extracted: {len(eeg_features)}")
    
    # Test complete feature vector creation
    print("\n=== Testing Complete Feature Vector Creation ===")
    feature_vector = fusion.create_feature_vector(sensor_data)
    feature_names = fusion.get_feature_names()
    
    print(f"Total features generated: {len(feature_vector)}")
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"Non-zero features: {np.count_nonzero(feature_vector)}")
    
    # Show sample features
    print("\nSample features:")
    for i, (name, value) in enumerate(zip(feature_names, feature_vector)):
        if i < 10:  # Show first 10 features
            print(f"  {name}: {value:.4f}")
        elif i == 10:
            print("  ...")
            break
    
    # Test DataFrame creation
    print("\n=== Testing DataFrame Creation ===")
    df = fusion.create_feature_dataframe(sensor_data)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)[:5]}...")  # Show first 5 column names
    
    # Show feature statistics
    print("\n=== Feature Statistics ===")
    print(f"Mean feature value: {np.mean(feature_vector):.4f}")
    print(f"Std feature value: {np.std(feature_vector):.4f}")
    print(f"Min feature value: {np.min(feature_vector):.4f}")
    print(f"Max feature value: {np.max(feature_vector):.4f}")
    
    # Test cross-sensor features
    print("\n=== Cross-Sensor Features ===")
    cross_features = {k: v for k, v in zip(feature_names, feature_vector) 
                     if any(keyword in k for keyword in ['interaction', 'bilateral', 'asymmetry'])}
    print(f"Cross-sensor features: {len(cross_features)}")
    for name, value in cross_features.items():
        print(f"  {name}: {value:.4f}")
    
    print("\n=== Sensor Fusion Pipeline Test Complete ===")
    print("✓ All components working successfully!")
    
    return fusion, sensor_data, feature_vector, df

if __name__ == "__main__":
    fusion, data, features, df = main()