# BeAST Sensor Fusion Pipeline

## Overview
This sensor fusion pipeline combines bilateral (left/right ear) sensor data from the BeAST system and creates feature vectors for machine learning models.

## Features Generated

### 1. **Bilateral Data Combination**
- Aligns left and right ear sensor data by timestamp
- Handles missing data gracefully
- Creates combined bilateral datasets

### 2. **PPG Features (15 features)**
- **Per ear**: Heart rate (mean, std, min, max), SpO2 (mean, std)
- **Cross-ear**: Bilateral differences, correlation between ears

### 3. **EEG Features (120 features)**
- **Per channel/ear**: Mean, std, RMS, skewness, kurtosis, total power, peak frequency
- **8 channels × 2 ears × 7 features = 112 features**
- **Cross-hemisphere**: Bilateral differences for each channel (8 features)

### 4. **IMU Features (Variable)**
- **Accelerometer**: Magnitude and individual axes (X, Y, Z)
- **Gyroscope**: Magnitude and individual axes (X, Y, Z) 
- **Activity**: Activity flags, To be modified: Step count
- **Cross-ear**: Bilateral motion differences

### 5. **Temperature Features (8 features)**
- **Per ear**: Body temperature (mean, std), ambient temperature (mean)
- **Cross-ear**: Bilateral temperature gradients

### 6. **Bioimpedance Features (8 features)**
- **Per ear**: Mean, std, median impedance values
- **Cross-ear**: Bilateral differences and ratios

### 7. **Cross-Sensor Interaction Features (19 features)**
- **HR-Temperature interaction**: Cardiovascular-thermal coupling
- **Motion-EEG artifacts**: Motion impact on brain signals
- **HRV-EEG stress indicators**: Heart rate variability vs brain activity
- **Temperature-Bioimpedance hydration**: Thermal-electrical tissue properties
- **Bilateral asymmetry index**: Overall left-right imbalance metric

## Total Features Generated
**178 features** from the test dataset, including:
- Individual sensor features from both ears
- Cross-sensor interactions
- Bilateral comparisons
- Statistical aggregations

## Usage

### Basic Usage
```python
from sensor_fusion import BeASTSensorFusion

# Initialize fusion pipeline
fusion = BeASTSensorFusion()

# Load sensor data
data = fusion.load_sensor_data('your_data_file.json')

# Create feature vector for ML
features = fusion.create_feature_vector(data)
feature_names = fusion.get_feature_names()

print(f"Generated {len(features)} features")
```

### Batch Processing
```python
# Process multiple files
file_paths = ['file1.json', 'file2.json', 'file3.json']
feature_matrix = fusion.batch_process_files(file_paths)

# Creates DataFrame with shape (n_files, n_features)
print(f"Feature matrix shape: {feature_matrix.shape}")
```

### DataFrame Creation
```python
# Create pandas DataFrame for analysis
df = fusion.create_feature_dataframe(data)

# Easy analysis and visualization
print(df.describe())
df.to_csv('features.csv')
```

## Data Format Support
The pipeline handles multiple data formats:
- **Legacy format**: Flat lists for sensor data
- **Current format**: Structured dictionaries with proper sensor organization
- **Flexible keys**: Supports both 'device'/'devices' and 'channel'/'channels'

## Key Capabilities

### 1. **Robust Data Handling**
- Handles missing sensors data
- Supports different sampling frequencies
- Aligns timestamps across sensors

### 2. **ML-Ready Output**
- Numpy arrays for direct ML model input
- Pandas DataFrames for analysis
- Consistent feature ordering
- Feature name mapping

### 3. **Physiological Insights**
- Cross-sensor correlations reveal physiological states
- Bilateral asymmetry detection
- Motion artifact identification
- Stress and fatigue indicators

### 4. **Scalable Processing**
- Batch processing for multiple sessions
- Memory-efficient feature extraction
- Configurable feature sets

## Applications

### Machine Learning Models
- **Classification**: Stress, fatigue, alertness states
- **Regression**: Cognitive load, performance prediction
- **Anomaly Detection**: Health monitoring, injury detection
- **Time Series**: Longitudinal health tracking

### Feature Engineering
- **Domain-specific features**: Military operational readiness
- **Cross-modal fusion**: Multi-sensor state estimation
- **Temporal features**: Can be extended for time-window analysis

## Performance
- **Test Results**: Successfully processed 600 sensor samples (5 minutes of data)
- **Feature Generation**: 178 features in <1 second
- **Memory Efficient**: Handles large datasets through batch processing

## Files
- `sensor_fusion.py`: Main fusion pipeline class
- `test_sensor_fusion.py`: Comprehensive test and demonstration script
- `README_sensor_fusion.md`: This documentation

The pipeline is ready for integration with ML models and provides a comprehensive feature set for BeAST sensor data analysis.