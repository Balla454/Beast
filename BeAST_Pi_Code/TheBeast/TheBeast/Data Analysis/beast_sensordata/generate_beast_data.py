#!/usr/bin/env python3
"""
Generate comprehensive BeAST sensor data for 5 minutes (300 seconds)
with realistic physiological variations and proper sampling frequencies.
"""

import json
import math
import random
import numpy as np
from datetime import datetime, timedelta

def generate_ppg_data(duration_seconds, heart_rate_base=72):
    """Generate PPG data with realistic heart rate and SpO2 variations"""
    samples = []
    spo2_base = 98
    
    for second in range(duration_seconds):
        # Slight heart rate variation (±3 bpm)
        hr_variation = random.uniform(-3, 3)
        heart_rate = max(65, min(80, heart_rate_base + hr_variation))
        
        # SpO2 variation (97-99%)
        spo2_variation = random.uniform(-1, 1)
        spo2 = max(97, min(99, spo2_base + spo2_variation))
        
        samples.append([int(heart_rate), int(spo2)])
    
    return samples

def generate_eeg_channel_data(duration_seconds, freq=250, channel_num=1):
    """Generate realistic EEG data for one channel"""
    total_samples = duration_seconds * freq
    samples = []
    
    # Different frequency components for realistic brain activity
    base_freq = 10  # Alpha waves around 10Hz
    theta_freq = 6  # Theta waves
    beta_freq = 20  # Beta waves
    
    for i in range(total_samples):
        t = i / freq
        
        # Combine multiple frequency components
        alpha_wave = 8 * math.sin(2 * math.pi * base_freq * t)
        theta_wave = 4 * math.sin(2 * math.pi * theta_freq * t)
        beta_wave = 2 * math.sin(2 * math.pi * beta_freq * t)
        
        # Add noise and baseline
        noise = random.uniform(-2, 2)
        baseline = 35  # mV baseline
        
        # Channel-specific variations
        channel_offset = (channel_num - 1) * 2
        
        value = baseline + alpha_wave + theta_wave + beta_wave + noise + channel_offset
        samples.append(max(26, min(48, int(value))))
    
    return samples

def generate_imu_data(duration_seconds, freq=500):
    """Generate IMU data with realistic motion patterns"""
    total_samples = duration_seconds * freq
    samples = []
    step_count = 0
    
    for i in range(total_samples):
        t = i / freq
        
        # Simulate slight head movements and gravity
        accel_x = random.uniform(-0.2, 0.2) + 0.1 * math.sin(0.5 * t)
        accel_y = random.uniform(-0.2, 0.2) + 0.05 * math.cos(0.3 * t)
        accel_z = -0.98 + random.uniform(-0.1, 0.1)  # Gravity with noise
        
        # Gyroscope data (small rotational movements)
        gyro_x = random.uniform(-5, 5) + 2 * math.sin(0.2 * t)
        gyro_y = random.uniform(-5, 5) + 1.5 * math.cos(0.15 * t)
        gyro_z = random.uniform(-3, 3)
        
        # Step detection (simple threshold-based)
        if i % (freq // 2) == 0:  # Check twice per second
            accel_magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            if accel_magnitude > 1.2:  # Walking threshold
                step_count += random.choice([0, 1])  # Occasional step
        
        activity_flag = 1 if abs(accel_x) > 0.15 or abs(accel_y) > 0.15 else 0
        
        samples.append({
            "accel": [round(accel_x, 3), round(accel_y, 3), round(accel_z, 3)],
            "gyro": [round(gyro_x, 1), round(gyro_y, 1), round(gyro_z, 1)],
            "steps": step_count,
            "activity": activity_flag
        })
    
    return samples

def generate_temp_data(duration_seconds, freq=0.2):
    """Generate temperature data (body and ambient)"""
    samples = []
    body_temp_base = 98.6  # Fahrenheit
    ambient_temp_base = 72.0
    
    # Only sample at the specified frequency
    sample_interval = int(1.0 / freq)  # Every 5 seconds
    
    for second in range(0, duration_seconds, sample_interval):
        # Body temperature variation (98.4-98.8°F)
        body_variation = random.uniform(-0.2, 0.2)
        body_temp = body_temp_base + body_variation
        
        # Ambient temperature variation (70-75°F)
        ambient_variation = random.uniform(-2, 3)
        ambient_temp = ambient_temp_base + ambient_variation
        
        samples.append([round(body_temp, 1), round(ambient_temp, 1)])
    
    return samples

def generate_bioz_data(duration_seconds, freq=800):
    """Generate bioimpedance data"""
    total_samples = duration_seconds * freq
    samples = []
    base_impedance = 130.0  # kOhms
    
    for i in range(total_samples):
        t = i / freq
        
        # Slight variations due to breathing and blood flow
        breathing_variation = 2 * math.sin(2 * math.pi * 0.25 * t)  # 15 breaths/min
        blood_flow_variation = 0.5 * math.sin(2 * math.pi * 1.2 * t)  # Heart rate effect
        noise = random.uniform(-0.5, 0.5)
        
        impedance = base_impedance + breathing_variation + blood_flow_variation + noise
        samples.append(round(max(125, min(135, impedance)), 1))
    
    return samples

def generate_beast_data():
    """Generate complete BeAST sensor data for 5 minutes"""
    duration_seconds = 300
    start_time = [2025, 10, 12, 14, 30, 0]
    
    # Generate all sensor data
    print("Generating PPG data...")
    ppg_data = generate_ppg_data(duration_seconds)
    
    print("Generating EEG data...")
    eeg_data = {}
    for channel in range(1, 9):
        eeg_data[channel] = generate_eeg_channel_data(duration_seconds, channel_num=channel)
    
    print("Generating IMU data...")
    imu_data = generate_imu_data(duration_seconds)
    
    print("Generating temperature data...")
    temp_data = generate_temp_data(duration_seconds)
    
    print("Generating bioimpedance data...")
    bioz_data = generate_bioz_data(duration_seconds)
    
    # Create the main data structure
    beast_data = []
    
    for second in range(duration_seconds):
        # Calculate current timestamp
        current_time = start_time.copy()
        current_time[5] += second  # Add seconds
        
        # Handle minute overflow
        if current_time[5] >= 60:
            current_time[4] += current_time[5] // 60
            current_time[5] = current_time[5] % 60
        
        # Handle hour overflow
        if current_time[4] >= 60:
            current_time[3] += current_time[4] // 60
            current_time[4] = current_time[4] % 60
        
        # Left ear data
        left_entry = {
            "time": current_time,
            "side": "left",
            "devices": [
                {
                    "name": "mcu",
                    "status": "active",
                    "freq": 1.0,
                    "data": [1, 1, 1, 1]
                },
                {
                    "name": "ppg",
                    "status": "active",
                    "freq": 100.0,
                    "data": ppg_data[second]
                },
                {
                    "name": "bioz",
                    "status": "active",
                    "freq": 800.0,
                    "data": [bioz_data[second * 800:(second + 1) * 800]]
                },
                {
                    "name": "eeg",
                    "status": "active",
                    "freq": 250.0,
                    "data": {
                        "channels": [
                            {
                                "number": i,
                                "set": eeg_data[i][second * 250:(second + 1) * 250]
                            } for i in range(1, 9)
                        ]
                    }
                },
                {
                    "name": "imu",
                    "status": "active",
                    "freq": 500.0,
                    "data": imu_data[second * 500:(second + 1) * 500]
                }
            ]
        }
        
        # Add temperature data only when available (every 5 seconds)
        if second % 5 == 0:
            temp_index = second // 5
            if temp_index < len(temp_data):
                left_entry["devices"].append({
                    "name": "temp",
                    "status": "active",
                    "freq": 0.2,
                    "data": temp_data[temp_index]
                })
        
        # Right ear data (similar but with slight variations)
        right_entry = {
            "time": current_time,
            "side": "right",
            "devices": [
                {
                    "name": "mcu",
                    "status": "active",
                    "freq": 1.0,
                    "data": [1, 1, 1, 1]
                },
                {
                    "name": "ppg",
                    "status": "active",
                    "freq": 100.0,
                    "data": [ppg_data[second][0] + random.randint(-1, 1), ppg_data[second][1]]
                },
                {
                    "name": "bioz",
                    "status": "active",
                    "freq": 800.0,
                    "data": [bioz_data[second * 800:(second + 1) * 800]]
                },
                {
                    "name": "eeg",
                    "status": "active",
                    "freq": 250.0,
                    "data": {
                        "channels": [
                            {
                                "number": i,
                                "set": [v + random.randint(-1, 1) for v in eeg_data[i][second * 250:(second + 1) * 250]]
                            } for i in range(1, 9)
                        ]
                    }
                },
                {
                    "name": "imu",
                    "status": "active",
                    "freq": 500.0,
                    "data": [{
                        "accel": [
                            sample["accel"][0] + random.uniform(-0.02, 0.02),
                            sample["accel"][1] + random.uniform(-0.02, 0.02),
                            sample["accel"][2] + random.uniform(-0.02, 0.02)
                        ],
                        "gyro": [
                            sample["gyro"][0] + random.uniform(-0.5, 0.5),
                            sample["gyro"][1] + random.uniform(-0.5, 0.5),
                            sample["gyro"][2] + random.uniform(-0.5, 0.5)
                        ],
                        "steps": sample["steps"],
                        "activity": sample["activity"]
                    } for sample in imu_data[second * 500:(second + 1) * 500]]
                }
            ]
        }
        
        # Add temperature data for right ear
        if second % 5 == 0:
            temp_index = second // 5
            if temp_index < len(temp_data):
                right_entry["devices"].append({
                    "name": "temp",
                    "status": "active",
                    "freq": 0.2,
                    "data": [temp_data[temp_index][0] + random.uniform(-0.1, 0.1), 
                            temp_data[temp_index][1]]
                })
        
        beast_data.extend([left_entry, right_entry])
        
        if (second + 1) % 30 == 0:
            print(f"Generated data for {second + 1}/{duration_seconds} seconds...")
    
    return beast_data

def main():
    print("Starting BeAST sensor data generation...")
    print("This will generate 5 minutes (300 seconds) of comprehensive sensor data.")
    print("Expected sample counts:")
    print("- PPG: 30,000 samples (100Hz × 300s)")
    print("- EEG: 75,000 samples per channel × 8 channels (250Hz × 300s)")
    print("- IMU: 150,000 samples (500Hz × 300s)")
    print("- IR Thermometer: 60 samples (0.2Hz × 300s)")
    print("- Bioimpedance: 240,000 samples (800Hz × 300s)")
    print()
    
    # Generate the data
    beast_data = generate_beast_data()
    
    # Save to file
    output_file = "/Users/anantsingh/beast_sensordata/beast_sensor_data_5min_2025-10-12_14-30-00.json"
    print(f"Saving data to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(beast_data, f, indent=2)
    
    print(f"Data generation complete!")
    print(f"Total entries: {len(beast_data)}")
    print(f"File size: ~{len(json.dumps(beast_data)) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()