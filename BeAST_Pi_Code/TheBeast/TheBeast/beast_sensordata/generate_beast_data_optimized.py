#!/usr/bin/env python3
"""
Generate comprehensive BeAST sensor data for 5 minutes (300 seconds)
with realistic physiological variations and proper sampling frequencies.
Optimized version with more efficient data structure.
"""

import json
import math
import random

def generate_realistic_variations():
    """Generate realistic physiological variations over time"""
    return {
        'hr_trend': random.uniform(-2, 2),  # Overall heart rate trend
        'temp_trend': random.uniform(-0.1, 0.1),  # Temperature drift
        'activity_level': random.uniform(0.5, 1.5)  # Activity multiplier
    }

def create_beast_entry(timestamp, side, second, variations):
    """Create a single BeAST data entry for one second"""
    
    # Heart rate with realistic variation (70-75 bpm base)
    base_hr = 72 + variations['hr_trend'] + random.uniform(-2, 2)
    heart_rate = max(65, min(80, int(base_hr)))
    
    # SpO2 (97-99%)
    spo2 = random.randint(97, 99)
    
    # Body temperature (98.4-98.8°F)
    body_temp = 98.6 + variations['temp_trend'] + random.uniform(-0.2, 0.2)
    body_temp = round(max(98.4, min(98.8, body_temp)), 1)
    
    # Ambient temperature (70-75°F)
    ambient_temp = round(72 + random.uniform(-2, 3), 1)
    
    # IMU data with realistic motion
    accel_x = round(random.uniform(-0.3, 0.3) * variations['activity_level'], 3)
    accel_y = round(random.uniform(-0.3, 0.3) * variations['activity_level'], 3)
    accel_z = round(-0.98 + random.uniform(-0.1, 0.1), 3)
    
    gyro_x = round(random.uniform(-10, 10) * variations['activity_level'], 1)
    gyro_y = round(random.uniform(-10, 10) * variations['activity_level'], 1)
    gyro_z = round(random.uniform(-5, 5), 1)
    
    steps = second // 10  # Rough step count
    activity = 1 if abs(accel_x) > 0.2 or abs(accel_y) > 0.2 else 0
    
    # Bioimpedance (125-135 kOhms)
    bioz = round(130 + random.uniform(-5, 5), 1)
    
    # Generate EEG samples (250 samples per second, per channel)
    eeg_channels = []
    for ch in range(1, 9):
        # Generate 250 samples for this second
        base_value = 35 + (ch - 1) * 2  # Channel offset
        samples = []
        for i in range(250):
            t = (second + i/250) / 60  # Time in minutes
            # Simulate brain wave patterns
            alpha = 6 * math.sin(2 * math.pi * 10 * t)  # 10Hz alpha
            theta = 3 * math.sin(2 * math.pi * 6 * t)   # 6Hz theta
            beta = 2 * math.sin(2 * math.pi * 20 * t)   # 20Hz beta
            noise = random.uniform(-3, 3)
            
            value = base_value + alpha + theta + beta + noise
            samples.append(max(26, min(48, int(value))))
        
        eeg_channels.append({
            "number": ch,
            "set": samples
        })
    
    # Create device entries
    devices = [
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
            "data": [heart_rate, spo2]
        },
        {
            "name": "imu",
            "status": "active",
            "freq": 500.0,
            "data": {
                "accel": [accel_x, accel_y, accel_z],
                "gyro": [gyro_x, gyro_y, gyro_z],
                "steps": steps,
                "activity": activity
            }
        },
        {
            "name": "bioz",
            "status": "active",
            "freq": 800.0,
            "data": [bioz]
        },
        {
            "name": "eeg",
            "status": "active",
            "freq": 250.0,
            "data": {
                "channels": eeg_channels
            }
        }
    ]
    
    # Add temperature data every 5 seconds (0.2Hz)
    if second % 5 == 0:
        devices.append({
            "name": "temp",
            "status": "active",
            "freq": 0.2,
            "data": [body_temp, ambient_temp]
        })
    
    return {
        "time": timestamp,
        "side": side,
        "devices": devices
    }

def main():
    print("Starting optimized BeAST sensor data generation...")
    print("Generating 5 minutes (300 seconds) of comprehensive sensor data...")
    
    duration_seconds = 300
    start_time = [2025, 10, 12, 14, 30, 0]
    
    # Generate realistic physiological variations
    left_variations = generate_realistic_variations()
    right_variations = generate_realistic_variations()
    
    beast_data = []
    
    for second in range(duration_seconds):
        # Calculate current timestamp
        current_time = start_time.copy()
        current_time[5] += second
        
        # Handle minute/hour overflow
        if current_time[5] >= 60:
            current_time[4] += current_time[5] // 60
            current_time[5] = current_time[5] % 60
        if current_time[4] >= 60:
            current_time[3] += current_time[4] // 60
            current_time[4] = current_time[4] % 60
        
        # Generate left ear data
        left_entry = create_beast_entry(current_time, "left", second, left_variations)
        beast_data.append(left_entry)
        
        # Generate right ear data (with slight variations)
        right_variations_adjusted = {
            'hr_trend': right_variations['hr_trend'] + random.uniform(-0.5, 0.5),
            'temp_trend': right_variations['temp_trend'] + random.uniform(-0.02, 0.02),
            'activity_level': right_variations['activity_level'] * random.uniform(0.95, 1.05)
        }
        right_entry = create_beast_entry(current_time, "right", second, right_variations_adjusted)
        beast_data.append(right_entry)
        
        # Progress update
        if (second + 1) % 60 == 0:
            print(f"Generated data for {second + 1}/{duration_seconds} seconds...")
    
    # Save to file
    output_file = "/Users/anantsingh/beast_sensordata/beast_sensor_data_5min_2025-10-12_14-30-00.json"
    print(f"Saving data to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(beast_data, f, indent=1)
    
    print("Data generation complete!")
    print(f"Total entries: {len(beast_data)} (300 seconds × 2 ears)")
    print(f"Each entry contains:")
    print("- PPG: Heart rate and SpO2 readings")
    print("- EEG: 250 samples × 8 channels per second")
    print("- IMU: Accelerometer, gyroscope, step count, activity flags")
    print("- Temperature: Body and ambient temperature (every 5 seconds)")
    print("- Bioimpedance: Impedance readings")
    
    # Verify file was created and get size
    import os
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"File size: {file_size:.1f} MB")
        
        # Quick verification of data structure
        with open(output_file, 'r') as f:
            data = json.load(f)
            print(f"Verification: Loaded {len(data)} entries successfully")
            
            # Check first entry structure
            first_entry = data[0]
            print(f"First entry timestamp: {first_entry['time']}")
            print(f"First entry side: {first_entry['side']}")
            print(f"Number of devices in first entry: {len(first_entry['devices'])}")
            
            # Count EEG samples in first entry
            for device in first_entry['devices']:
                if device['name'] == 'eeg':
                    total_eeg_samples = sum(len(ch['set']) for ch in device['data']['channels'])
                    print(f"EEG samples in first entry: {total_eeg_samples}")
                    break

if __name__ == "__main__":
    main()