#!/usr/bin/env python3
"""
beast Sensor Interface
======================
Handles communication with beast earpiece sensors.
Parses JSONL data stream from Arduino/sensors.
"""

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger('beast.Sensors')


@dataclass
class SensorReading:
    """Single sensor reading from earpiece"""
    timestamp: float
    side: str  # 'left' or 'right'
    eeg: List[float]  # 8 channels
    ppg_ir: float
    ppg_red: float
    imu_accel: List[float]  # x, y, z
    imu_gyro: List[float]  # x, y, z
    temperature: float
    bioimpedance: float


class SensorInterface:
    """
    Interface for beast earpiece sensors.
    
    Supports:
    - Serial connection to Arduino
    - JSONL file replay for testing
    - Real-time data streaming
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize sensor interface.
        
        Args:
            config: Configuration dict with connection settings
        """
        self.config = config or {}
        self.running = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.callbacks: List[Callable] = []
        self._thread = None
        self._serial = None
        
    def connect(self, port: str = None, baudrate: int = 115200) -> bool:
        """
        Connect to sensor hardware.
        
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0')
            baudrate: Serial baudrate
            
        Returns:
            True if connected successfully
        """
        port = port or self.config.get('serial_port', '/dev/ttyUSB0')
        
        try:
            import serial
            self._serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1.0
            )
            logger.info(f"Connected to sensors on {port}")
            return True
        except ImportError:
            logger.warning("pyserial not installed, using simulation mode")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to sensors: {e}")
            return False
            
    def start_streaming(self, source: str = None):
        """
        Start receiving sensor data.
        
        Args:
            source: Optional JSONL file for replay mode
        """
        if self.running:
            return
            
        self.running = True
        
        if source and Path(source).exists():
            # Replay mode
            self._thread = threading.Thread(
                target=self._replay_stream,
                args=(source,),
                daemon=True
            )
        elif self._serial:
            # Serial mode
            self._thread = threading.Thread(
                target=self._serial_stream,
                daemon=True
            )
        else:
            # Simulation mode
            self._thread = threading.Thread(
                target=self._simulate_stream,
                daemon=True
            )
            
        self._thread.start()
        logger.info("Sensor streaming started")
        
    def stop_streaming(self):
        """Stop receiving sensor data"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Sensor streaming stopped")
        
    def get_reading(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get next sensor reading.
        
        Args:
            timeout: Max time to wait for reading
            
        Returns:
            Sensor data dict or None
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def register_callback(self, callback: Callable):
        """
        Register callback for sensor data.
        
        Args:
            callback: Function called with each reading
        """
        self.callbacks.append(callback)
        
    def _process_data(self, data: Dict):
        """Process and distribute sensor data"""
        # Add to queue
        try:
            self.data_queue.put_nowait(data)
        except queue.Full:
            # Drop oldest reading
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(data)
            except:
                pass
                
        # Call registered callbacks
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
                
    def _serial_stream(self):
        """Read from serial connection"""
        buffer = ""
        
        while self.running:
            try:
                if self._serial.in_waiting:
                    buffer += self._serial.read(self._serial.in_waiting).decode('utf-8')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line:
                            try:
                                data = json.loads(line)
                                self._process_data(data)
                            except json.JSONDecodeError:
                                pass
                else:
                    time.sleep(0.001)  # Avoid busy waiting
                    
            except Exception as e:
                logger.error(f"Serial read error: {e}")
                time.sleep(0.1)
                
    def _replay_stream(self, filepath: str):
        """Replay data from JSONL file"""
        logger.info(f"Replaying from {filepath}")
        
        with open(filepath, 'r') as f:
            last_ts = None
            
            for line in f:
                if not self.running:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Maintain timing
                    ts = data.get('timestamp_ms', 0) / 1000.0
                    if last_ts is not None:
                        delay = ts - last_ts
                        if 0 < delay < 1.0:  # Max 1 second delay
                            time.sleep(delay)
                    last_ts = ts
                    
                    self._process_data(data)
                    
                except json.JSONDecodeError:
                    continue
                    
        logger.info("Replay complete")
        
    def _simulate_stream(self):
        """Generate simulated sensor data"""
        import random
        import math
        
        logger.info("Using simulated sensor data")
        
        t = 0
        while self.running:
            # Generate realistic-ish data
            data = {
                'timestamp_ms': int(time.time() * 1000),
                'left': {
                    'eeg': [random.gauss(0, 10) + 5 * math.sin(2 * math.pi * 10 * t) for _ in range(8)],
                    'ppg_ir': 50000 + 5000 * math.sin(2 * math.pi * 1.2 * t) + random.gauss(0, 500),
                    'ppg_red': 40000 + 4000 * math.sin(2 * math.pi * 1.2 * t) + random.gauss(0, 400),
                    'imu_accel': [random.gauss(0, 0.1), random.gauss(0, 0.1), random.gauss(9.8, 0.1)],
                    'imu_gyro': [random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)],
                    'temperature': 36.5 + random.gauss(0, 0.1),
                    'bioimpedance': 500 + random.gauss(0, 10)
                },
                'right': {
                    'eeg': [random.gauss(0, 10) + 5 * math.sin(2 * math.pi * 10 * t + 0.1) for _ in range(8)],
                    'ppg_ir': 51000 + 5100 * math.sin(2 * math.pi * 1.2 * t) + random.gauss(0, 510),
                    'ppg_red': 41000 + 4100 * math.sin(2 * math.pi * 1.2 * t) + random.gauss(0, 410),
                    'imu_accel': [random.gauss(0, 0.1), random.gauss(0, 0.1), random.gauss(9.8, 0.1)],
                    'imu_gyro': [random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)],
                    'temperature': 36.6 + random.gauss(0, 0.1),
                    'bioimpedance': 505 + random.gauss(0, 10)
                }
            }
            
            self._process_data(data)
            
            t += 0.004  # 250 Hz
            time.sleep(0.004)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing SensorInterface...")
    
    sensor = SensorInterface()
    sensor.start_streaming()  # Simulation mode
    
    print("Reading 10 samples...")
    for i in range(10):
        reading = sensor.get_reading(timeout=1.0)
        if reading:
            print(f"  Sample {i+1}: left_hr_proxy={reading['left']['ppg_ir']:.0f}")
        time.sleep(0.1)
        
    sensor.stop_streaming()
    print("Test complete!")
