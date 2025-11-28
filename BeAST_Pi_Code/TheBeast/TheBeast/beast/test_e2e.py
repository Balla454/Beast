#!/usr/bin/env python3
"""
BeAST End-to-End Test with Simulated Data
==========================================
Tests the complete pipeline:
1. Simulated sensor data generation
2. Feature extraction
3. Metric calculation
4. Database storage
5. RAG query processing
6. Voice output (optional)

Run: python3 test_e2e.py
"""

import sys
import time
import random
import math
import logging
import tempfile
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('BeAST.E2E')

# Add beast directory to path
BEAST_DIR = Path(__file__).parent
sys.path.insert(0, str(BEAST_DIR))


class SimulatedSensorStream:
    """Generate realistic simulated sensor data"""
    
    def __init__(self, scenario: str = "normal"):
        """
        Initialize simulator with a scenario.
        
        Scenarios:
        - normal: Healthy baseline
        - stressed: Elevated stress/cognitive load
        - fatigued: High fatigue, low alertness
        - active: Elevated HR, high activity
        - dehydrated: Low hydration
        """
        self.scenario = scenario
        self.t = 0
        self.sample_rate = 250  # Hz
        
        # Scenario parameters
        self.params = self._get_scenario_params(scenario)
        
    def _get_scenario_params(self, scenario: str) -> dict:
        """Get parameters for each scenario"""
        base = {
            'hr_base': 72, 'hr_var': 5,
            'spo2_base': 98, 'spo2_var': 1,
            'cognitive_base': 35, 'cognitive_var': 10,
            'alertness_base': 75, 'alertness_var': 5,
            'fatigue_base': 25, 'fatigue_var': 5,
            'hydration_base': 70, 'hydration_var': 5,
            'temp_base': 36.6, 'temp_var': 0.2,
            'motion_base': 9.8, 'motion_var': 0.2,
        }
        
        if scenario == "stressed":
            base.update({
                'hr_base': 95, 'hr_var': 10,
                'cognitive_base': 75, 'cognitive_var': 10,
                'alertness_base': 80, 'alertness_var': 5,
                'fatigue_base': 45, 'fatigue_var': 10,
            })
        elif scenario == "fatigued":
            base.update({
                'hr_base': 65, 'hr_var': 5,
                'cognitive_base': 60, 'cognitive_var': 10,
                'alertness_base': 35, 'alertness_var': 10,
                'fatigue_base': 78, 'fatigue_var': 8,
            })
        elif scenario == "active":
            base.update({
                'hr_base': 125, 'hr_var': 15,
                'spo2_base': 96, 'spo2_var': 2,
                'motion_base': 14, 'motion_var': 2,
                'temp_base': 37.2, 'temp_var': 0.3,
            })
        elif scenario == "dehydrated":
            base.update({
                'hr_base': 85, 'hr_var': 8,
                'hydration_base': 35, 'hydration_var': 5,
                'fatigue_base': 55, 'fatigue_var': 10,
            })
            
        return base
        
    def generate_sample(self) -> dict:
        """Generate one sensor sample"""
        p = self.params
        t = self.t / self.sample_rate
        
        # Add slow drift and fast variations
        drift = math.sin(2 * math.pi * t / 60) * 0.1  # 1-minute cycle
        
        # Heart rate with realistic variation
        hr = p['hr_base'] + p['hr_var'] * (math.sin(2 * math.pi * t / 30) + random.gauss(0, 0.3))
        
        # SpO2 (very stable normally)
        spo2 = p['spo2_base'] + random.gauss(0, p['spo2_var'] * 0.3)
        spo2 = max(88, min(100, spo2))
        
        # Cognitive metrics (slower variation)
        cognitive = p['cognitive_base'] + p['cognitive_var'] * (drift + random.gauss(0, 0.2))
        alertness = p['alertness_base'] + p['alertness_var'] * (-drift + random.gauss(0, 0.2))
        fatigue = p['fatigue_base'] + p['fatigue_var'] * (drift * 0.5 + random.gauss(0, 0.2))
        
        # Clamp to valid ranges
        cognitive = max(0, min(100, cognitive))
        alertness = max(0, min(100, alertness))
        fatigue = max(0, min(100, fatigue))
        
        # Hydration (very slow change)
        hydration = p['hydration_base'] + random.gauss(0, 1)
        hydration = max(0, min(100, hydration))
        
        # Temperature
        temp = p['temp_base'] + p['temp_var'] * random.gauss(0, 0.3)
        
        # Motion (accelerometer magnitude)
        motion = p['motion_base'] + p['motion_var'] * abs(random.gauss(0, 1))
        
        # EEG bands (simplified)
        alpha = 10 + 5 * (1 - cognitive/100) + random.gauss(0, 1)
        beta = 8 + 5 * (cognitive/100) + random.gauss(0, 1)
        theta = 5 + 3 * (fatigue/100) + random.gauss(0, 0.5)
        delta = 12 + 5 * (fatigue/100) + random.gauss(0, 1)
        gamma = 3 + 2 * (alertness/100) + random.gauss(0, 0.3)
        
        self.t += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'heart_rate': hr,
            'spo2': spo2,
            'cognitive_load': cognitive,
            'alertness': alertness,
            'fatigue': fatigue,
            'hydration': hydration,
            'core_temp': temp,
            'motion_magnitude': motion,
            'eeg_alpha': max(0, alpha),
            'eeg_beta': max(0, beta),
            'eeg_theta': max(0, theta),
            'eeg_delta': max(0, delta),
            'eeg_gamma': max(0, gamma),
            'ambient_temp': 23.5 + random.gauss(0, 0.5),
            'humidity': 55 + random.gauss(0, 3),
            'heat_index': 24 + random.gauss(0, 1),
        }
        
    def generate_history(self, duration_seconds: int = 3600) -> list:
        """Generate historical data for trend queries"""
        samples = []
        num_samples = duration_seconds  # 1 sample per second for aggregates
        
        for i in range(num_samples):
            # Temporarily adjust time
            self.t = i * self.sample_rate
            sample = self.generate_sample()
            sample['timestamp'] = (datetime.now() - timedelta(seconds=duration_seconds-i)).isoformat()
            samples.append(sample)
            
        return samples


class MockDatabase:
    """Mock database that mimics the PostgreSQL schema"""
    
    def __init__(self):
        # Use in-memory SQLite for testing
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self.latest_data = {}
        
    def _create_tables(self):
        """Create tables matching Antonio's schema"""
        cur = self.conn.cursor()
        
        # Simplified version of the schema
        cur.execute('''
            CREATE TABLE ppg_reading (
                id INTEGER PRIMARY KEY,
                ts TIMESTAMP,
                hr_bpm REAL,
                spo2_pct REAL
            )
        ''')
        
        cur.execute('''
            CREATE TABLE cognitive_metrics (
                id INTEGER PRIMARY KEY,
                ts TIMESTAMP,
                cognitive_load REAL,
                alertness REAL,
                fatigue REAL
            )
        ''')
        
        cur.execute('''
            CREATE TABLE eeg_band (
                id INTEGER PRIMARY KEY,
                ts TIMESTAMP,
                alpha REAL,
                beta REAL,
                gamma REAL,
                delta REAL,
                theta REAL
            )
        ''')
        
        cur.execute('''
            CREATE TABLE hydration_reading (
                id INTEGER PRIMARY KEY,
                ts TIMESTAMP,
                hydration_pct REAL
            )
        ''')
        
        cur.execute('''
            CREATE TABLE temp_reading (
                id INTEGER PRIMARY KEY,
                ts TIMESTAMP,
                core_temp_c REAL
            )
        ''')
        
        cur.execute('''
            CREATE TABLE environment (
                id INTEGER PRIMARY KEY,
                ts TIMESTAMP,
                ambient_temp REAL,
                humidity_pct REAL,
                heat_index_c REAL
            )
        ''')
        
        self.conn.commit()
        
    def insert_sample(self, sample: dict):
        """Insert a sensor sample into the database"""
        cur = self.conn.cursor()
        ts = sample['timestamp']
        
        cur.execute(
            'INSERT INTO ppg_reading (ts, hr_bpm, spo2_pct) VALUES (?, ?, ?)',
            (ts, sample['heart_rate'], sample['spo2'])
        )
        
        cur.execute(
            'INSERT INTO cognitive_metrics (ts, cognitive_load, alertness, fatigue) VALUES (?, ?, ?, ?)',
            (ts, sample['cognitive_load'], sample['alertness'], sample['fatigue'])
        )
        
        cur.execute(
            'INSERT INTO eeg_band (ts, alpha, beta, gamma, delta, theta) VALUES (?, ?, ?, ?, ?, ?)',
            (ts, sample['eeg_alpha'], sample['eeg_beta'], sample['eeg_gamma'], 
             sample['eeg_delta'], sample['eeg_theta'])
        )
        
        cur.execute(
            'INSERT INTO hydration_reading (ts, hydration_pct) VALUES (?, ?)',
            (ts, sample['hydration'])
        )
        
        cur.execute(
            'INSERT INTO temp_reading (ts, core_temp_c) VALUES (?, ?)',
            (ts, sample['core_temp'])
        )
        
        cur.execute(
            'INSERT INTO environment (ts, ambient_temp, humidity_pct, heat_index_c) VALUES (?, ?, ?, ?)',
            (ts, sample['ambient_temp'], sample['humidity'], sample['heat_index'])
        )
        
        self.conn.commit()
        self.latest_data = sample
        
    def get_latest(self) -> dict:
        """Get the latest readings (mimics PostgreSQL query)"""
        return self.latest_data
        
    def get_hourly_stats(self) -> dict:
        """Get hourly statistics"""
        cur = self.conn.cursor()
        
        cur.execute('''
            SELECT AVG(hr_bpm) as avg_hr, MIN(hr_bpm) as min_hr, MAX(hr_bpm) as max_hr,
                   AVG(spo2_pct) as avg_spo2
            FROM ppg_reading
        ''')
        row = cur.fetchone()
        
        if row and row['avg_hr']:
            return {
                'hr_avg_1h': row['avg_hr'],
                'hr_min_1h': row['min_hr'],
                'hr_max_1h': row['max_hr'],
                'spo2_avg_1h': row['avg_spo2']
            }
        return {}


class E2ETestRunner:
    """Run end-to-end tests"""
    
    def __init__(self, scenario: str = "normal", use_voice: bool = False):
        self.scenario = scenario
        self.use_voice = use_voice
        self.sensor = SimulatedSensorStream(scenario)
        self.db = MockDatabase()
        self.rag = None
        
    def setup(self):
        """Set up the test environment"""
        logger.info(f"Setting up E2E test with scenario: {self.scenario}")
        
        # Generate some historical data
        logger.info("Generating 1 hour of simulated sensor history...")
        history = self.sensor.generate_history(3600)
        
        # Insert into database
        for i, sample in enumerate(history):
            if i % 60 == 0:  # Insert 1 per minute for speed
                self.db.insert_sample(sample)
                
        logger.info(f"Inserted {len(history)//60} samples into mock database")
        
        # Initialize RAG with mock data provider
        from rag.health_rag import HealthRAG
        
        self.rag = HealthRAG(config={})
        
        # Override the health context method to use our mock DB
        def get_mock_context():
            latest = self.db.get_latest()
            stats = self.db.get_hourly_stats()
            return {**latest, **stats}
            
        self.rag._get_user_health_context = get_mock_context
        
        logger.info("RAG system initialized with mock data provider")
        
        # Initialize TTS if requested
        if self.use_voice:
            try:
                from voice.text_to_speech import TextToSpeech
                self.tts = TextToSpeech({})
                logger.info("TTS initialized")
            except Exception as e:
                logger.warning(f"TTS not available: {e}")
                self.tts = None
        else:
            self.tts = None
            
    def update_sensors(self):
        """Simulate real-time sensor update"""
        sample = self.sensor.generate_sample()
        self.db.insert_sample(sample)
        return sample
        
    def query(self, question: str) -> str:
        """Process a query through the RAG system"""
        response = self.rag.query(question)
        
        if self.tts and self.use_voice:
            self.tts.speak(response)
            
        return response
        
    def run_test_suite(self):
        """Run comprehensive test suite"""
        print("\n" + "=" * 70)
        print(f"BeAST End-to-End Test - Scenario: {self.scenario.upper()}")
        print("=" * 70)
        
        # Test queries organized by category
        test_queries = [
            # Basic vitals
            ("What is my heart rate?", "heart_rate"),
            ("What's my oxygen level?", "spo2"),
            ("What's my temperature?", "temperature"),
            
            # Cognitive/Mental
            ("What is my cognitive load?", "cognitive"),
            ("How alert am I?", "alertness"),
            ("Am I tired?", "fatigue"),
            ("What is my stress level?", "stress"),
            
            # Physical
            ("What's my hydration level?", "hydration"),
            ("What's my activity level?", "activity"),
            
            # Assessments
            ("Am I fit for duty?", "fitness"),
            ("Do I need a break?", "break"),
            ("Are my vitals normal?", "threshold"),
            
            # Summary
            ("What's my overall status?", "summary"),
        ]
        
        results = []
        
        for question, category in test_queries:
            # Update sensors before each query
            self.update_sensors()
            
            print(f"\n{'─' * 70}")
            print(f"Q: {question}")
            
            start = time.time()
            response = self.query(question)
            elapsed = (time.time() - start) * 1000
            
            print(f"A: {response}")
            print(f"   [{category}] Response time: {elapsed:.1f}ms")
            
            results.append({
                'question': question,
                'category': category,
                'response': response,
                'time_ms': elapsed
            })
            
        # Print summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        avg_time = sum(r['time_ms'] for r in results) / len(results)
        print(f"Total queries: {len(results)}")
        print(f"Average response time: {avg_time:.1f}ms")
        print(f"Scenario: {self.scenario}")
        
        # Show current sensor state
        latest = self.db.get_latest()
        print(f"\nFinal sensor state:")
        print(f"  Heart Rate: {latest.get('heart_rate', 0):.0f} BPM")
        print(f"  SpO2: {latest.get('spo2', 0):.0f}%")
        print(f"  Cognitive Load: {latest.get('cognitive_load', 0):.0f}/100")
        print(f"  Fatigue: {latest.get('fatigue', 0):.0f}/100")
        print(f"  Alertness: {latest.get('alertness', 0):.0f}%")
        print(f"  Hydration: {latest.get('hydration', 0):.0f}%")
        
        return results


def run_interactive_mode(scenario: str = "normal"):
    """Run in interactive mode - ask questions live"""
    print("\n" + "=" * 70)
    print("BeAST Interactive Test Mode")
    print("=" * 70)
    print(f"Scenario: {scenario}")
    print("Type your questions, or 'quit' to exit")
    print("Commands: 'update' - refresh sensors, 'scenario <name>' - change scenario")
    print("=" * 70)
    
    runner = E2ETestRunner(scenario)
    runner.setup()
    
    while True:
        try:
            user_input = input("\n🎤 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
                
            if user_input.lower() == 'update':
                sample = runner.update_sensors()
                print(f"📊 Sensors updated: HR={sample['heart_rate']:.0f}, "
                      f"Fatigue={sample['fatigue']:.0f}, "
                      f"Cognitive={sample['cognitive_load']:.0f}")
                continue
                
            if user_input.lower().startswith('scenario '):
                new_scenario = user_input.split(' ', 1)[1]
                if new_scenario in ['normal', 'stressed', 'fatigued', 'active', 'dehydrated']:
                    runner = E2ETestRunner(new_scenario)
                    runner.setup()
                    print(f"✓ Switched to scenario: {new_scenario}")
                else:
                    print("Unknown scenario. Options: normal, stressed, fatigued, active, dehydrated")
                continue
                
            # Update sensors and query
            runner.update_sensors()
            response = runner.query(user_input)
            print(f"🤖 BeAST: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BeAST End-to-End Test')
    parser.add_argument('--scenario', '-s', 
                        choices=['normal', 'stressed', 'fatigued', 'active', 'dehydrated'],
                        default='normal',
                        help='Test scenario to simulate')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--voice', '-v', action='store_true',
                        help='Enable voice output (TTS)')
    parser.add_argument('--all-scenarios', '-a', action='store_true',
                        help='Run tests for all scenarios')
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_mode(args.scenario)
    elif args.all_scenarios:
        # Run all scenarios
        for scenario in ['normal', 'stressed', 'fatigued', 'active', 'dehydrated']:
            runner = E2ETestRunner(scenario, use_voice=args.voice)
            runner.setup()
            runner.run_test_suite()
            print("\n")
    else:
        # Run single scenario
        runner = E2ETestRunner(args.scenario, use_voice=args.voice)
        runner.setup()
        runner.run_test_suite()


if __name__ == "__main__":
    main()
