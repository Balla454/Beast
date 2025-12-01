#!/usr/bin/env python3
"""
Database Manager for beast
==========================
Local SQLite database for storing health metrics, sessions, and interactions.

Compatible with the external PostgreSQL schema for easy upload/sync.
"""

import sqlite3
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading

logger = logging.getLogger('beast.Database')


class DatabaseManager:
    """
    Manages local SQLite database for beast.
    
    Stores:
    - Session data
    - Sensor metrics (EEG, PPG, etc.)
    - Calculated health metrics (cognitive load, fatigue, etc.)
    - Voice interaction logs
    - User preferences/baselines
    """
    
    def __init__(self, db_path: str = None):
        # Use environment variable or relative path as default
        if db_path is None:
            db_path = os.environ.get(
                'BEAST_DATABASE',
                str(Path(__file__).parent.parent / 'data' / 'beast_local.db')
            )
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self._lock = threading.Lock()
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"Database initialized: {self.db_path}")
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (thread-safe)"""
        if self.conn is None:
            self.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            self.conn.row_factory = sqlite3.Row
        return self.conn
        
    def _initialize_database(self):
        """Create database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                device_side TEXT,
                activity_type TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # EEG aggregate metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eeg_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP NOT NULL,
                delta_power REAL,
                theta_power REAL,
                alpha_power REAL,
                beta_power REAL,
                gamma_power REAL,
                alpha_theta_ratio REAL,
                beta_alpha_ratio REAL,
                engagement_index REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Physiological metrics (PPG, HRV, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS physiological_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP NOT NULL,
                heart_rate REAL,
                spo2 REAL,
                hrv_rmssd REAL,
                hrv_sdnn REAL,
                hrv_lf_hf_ratio REAL,
                core_temperature REAL,
                ambient_temperature REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Calculated health metrics (the 10 key metrics)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calculated_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP NOT NULL,
                cognitive_load REAL,
                tiredness REAL,
                fatigue REAL,
                attention_focus REAL,
                stress_index REAL,
                neurovascular_coupling REAL,
                metabolic_stress REAL,
                compensation_load REAL,
                fatigue_severity REAL,
                attention_capacity REAL,
                -- Zone classifications (1-4)
                cognitive_load_zone INTEGER,
                fatigue_zone INTEGER,
                stress_zone INTEGER,
                attention_zone INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Zone transitions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zone_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP NOT NULL,
                metric_name TEXT NOT NULL,
                from_zone INTEGER,
                to_zone INTEGER,
                metric_value REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Voice interactions log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP NOT NULL,
                user_query TEXT,
                response TEXT,
                response_time_ms INTEGER,
                health_context TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # User preferences and baselines
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                baseline_value REAL,
                baseline_std REAL,
                sample_count INTEGER,
                last_updated TIMESTAMP,
                UNIQUE(user_id, metric_name)
            )
        """)
        
        # Alerts history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP NOT NULL,
                alert_type TEXT,
                severity TEXT,
                metric_name TEXT,
                metric_value REAL,
                message TEXT,
                acknowledged INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_calculated_timestamp 
            ON calculated_metrics(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_physio_timestamp 
            ON physiological_metrics(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_interactions_timestamp 
            ON interactions(timestamp)
        """)
        
        conn.commit()
        logger.info("Database tables created/verified")
        
    # ==================== Session Management ====================
    
    def create_session(self, session_id: str, user_id: str = "default",
                      device_side: str = "left", activity_type: str = "general") -> str:
        """Create a new monitoring session"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sessions (session_id, user_id, start_time, device_side, activity_type)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, user_id, datetime.now(), device_side, activity_type))
            
            conn.commit()
            logger.info(f"Session created: {session_id}")
            
        return session_id
        
    def end_session(self, session_id: str, notes: str = None):
        """End a monitoring session"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE sessions SET end_time = ?, notes = ?
                WHERE session_id = ?
            """, (datetime.now(), notes, session_id))
            
            conn.commit()
            logger.info(f"Session ended: {session_id}")
            
    def get_active_session(self) -> Optional[str]:
        """Get the current active session ID"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_id FROM sessions
                WHERE end_time IS NULL
                ORDER BY start_time DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            return row['session_id'] if row else None

    # ==================== Sensor Queries ====================
    
    def get_sensor_data_at(self, sensor_type: str, minutes_ago: int = 0) -> Optional[Dict[str, Any]]:
        """
        Get sensor data at a specific time in the past.
        
        Args:
            sensor_type: Type of sensor (ppg, eeg, temp, ambient, etc.)
            minutes_ago: How many minutes in the past (0 = most recent)
            
        Returns:
            Dict with 'data', 'timestamp', and sensor-specific parsed fields
        """
        try:
            from datetime import timedelta
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Get the most recent timestamp for this sensor type
                cursor.execute(
                    """
                    SELECT MAX(timestamp) FROM sensor_data WHERE sensor_type=?
                    """,
                    (sensor_type,)
                )
                max_time_row = cursor.fetchone()
                if not max_time_row or not max_time_row[0]:
                    return None
                
                # Parse the max timestamp and subtract the requested minutes
                max_timestamp = datetime.fromisoformat(max_time_row[0])
                cutoff = max_timestamp - timedelta(minutes=minutes_ago)
                
                cursor.execute(
                    """
                    SELECT sensor_data, timestamp
                    FROM sensor_data
                    WHERE sensor_type=? AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (sensor_type, cutoff.isoformat())
                )
                row = cursor.fetchone()
                if not row:
                    return None
                    
                # Parse the JSON data
                data = json.loads(row['sensor_data']) if isinstance(row['sensor_data'], str) else row['sensor_data']
                
                result = {
                    'sensor_type': sensor_type,
                    'data': data,
                    'timestamp': row['timestamp']
                }
                
                # Add sensor-specific parsed fields
                if sensor_type == 'ppg' and isinstance(data, (list, tuple)):
                    result['heart_rate'] = data[0] if len(data) > 0 else None
                    result['spo2'] = data[1] if len(data) > 1 else None
                elif sensor_type == 'temp' and isinstance(data, (list, tuple)):
                    result['core_temp'] = data[0] if len(data) > 0 else None
                    result['skin_temp'] = data[1] if len(data) > 1 else None
                elif sensor_type == 'ambient' and isinstance(data, (list, tuple)):
                    result['ambient_temp'] = data[0] if len(data) > 0 else None
                    result['humidity'] = data[1] if len(data) > 1 else None
                    
                return result
                
        except Exception as e:
            logger.error(f"Failed sensor query for {sensor_type}: {e}", exc_info=True)
            return None
    
    def get_heart_rate_at(self, minutes_ago: int = 0) -> Optional[Dict[str, Any]]:
        """Return the latest heart rate at or before now - minutes_ago.
        Uses `sensor_data` table rows where sensor_type='ppg' and expects
        sensor_data json array format: [heart_rate, spo2].
        """
        try:
            from datetime import timedelta
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Get the most recent timestamp for PPG data
                cursor.execute(
                    """
                    SELECT MAX(timestamp) FROM sensor_data WHERE sensor_type='ppg'
                    """
                )
                max_time_row = cursor.fetchone()
                if not max_time_row or not max_time_row[0]:
                    return None
                
                # Parse the max timestamp and subtract the requested minutes
                max_timestamp = datetime.fromisoformat(max_time_row[0])
                cutoff = max_timestamp - timedelta(minutes=minutes_ago)
                
                cursor.execute(
                    """
                    SELECT sensor_data, timestamp
                    FROM sensor_data
                    WHERE sensor_type='ppg' AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (cutoff.isoformat(),)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                data = json.loads(row['sensor_data']) if isinstance(row['sensor_data'], str) else row['sensor_data']
                hr = None
                spo2 = None
                if isinstance(data, (list, tuple)) and len(data) >= 1:
                    hr = data[0]
                    spo2 = data[1] if len(data) > 1 else None
                return {
                    'heart_rate': hr,
                    'spo2': spo2,
                    'timestamp': row['timestamp']
                }
        except Exception as e:
            logger.error(f"Failed heart rate query: {e}", exc_info=True)
            return None
    
    def get_sensor_statistics(self, sensor_type: str, minutes: int = 60) -> Optional[Dict[str, Any]]:
        """
        Get statistical summary of sensor data over a time period.
        
        Args:
            sensor_type: Type of sensor (ppg, temp, etc.)
            minutes: Time window in minutes (default: 60)
            
        Returns:
            Dict with min, max, avg, count for the sensor data
        """
        try:
            from datetime import timedelta
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Get the most recent timestamp for this sensor type
                cursor.execute(
                    """
                    SELECT MAX(timestamp) FROM sensor_data WHERE sensor_type=?
                    """,
                    (sensor_type,)
                )
                max_time_row = cursor.fetchone()
                if not max_time_row or not max_time_row[0]:
                    return None
                
                # Parse the max timestamp and calculate the time window
                max_timestamp = datetime.fromisoformat(max_time_row[0])
                cutoff = max_timestamp - timedelta(minutes=minutes)
                
                cursor.execute(
                    """
                    SELECT sensor_data, timestamp
                    FROM sensor_data
                    WHERE sensor_type=? AND timestamp >= ?
                    ORDER BY timestamp ASC
                    """,
                    (sensor_type, cutoff.isoformat())
                )
                rows = cursor.fetchall()
                
                if not rows:
                    return None
                
                # Parse all data points
                values = []
                for row in rows:
                    data = json.loads(row['sensor_data']) if isinstance(row['sensor_data'], str) else row['sensor_data']
                    if isinstance(data, (list, tuple)) and len(data) > 0:
                        values.append(data)
                
                if not values:
                    return None
                
                result = {
                    'sensor_type': sensor_type,
                    'count': len(values),
                    'time_window_minutes': minutes,
                    'start_time': rows[0]['timestamp'],
                    'end_time': rows[-1]['timestamp']
                }
                
                # Calculate stats for each field in the data
                if sensor_type == 'ppg':
                    hrs = [v[0] for v in values if len(v) > 0]
                    spo2s = [v[1] for v in values if len(v) > 1]
                    if hrs:
                        result['heart_rate'] = {
                            'min': min(hrs),
                            'max': max(hrs),
                            'avg': sum(hrs) / len(hrs)
                        }
                    if spo2s:
                        result['spo2'] = {
                            'min': min(spo2s),
                            'max': max(spo2s),
                            'avg': sum(spo2s) / len(spo2s)
                        }
                elif sensor_type == 'temp':
                    temps = [v[0] for v in values if len(v) > 0]
                    if temps:
                        result['core_temp'] = {
                            'min': min(temps),
                            'max': max(temps),
                            'avg': sum(temps) / len(temps)
                        }
                
                return result
                
        except Exception as e:
            logger.error(f"Failed sensor statistics query: {e}", exc_info=True)
            return None
            
    # ==================== Metrics Storage ====================
    
    def insert_calculated_metrics(self, session_id: str, metrics: Dict[str, float],
                                  zones: Dict[str, int] = None):
        """Insert calculated health metrics"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            zones = zones or {}
            
            cursor.execute("""
                INSERT INTO calculated_metrics (
                    session_id, timestamp,
                    cognitive_load, tiredness, fatigue, attention_focus,
                    stress_index, neurovascular_coupling, metabolic_stress,
                    compensation_load, fatigue_severity, attention_capacity,
                    cognitive_load_zone, fatigue_zone, stress_zone, attention_zone
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, datetime.now(),
                metrics.get('cognitive_load'),
                metrics.get('tiredness'),
                metrics.get('fatigue'),
                metrics.get('attention_focus'),
                metrics.get('stress_index'),
                metrics.get('neurovascular_coupling'),
                metrics.get('metabolic_stress'),
                metrics.get('compensation_load'),
                metrics.get('fatigue_severity'),
                metrics.get('attention_capacity'),
                zones.get('cognitive_load_zone'),
                zones.get('fatigue_zone'),
                zones.get('stress_zone'),
                zones.get('attention_zone')
            ))
            
            conn.commit()
            
    def insert_physiological_metrics(self, session_id: str, metrics: Dict[str, float]):
        """Insert physiological sensor metrics"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO physiological_metrics (
                    session_id, timestamp,
                    heart_rate, spo2, hrv_rmssd, hrv_sdnn, hrv_lf_hf_ratio,
                    core_temperature, ambient_temperature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, datetime.now(),
                metrics.get('heart_rate'),
                metrics.get('spo2'),
                metrics.get('hrv_rmssd'),
                metrics.get('hrv_sdnn'),
                metrics.get('hrv_lf_hf_ratio'),
                metrics.get('core_temperature'),
                metrics.get('ambient_temperature')
            ))
            
            conn.commit()
            
    # ==================== Data Retrieval ====================
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the most recent health metrics"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # First try to get data from sensor_data table (synthetic playback)
            try:
                # Get latest PPG data for heart rate
                cursor.execute("""
                    SELECT sensor_data, timestamp FROM sensor_data
                    WHERE sensor_type = 'ppg'
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                ppg_row = cursor.fetchone()
                
                if ppg_row:
                    # Parse the sensor_data JSON to extract heart rate
                    import json
                    sensor_data = json.loads(ppg_row['sensor_data'])
                    # PPG data format: [hr_bpm, spo2_pct, ...]
                    heart_rate = sensor_data[0] if len(sensor_data) > 0 else None
                    spo2 = sensor_data[1] if len(sensor_data) > 1 else None
                    
                    return {
                        'heart_rate': heart_rate,
                        'spo2': spo2,
                        'timestamp': ppg_row['timestamp']
                    }
            except Exception as e:
                logger.debug(f"No synthetic sensor_data available: {e}")
            
            # Fallback: Get calculated metrics
            cursor.execute("""
                SELECT * FROM calculated_metrics
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            calc_row = cursor.fetchone()
            
            # Get physiological metrics
            cursor.execute("""
                SELECT * FROM physiological_metrics
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            physio_row = cursor.fetchone()
            
            if not calc_row and not physio_row:
                return None
                
            result = {}
            
            if calc_row:
                result.update({
                    'cognitive_load': calc_row['cognitive_load'],
                    'tiredness': calc_row['tiredness'],
                    'fatigue': calc_row['fatigue'],
                    'attention_focus': calc_row['attention_focus'],
                    'stress_index': calc_row['stress_index'],
                    'timestamp': calc_row['timestamp']
                })
                
            if physio_row:
                result.update({
                    'heart_rate': physio_row['heart_rate'],
                    'spo2': physio_row['spo2'],
                    'hrv_rmssd': physio_row['hrv_rmssd'],
                    'core_temperature': physio_row['core_temperature']
                })
                
            return result
            
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for the past N hours"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT c.*, p.heart_rate, p.spo2
                FROM calculated_metrics c
                LEFT JOIN physiological_metrics p 
                    ON c.session_id = p.session_id 
                    AND ABS(strftime('%s', c.timestamp) - strftime('%s', p.timestamp)) < 2
                WHERE c.timestamp > datetime('now', ?)
                ORDER BY c.timestamp DESC
            """, (f'-{hours} hours',))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
    # ==================== Interactions ====================
    
    def log_interaction(self, query: str, response: str, 
                       response_time_ms: int = None,
                       session_id: str = None):
        """Log a voice interaction"""
        logger.debug("Logging interaction to database...")
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get health context
            latest = self.get_latest_metrics()
            health_context = json.dumps(latest) if latest else None
            
            # Get active session if not provided
            if not session_id:
                session_id = self.get_active_session()
                
            cursor.execute("""
                INSERT INTO interactions (
                    session_id, timestamp, user_query, response,
                    response_time_ms, health_context
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id, datetime.now(), query, response,
                response_time_ms, health_context
            ))
            
            conn.commit()
            logger.debug("Interaction logged successfully")
            
    def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent voice interactions"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM interactions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
    # ==================== Alerts ====================
    
    def add_alert(self, alert_type: str, severity: str, 
                 metric_name: str, metric_value: float, message: str,
                 session_id: str = None):
        """Add a health alert"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if not session_id:
                session_id = self.get_active_session()
                
            cursor.execute("""
                INSERT INTO alerts (
                    session_id, timestamp, alert_type, severity,
                    metric_name, metric_value, message
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, datetime.now(), alert_type, severity,
                metric_name, metric_value, message
            ))
            
            conn.commit()
            logger.warning(f"Alert added: {severity} - {message}")
            
    def get_unacknowledged_alerts(self) -> List[Dict[str, Any]]:
        """Get unacknowledged alerts"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM alerts
                WHERE acknowledged = 0
                ORDER BY timestamp DESC
            """)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
    # ==================== User Baselines ====================
    
    def update_baseline(self, user_id: str, metric_name: str, 
                       value: float, std: float = None, count: int = 1):
        """Update user baseline for a metric"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO user_baselines (
                    user_id, metric_name, baseline_value, baseline_std,
                    sample_count, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, metric_name) DO UPDATE SET
                    baseline_value = ?,
                    baseline_std = ?,
                    sample_count = sample_count + ?,
                    last_updated = ?
            """, (
                user_id, metric_name, value, std, count, datetime.now(),
                value, std, count, datetime.now()
            ))
            
            conn.commit()
            
    def get_baselines(self, user_id: str) -> Dict[str, Dict[str, float]]:
        """Get all baselines for a user"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT metric_name, baseline_value, baseline_std, sample_count
                FROM user_baselines
                WHERE user_id = ?
            """, (user_id,))
            
            rows = cursor.fetchall()
            return {
                row['metric_name']: {
                    'value': row['baseline_value'],
                    'std': row['baseline_std'],
                    'count': row['sample_count']
                }
                for row in rows
            }
            
    # ==================== Export ====================
    
    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """Export all data for a session (for sync/backup)"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            session = dict(cursor.fetchone()) if cursor.fetchone() else {}
            
            # Get calculated metrics
            cursor.execute("""
                SELECT * FROM calculated_metrics WHERE session_id = ?
            """, (session_id,))
            calculated = [dict(row) for row in cursor.fetchall()]
            
            # Get physiological metrics
            cursor.execute("""
                SELECT * FROM physiological_metrics WHERE session_id = ?
            """, (session_id,))
            physiological = [dict(row) for row in cursor.fetchall()]
            
            # Get interactions
            cursor.execute("""
                SELECT * FROM interactions WHERE session_id = ?
            """, (session_id,))
            interactions = [dict(row) for row in cursor.fetchall()]
            
            # Get alerts
            cursor.execute("""
                SELECT * FROM alerts WHERE session_id = ?
            """, (session_id,))
            alerts = [dict(row) for row in cursor.fetchall()]
            
            return {
                'session': session,
                'calculated_metrics': calculated,
                'physiological_metrics': physiological,
                'interactions': interactions,
                'alerts': alerts
            }
            
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing Database Manager...")
    
    db = DatabaseManager("/tmp/test_beast.db")
    
    # Create session
    session_id = db.create_session("test_001", "user_001")
    
    # Insert metrics
    db.insert_calculated_metrics(session_id, {
        'cognitive_load': 45.5,
        'fatigue': 32.0,
        'stress_index': 28.5,
        'attention_focus': 72.0
    })
    
    db.insert_physiological_metrics(session_id, {
        'heart_rate': 75,
        'spo2': 98,
        'hrv_rmssd': 45.2
    })
    
    # Log interaction
    db.log_interaction("How is my heart rate?", "Your heart rate is 75 BPM.", session_id=session_id)
    
    # Get latest
    latest = db.get_latest_metrics()
    print(f"\nLatest metrics: {latest}")
    
    # Get interactions
    interactions = db.get_recent_interactions(5)
    print(f"\nRecent interactions: {len(interactions)}")
    
    db.close()
    print("\nTest complete!")
