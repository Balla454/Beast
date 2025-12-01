#!/usr/bin/env python3
"""
Health RAG System for BeAST
===========================
Retrieval-Augmented Generation for health queries.

Combines:
- User health data from PostgreSQL database (Antonio's schema)
- Knowledge base with medical/health information
- LLM for response generation

Database Schema (from Database for Antonio/beast_schema.sql):
- raw.earpiece_data: EEG raw time series
- raw.eeg_band: EEG band powers (alpha, beta, gamma, delta, theta)
- raw.cognitive_metrics: cognitive_load, alertness, fatigue
- raw.ppg_reading: hr_bpm, spo2_pct
- raw.hydration_reading: hydration_pct
- raw.temp_reading: core_temp_c
- raw.bioz_reading: IMU accel/gyro data
- raw.environment: ambient_temp, humidity_pct, heat_index_c
- agg.zone_durations: aggregated zone time per metric
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger('BeAST.RAG')

# Optional imports with availability flags
TRANSFORMERS_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False
FAISS_AVAILABLE = False
PSYCOPG2_AVAILABLE = False
OLLAMA_AVAILABLE = False

# Force offline mode for HuggingFace
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Check if Ollama is running locally (no network required)
try:
    import socket
    # Check if Ollama is listening on localhost:11434
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('127.0.0.1', 11434))
    sock.close()
    if result == 0:
        OLLAMA_AVAILABLE = True
        logger.info("Ollama available (local)")
except Exception:
    pass

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers available")
except ImportError:
    logger.warning("transformers not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence Transformers available")
except ImportError:
    logger.warning("sentence-transformers not available")

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
    logger.info("FAISS available")
except ImportError:
    logger.warning("faiss not available")

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
    logger.info("PostgreSQL (psycopg2) available")
except ImportError:
    logger.warning("psycopg2 not available - install with: pip install psycopg2-binary")


class HealthRAG:
    """
    Health-focused RAG system for BeAST.
    
    Retrieves user health data from PostgreSQL database (Antonio's schema)
    and generates personalized responses.
    
    Usage:
        rag = HealthRAG(config)
        response = rag.query("How is my heart rate?")
    """
    
    def __init__(self, config: dict, database=None):
        """
        Initialize Health RAG system.
        
        Args:
            config: RAG configuration dict with database connection info
            database: Optional DatabaseManager instance (fallback)
        """
        self.config = config
        self.database = database  # Fallback SQLite database
        self.pg_conn = None  # PostgreSQL connection
        
        # Model components
        self.embedder = None
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        # Knowledge base
        self.knowledge_index = None
        self.knowledge_docs = []
        
        # Response cache
        self._cache = {}
        self._cache_max = 50
        
        # Health data context
        self.user_context = {}
        
        # Device/session filtering
        self.device_id = config.get('device_id')
        self.session_id = config.get('session_id')
        
        # Initialize
        self._initialize()
        
    def _initialize(self):
        """Initialize RAG components"""
        logger.info("Initializing Health RAG System...")
        
        # Connect to PostgreSQL
        self._connect_postgres()
        
        # Load embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._load_embedder()
            
        # Load LLM
        if TRANSFORMERS_AVAILABLE:
            self._load_llm()
            
        # Load knowledge base
        self._load_knowledge_base()
        
        logger.info("Health RAG System initialized")
        
    def _connect_postgres(self):
        """Connect to PostgreSQL database"""
        if not PSYCOPG2_AVAILABLE:
            logger.warning("psycopg2 not available, will use fallback")
            return
            
        db_config = self.config.get('database', {})
        
        # Get password from env var first, then config
        db_password = os.environ.get('BEAST_DB_PASSWORD', db_config.get('password', ''))
        
        try:
            self.pg_conn = psycopg2.connect(
                host=os.environ.get('BEAST_DB_HOST', db_config.get('host', 'localhost')),
                port=int(os.environ.get('BEAST_DB_PORT', db_config.get('port', 5432))),
                database=os.environ.get('BEAST_DB_NAME', db_config.get('database', 'beast')),
                user=os.environ.get('BEAST_DB_USER', db_config.get('user', 'beast')),
                password=db_password,
                connect_timeout=5
            )
            self.pg_conn.autocommit = True
            logger.info(f"Connected to PostgreSQL: {db_config.get('host', 'localhost')}/{db_config.get('database', 'beast')}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self.pg_conn = None
        
    def _load_embedder(self):
        """Load sentence embedding model (offline mode)"""
        model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        
        try:
            logger.info(f"Loading embedding model: {model_name} (local/offline)")
            # Load from cache only - don't download
            self.embedder = SentenceTransformer(model_name, device='cpu')
            logger.info("Embedding model loaded (offline)")
        except Exception as e:
            logger.error(f"Failed to load embedder: {e}")
            logger.error("Make sure the model is cached. Run once with internet to download.")
            
    def _load_llm(self):
        """Load language model for generation - prefer Ollama with gemma2:2b (offline)"""
        llm_config = self.config.get('llm', {})
        
        # Prefer Ollama with gemma2:2b (runs locally, no internet needed)
        if OLLAMA_AVAILABLE:
            self.ollama_model = llm_config.get('ollama_model', 'gemma2:2b')
            self.ollama_url = llm_config.get('ollama_url', 'http://localhost:11434')
            
            # Just use Ollama - models are already pulled locally
            # No need to verify via HTTP, the socket check confirmed it's running
            logger.info(f"Using Ollama with model: {self.ollama_model} (local/offline)")
            return
                
        # Fallback to local model path
        model_path = llm_config.get('model_path', '')
        
        if model_path and Path(model_path).exists():
            try:
                logger.info(f"Loading local LLM from: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Local LLM loaded")
                return
            except Exception as e:
                logger.error(f"Failed to load local LLM: {e}")
                
        # Fall back to rule-based responses (no model download)
        logger.warning("No LLM available - using rule-based responses")
        logger.info("To enable LLM: start Ollama with 'ollama serve' and pull a model")
            
    def _load_knowledge_base(self):
        """Load health knowledge base"""
        # Built-in health knowledge
        self.knowledge_docs = [
            # Heart rate knowledge
            {
                "topic": "heart_rate",
                "content": "Normal resting heart rate is 60-100 BPM. Below 60 may indicate bradycardia, above 100 may indicate tachycardia. Athletes may have lower resting rates. Heart rate varies with activity, stress, and health."
            },
            {
                "topic": "heart_rate_zones",
                "content": "Heart rate zones: Zone 1 (50-60% max) recovery, Zone 2 (60-70%) fat burning, Zone 3 (70-80%) aerobic, Zone 4 (80-90%) anaerobic, Zone 5 (90-100%) maximum effort."
            },
            # SpO2 knowledge
            {
                "topic": "spo2",
                "content": "Normal blood oxygen (SpO2) is 95-100%. Below 95% may indicate hypoxemia. Below 90% is concerning and needs attention. Altitude, activity, and respiratory conditions affect SpO2."
            },
            # Cognitive load
            {
                "topic": "cognitive_load",
                "content": "Cognitive load measures mental workload. 0-40 is optimal, 40-60 is elevated, 60-80 is high (consider reducing tasks), 80-100 indicates overload and performance degradation."
            },
            # Fatigue
            {
                "topic": "fatigue",
                "content": "Fatigue levels: 0-30 is fresh/optimal, 30-50 shows mild fatigue, 50-70 is moderate fatigue needing rest, 70-100 is severe fatigue requiring immediate rest."
            },
            # Stress
            {
                "topic": "stress",
                "content": "Stress index: 0-30 is calm/optimal, 30-50 is manageable stress, 50-75 is high stress where intervention helps, 75-100 is extreme stress requiring immediate action."
            },
            # Attention
            {
                "topic": "attention",
                "content": "Attention focus: 70-100 is excellent attention, 50-70 is adequate with slight lapses, 30-50 is poor with frequent lapses, 0-30 indicates severe attention deficit."
            },
            # Hydration
            {
                "topic": "hydration",
                "content": "Proper hydration is crucial for performance. Bioimpedance can indicate hydration status. Signs of dehydration include increased heart rate, decreased performance, and fatigue."
            },
            # Temperature
            {
                "topic": "core_temperature",
                "content": "Normal core temperature is 97-99°F (36.1-37.2°C). Above 100.4°F indicates fever. High ambient temperature combined with activity can cause heat stress."
            },
            # Sleep and recovery
            {
                "topic": "recovery",
                "content": "Good recovery requires adequate sleep (7-9 hours), proper hydration, nutrition, and stress management. HRV (heart rate variability) can indicate recovery status."
            }
        ]
        
        # Build index if FAISS available
        if FAISS_AVAILABLE and self.embedder:
            self._build_knowledge_index()
            
    def _build_knowledge_index(self):
        """Build FAISS index for knowledge retrieval"""
        try:
            # Embed all documents
            texts = [doc['content'] for doc in self.knowledge_docs]
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.knowledge_index = faiss.IndexFlatIP(dimension)  # Inner product
            
            # Normalize and add
            faiss.normalize_L2(embeddings)
            self.knowledge_index.add(embeddings)
            
            logger.info(f"Knowledge index built with {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build knowledge index: {e}")
            
    def _get_user_health_context(self) -> Dict[str, Any]:
        """Get current user health data from PostgreSQL database"""
        
        # Try PostgreSQL first
        if self.pg_conn:
            try:
                return self._query_postgres_health_data()
            except Exception as e:
                logger.error(f"PostgreSQL query failed: {e}")
                # Try to reconnect
                self._connect_postgres()
                
        # Fallback to SQLite database
        if self.database:
            try:
                latest = self.database.get_latest_metrics()
                if latest:
                    self.user_context = {
                        'heart_rate': latest.get('heart_rate'),
                        'spo2': latest.get('spo2'),
                        'cognitive_load': latest.get('cognitive_load'),
                        'fatigue': latest.get('fatigue'),
                        'stress': latest.get('stress_index'),
                        'attention': latest.get('attention_focus'),
                        'timestamp': latest.get('timestamp')
                    }
                return self.user_context
            except Exception as e:
                logger.error(f"SQLite fallback failed: {e}")
                
        return {}
        
    def _query_postgres_health_data(self) -> Dict[str, Any]:
        """Query health data from PostgreSQL (Antonio's schema)"""
        context = {}
        
        with self.pg_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Build WHERE clause for device/session filtering
            where_clause = "WHERE ts > NOW() - INTERVAL '5 minutes'"
            params = []
            
            if self.device_id:
                where_clause += " AND device_id = %s"
                params.append(self.device_id)
            if self.session_id:
                where_clause += " AND session_id = %s"
                params.append(self.session_id)
                
            # Query PPG (heart rate, SpO2)
            cur.execute(f"""
                SELECT hr_bpm, spo2_pct, ts 
                FROM raw.ppg_reading 
                {where_clause}
                ORDER BY ts DESC LIMIT 1
            """, params)
            row = cur.fetchone()
            if row:
                context['heart_rate'] = row['hr_bpm']
                context['spo2'] = row['spo2_pct']
                context['ppg_timestamp'] = row['ts']
                
            # Query cognitive metrics
            cur.execute(f"""
                SELECT cognitive_load, alertness, fatigue, ts
                FROM raw.cognitive_metrics
                {where_clause}
                ORDER BY ts DESC LIMIT 1
            """, params)
            row = cur.fetchone()
            if row:
                context['cognitive_load'] = row['cognitive_load']
                context['alertness'] = row['alertness']
                context['fatigue'] = row['fatigue']
                context['cognitive_timestamp'] = row['ts']
                
            # Query EEG bands
            cur.execute(f"""
                SELECT alpha, beta, gamma, delta, theta, ts
                FROM raw.eeg_band
                {where_clause}
                ORDER BY ts DESC LIMIT 1
            """, params)
            row = cur.fetchone()
            if row:
                context['eeg_alpha'] = row['alpha']
                context['eeg_beta'] = row['beta']
                context['eeg_gamma'] = row['gamma']
                context['eeg_delta'] = row['delta']
                context['eeg_theta'] = row['theta']
                context['eeg_timestamp'] = row['ts']
                
            # Query hydration
            cur.execute(f"""
                SELECT hydration_pct, ts
                FROM raw.hydration_reading
                {where_clause}
                ORDER BY ts DESC LIMIT 1
            """, params)
            row = cur.fetchone()
            if row:
                context['hydration'] = row['hydration_pct']
                context['hydration_timestamp'] = row['ts']
                
            # Query temperature
            cur.execute(f"""
                SELECT core_temp_c, ts
                FROM raw.temp_reading
                {where_clause}
                ORDER BY ts DESC LIMIT 1
            """, params)
            row = cur.fetchone()
            if row:
                context['core_temp'] = row['core_temp_c']
                context['temp_timestamp'] = row['ts']
                
            # Query IMU (for motion/activity)
            cur.execute(f"""
                SELECT accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, ts
                FROM raw.bioz_reading
                {where_clause}
                ORDER BY ts DESC LIMIT 1
            """, params)
            row = cur.fetchone()
            if row:
                # Calculate motion magnitude
                import math
                accel_mag = math.sqrt(row['accel_x']**2 + row['accel_y']**2 + row['accel_z']**2)
                context['motion_magnitude'] = accel_mag
                context['accel'] = [row['accel_x'], row['accel_y'], row['accel_z']]
                context['gyro'] = [row['gyro_x'], row['gyro_y'], row['gyro_z']]
                context['imu_timestamp'] = row['ts']
                
            # Query environment
            cur.execute(f"""
                SELECT ambient_temp, humidity_pct, heat_index_c, ts
                FROM raw.environment
                {where_clause.replace('device_id', 'device_id')}
                ORDER BY ts DESC LIMIT 1
            """, params[:1] if self.device_id else [])  # environment only has device_id
            row = cur.fetchone()
            if row:
                context['ambient_temp'] = row['ambient_temp']
                context['humidity'] = row['humidity_pct']
                context['heat_index'] = row['heat_index_c']
                context['env_timestamp'] = row['ts']
                
            # Query zone durations (aggregates) for current session
            if self.session_id:
                cur.execute("""
                    SELECT metric, zone, duration_s
                    FROM agg.zone_durations
                    WHERE session_id = %s
                    ORDER BY computed_at DESC
                """, [self.session_id])
                zones = cur.fetchall()
                if zones:
                    context['zone_durations'] = {}
                    for row in zones:
                        metric = row['metric']
                        if metric not in context['zone_durations']:
                            context['zone_durations'][metric] = {}
                        context['zone_durations'][metric][row['zone']] = row['duration_s']
                        
            # Get historical averages (last hour)
            cur.execute(f"""
                SELECT 
                    AVG(hr_bpm) as avg_hr,
                    MIN(hr_bpm) as min_hr,
                    MAX(hr_bpm) as max_hr,
                    AVG(spo2_pct) as avg_spo2
                FROM raw.ppg_reading
                WHERE ts > NOW() - INTERVAL '1 hour'
                {' AND device_id = %s' if self.device_id else ''}
            """, [self.device_id] if self.device_id else [])
            row = cur.fetchone()
            if row and row['avg_hr']:
                context['hr_avg_1h'] = row['avg_hr']
                context['hr_min_1h'] = row['min_hr']
                context['hr_max_1h'] = row['max_hr']
                context['spo2_avg_1h'] = row['avg_spo2']
                
        self.user_context = context
        return context
    
    def get_session_summary(self, session_id: str = None) -> Dict[str, Any]:
        """Get comprehensive summary for a session"""
        session_id = session_id or self.session_id
        if not session_id or not self.pg_conn:
            return {}
            
        summary = {}
        
        with self.pg_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Session duration and data points
            cur.execute("""
                SELECT 
                    MIN(ts) as start_time,
                    MAX(ts) as end_time,
                    COUNT(*) as data_points
                FROM raw.ppg_reading
                WHERE session_id = %s
            """, [session_id])
            row = cur.fetchone()
            if row and row['start_time']:
                summary['start_time'] = row['start_time']
                summary['end_time'] = row['end_time']
                summary['duration_minutes'] = (row['end_time'] - row['start_time']).total_seconds() / 60
                summary['ppg_data_points'] = row['data_points']
                
            # HR statistics
            cur.execute("""
                SELECT 
                    AVG(hr_bpm) as avg_hr,
                    MIN(hr_bpm) as min_hr,
                    MAX(hr_bpm) as max_hr,
                    STDDEV(hr_bpm) as std_hr
                FROM raw.ppg_reading
                WHERE session_id = %s
            """, [session_id])
            row = cur.fetchone()
            if row:
                summary['hr_avg'] = row['avg_hr']
                summary['hr_min'] = row['min_hr']
                summary['hr_max'] = row['max_hr']
                summary['hr_variability'] = row['std_hr']
                
            # Cognitive metrics averages
            cur.execute("""
                SELECT 
                    AVG(cognitive_load) as avg_cognitive,
                    AVG(alertness) as avg_alertness,
                    AVG(fatigue) as avg_fatigue
                FROM raw.cognitive_metrics
                WHERE session_id = %s
            """, [session_id])
            row = cur.fetchone()
            if row:
                summary['cognitive_load_avg'] = row['avg_cognitive']
                summary['alertness_avg'] = row['avg_alertness']
                summary['fatigue_avg'] = row['avg_fatigue']
                
            # Zone time breakdown
            cur.execute("""
                SELECT metric, zone, duration_s
                FROM agg.zone_durations
                WHERE session_id = %s
            """, [session_id])
            zones = cur.fetchall()
            if zones:
                summary['zone_breakdown'] = {}
                for row in zones:
                    metric = row['metric']
                    if metric not in summary['zone_breakdown']:
                        summary['zone_breakdown'][metric] = {}
                    summary['zone_breakdown'][metric][row['zone']] = row['duration_s']
                    
        return summary
            
    def _retrieve_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant knowledge documents"""
        if not self.knowledge_index or not self.embedder:
            # Fall back to keyword matching
            return self._keyword_match(query, top_k)
            
        try:
            # Embed query
            query_embedding = self.embedder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.knowledge_index.search(query_embedding, top_k)
            
            # Get documents
            results = []
            for idx in indices[0]:
                if idx < len(self.knowledge_docs):
                    results.append(self.knowledge_docs[idx]['content'])
                    
            return results
            
        except Exception as e:
            logger.error(f"Knowledge retrieval error: {e}")
            return []
            
    def _keyword_match(self, query: str, top_k: int = 3) -> List[str]:
        """Simple keyword matching fallback"""
        query_lower = query.lower()
        
        # Keywords to topics mapping
        keyword_map = {
            'heart': ['heart_rate', 'heart_rate_zones'],
            'pulse': ['heart_rate'],
            'bpm': ['heart_rate'],
            'oxygen': ['spo2'],
            'spo2': ['spo2'],
            'breathing': ['spo2'],
            'cognitive': ['cognitive_load'],
            'mental': ['cognitive_load'],
            'thinking': ['cognitive_load'],
            'tired': ['fatigue'],
            'fatigue': ['fatigue'],
            'exhausted': ['fatigue'],
            'stress': ['stress'],
            'anxious': ['stress'],
            'attention': ['attention'],
            'focus': ['attention'],
            'water': ['hydration'],
            'hydration': ['hydration'],
            'thirsty': ['hydration'],
            'temperature': ['core_temperature'],
            'fever': ['core_temperature'],
            'hot': ['core_temperature'],
            'sleep': ['recovery'],
            'recovery': ['recovery'],
            'rest': ['recovery', 'fatigue']
        }
        
        # Find matching topics
        matched_topics = set()
        for keyword, topics in keyword_map.items():
            if keyword in query_lower:
                matched_topics.update(topics)
                
        # Get documents for matched topics
        results = []
        for doc in self.knowledge_docs:
            if doc['topic'] in matched_topics:
                results.append(doc['content'])
                if len(results) >= top_k:
                    break
                    
        return results
        
    def _generate_response(self, query: str, context: str, health_data: Dict) -> str:
        """Generate response using LLM or rules"""
        
        # Check cache
        cache_key = hashlib.md5(f"{query}_{str(health_data)}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Build prompt
        prompt = self._build_prompt(query, context, health_data)
        
        # Try LLM generation (prefer Ollama)
        if OLLAMA_AVAILABLE and hasattr(self, 'ollama_model'):
            response = self._generate_with_ollama(prompt)
        elif self.model and self.tokenizer:
            response = self._generate_with_llm(prompt)
        elif self.generator:
            response = self._generate_with_pipeline(prompt)
        else:
            response = self._generate_rule_based(query, context, health_data)
            
        # Cache response
        if len(self._cache) >= self._cache_max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = response
        
        return response
        
    def _build_prompt(self, query: str, context: str, health_data: Dict) -> str:
        """Build prompt for LLM"""
        # Format health data
        health_str = ""
        if health_data:
            health_items = []
            if health_data.get('heart_rate'):
                health_items.append(f"Heart rate: {health_data['heart_rate']:.0f} BPM")
            if health_data.get('spo2'):
                health_items.append(f"SpO2: {health_data['spo2']:.0f}%")
            if health_data.get('cognitive_load'):
                health_items.append(f"Cognitive load: {health_data['cognitive_load']:.0f}/100")
            if health_data.get('fatigue'):
                health_items.append(f"Fatigue: {health_data['fatigue']:.0f}/100")
            if health_data.get('stress'):
                health_items.append(f"Stress: {health_data['stress']:.0f}/100")
            if health_data.get('attention'):
                health_items.append(f"Attention: {health_data['attention']:.0f}/100")
                
            if health_items:
                health_str = "Current readings: " + ", ".join(health_items) + ".\n"
                
        prompt = f"""You are BeAST, a health monitoring assistant. Be concise and helpful. Do not use emojis.

{health_str}
Knowledge: {context[:500]}

User: {query}
BeAST:"""
        
        return prompt
        
    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate response using Ollama API with gemma2:2b"""
        try:
            import requests
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 100,  # Max tokens
                        "stop": ["\n\n", "User:", "Human:"]
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '').strip()
                
                # Clean up response
                if text:
                    # Remove any trailing incomplete sentences
                    if text and text[-1] not in '.!?':
                        last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                        if last_punct > 0:
                            text = text[:last_punct + 1]
                    return text
                    
            logger.warning(f"Ollama returned status {response.status_code}")
            return self._generate_rule_based("", "", {})
            
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return self._generate_rule_based("", "", {})
        
    def _generate_with_llm(self, prompt: str) -> str:
        """Generate with local LLM"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=800,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=80,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response if response else self._generate_rule_based("", "", {})
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._generate_rule_based("", "", {})
            
    def _generate_with_pipeline(self, prompt: str) -> str:
        """Generate with transformers pipeline"""
        try:
            result = self.generator(
                prompt,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256
            )
            
            response = result[0]['generated_text']
            # Extract just the response part
            if "BeAST:" in response:
                response = response.split("BeAST:")[-1].strip()
                
            return response
            
        except Exception as e:
            logger.error(f"Pipeline generation error: {e}")
            return self._generate_rule_based("", "", {})
            
    def _generate_rule_based(self, query: str, context: str, health_data: Dict) -> str:
        """Generate response using rules (fallback)"""
        query_lower = query.lower()
        
        # =====================================================================
        # HEART RATE / CARDIOVASCULAR
        # =====================================================================
        if any(w in query_lower for w in ['heart rate', 'heartrate', 'pulse', 'bpm', 'hr ', 'my hr', 'heart beat', 'beating']):
            return self._handle_heart_rate_query(query_lower, health_data)
            
        # HRV queries
        if any(w in query_lower for w in ['hrv', 'heart rate variability', 'variability']):
            return self._handle_hrv_query(health_data)
            
        # =====================================================================
        # BLOOD PRESSURE (not directly measured, but handle gracefully)
        # =====================================================================
        if any(w in query_lower for w in ['blood pressure', 'bp ', 'systolic', 'diastolic']):
            return "Blood pressure is not directly measured by BeAST sensors. However, your heart rate and other metrics can provide related insights. Consider using a dedicated blood pressure monitor."
            
        # =====================================================================
        # BODY TEMPERATURE
        # =====================================================================
        if any(w in query_lower for w in ['temperature', 'temp', 'fever', 'core temp', 'skin temp', 'body temp']):
            return self._handle_temperature_query(query_lower, health_data)
            
        # =====================================================================
        # RESPIRATORY / OXYGEN
        # =====================================================================
        if any(w in query_lower for w in ['oxygen', 'spo2', 'o2', 'breathing', 'respiratory', 'breath', 'saturation']):
            return self._handle_oxygen_query(query_lower, health_data)
            
        # =====================================================================
        # STRESS
        # =====================================================================
        if any(w in query_lower for w in ['stress', 'stressed', 'anxious', 'anxiety', 'calm', 'relaxed']):
            return self._handle_stress_query(query_lower, health_data)
            
        # =====================================================================
        # COGNITIVE LOAD
        # =====================================================================
        if any(w in query_lower for w in ['cognitive', 'mental load', 'mental burden', 'cognitive load', 'cognitive performance', 'thinking']):
            return self._handle_cognitive_query(query_lower, health_data)
            
        # =====================================================================
        # ALERTNESS & FATIGUE
        # =====================================================================
        if any(w in query_lower for w in ['alert', 'awake', 'vigilance', 'vigilant', 'readiness', 'ready']):
            return self._handle_alertness_query(query_lower, health_data)
            
        if any(w in query_lower for w in ['tired', 'fatigue', 'exhausted', 'drowsy', 'sleepy', 'energy', 'fatigue score', 'fatigue level']):
            return self._handle_fatigue_query(query_lower, health_data)
            
        # =====================================================================
        # HYDRATION & METABOLIC
        # =====================================================================
        if any(w in query_lower for w in ['hydrat', 'water', 'thirst', 'dehydrat', 'sweat', 'electrolyte']):
            return self._handle_hydration_query(query_lower, health_data)
            
        if any(w in query_lower for w in ['metabolic', 'calorie', 'calories', 'burned']):
            return self._handle_metabolic_query(health_data)
            
        # =====================================================================
        # ACTIVITY & PERFORMANCE
        # =====================================================================
        if any(w in query_lower for w in ['activity', 'active', 'steps', 'exertion', 'motion', 'movement', 'moving', 'still']):
            return self._handle_activity_query(query_lower, health_data)
            
        # =====================================================================
        # TREND & TIME-BASED QUERIES
        # =====================================================================
        if any(w in query_lower for w in ['ago', 'minutes ago', 'hour ago', 'average', 'highest', 'lowest', 'last hour', 'today', 'history', 'trend', 'changed', 'over the']):
            return self._handle_trend_query(query_lower, health_data)
            
        # =====================================================================
        # THRESHOLD / ALERT STATUS QUERIES
        # =====================================================================
        if any(w in query_lower for w in ['abnormal', 'normal', 'safe', 'attention', 'health status', 'overall', 'everything', 'any metrics', 'all vitals', 'vital signs', 'vitals']):
            return self._handle_threshold_query(query_lower, health_data)
            
        # =====================================================================
        # YES/NO THRESHOLD ASSESSMENTS
        # =====================================================================
        if any(w in query_lower for w in ['am i dehydrated', 'am i too', 'do i need', 'should i', 'am i ready', 'am i fit', 'am i over']):
            return self._handle_yesno_query(query_lower, health_data)
            
        # =====================================================================
        # DEVICE / SYSTEM STATUS
        # =====================================================================
        if any(w in query_lower for w in ['battery', 'power', 'sensor', 'connected', 'last reading', 'signal', 'working', 'device']):
            return self._handle_device_query(query_lower, health_data)
            
        # =====================================================================
        # EEG / BRAIN WAVE QUERIES
        # =====================================================================
        if any(w in query_lower for w in ['eeg', 'brain wave', 'alpha', 'beta', 'theta', 'gamma', 'delta', 'wave']):
            return self._handle_eeg_query(health_data)
            
        # =====================================================================
        # ENVIRONMENT QUERIES
        # =====================================================================
        if any(w in query_lower for w in ['environment', 'humidity', 'ambient', 'weather', 'room', 'heat index']):
            return self._handle_environment_query(health_data)
            
        # =====================================================================
        # ZONE DURATION QUERIES
        # =====================================================================
        if any(w in query_lower for w in ['zone', 'time in', 'duration', 'how long']):
            return self._handle_zone_query(health_data)
            
        # =====================================================================
        # STATUS / SUMMARY QUERIES
        # =====================================================================
        if any(w in query_lower for w in ['status', 'how am i', 'overview', 'summary', 'check']):
            return self._generate_status_summary(health_data)
            
        # Default response
        if context:
            return f"Based on health knowledge: {context[:200]}"
        return "I'm here to help with your health questions. You can ask about heart rate, oxygen, stress, fatigue, cognitive load, alertness, hydration, temperature, activity, or your overall status."

    # =========================================================================
    # HANDLER METHODS FOR EACH QUERY TYPE
    # =========================================================================
    
    def _handle_heart_rate_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle heart rate related queries"""
        hr = health_data.get('heart_rate')
        
        if hr is None:
            return "I don't have current heart rate data. Make sure sensors are connected."
        
        # Basic heart rate
        response = f"Your heart rate is {hr:.0f} BPM"
        
        # Determine status
        if hr < 60:
            status = "below normal (bradycardia range)"
            assessment = "This could indicate good cardiovascular fitness or bradycardia."
        elif hr > 100:
            status = "elevated (tachycardia range)"
            assessment = "This could be from physical activity, stress, caffeine, or other factors."
        else:
            status = "normal"
            assessment = "This is within the healthy resting range of 60-100 BPM."
            
        # Check for specific question types
        if 'resting' in query_lower:
            response = f"Your current heart rate is {hr:.0f} BPM. For accurate resting heart rate, measure after sitting quietly for 5+ minutes."
        elif 'normal' in query_lower or 'elevated' in query_lower or 'low' in query_lower:
            response = f"Your heart rate is {hr:.0f} BPM, which is {status}. {assessment}"
        elif 'average' in query_lower or 'today' in query_lower or 'hour' in query_lower:
            if health_data.get('hr_avg_1h'):
                response = f"Your average heart rate over the last hour is {health_data['hr_avg_1h']:.0f} BPM (range: {health_data.get('hr_min_1h', 0):.0f}-{health_data.get('hr_max_1h', 0):.0f}). Current: {hr:.0f} BPM."
            else:
                response = f"Your current heart rate is {hr:.0f} BPM. Historical data is not available."
        elif 'fast' in query_lower or 'beating' in query_lower:
            response = f"Your heart is beating at {hr:.0f} BPM, which is {status}."
        else:
            response += f", which is {status}."
            if health_data.get('hr_avg_1h'):
                response += f" Average over last hour: {health_data['hr_avg_1h']:.0f} BPM."
                
        return response
        
    def _handle_hrv_query(self, health_data: Dict) -> str:
        """Handle HRV queries"""
        # HRV can be derived from HR variability in the data
        hr_std = health_data.get('hr_variability') or health_data.get('hrv')
        
        if hr_std is not None:
            if hr_std > 50:
                status = "good - indicating healthy autonomic function and recovery"
            elif hr_std > 30:
                status = "moderate - within normal range"
            else:
                status = "low - may indicate stress or fatigue"
            return f"Your heart rate variability (HRV) is approximately {hr_std:.1f} ms, which is {status}."
        
        # Try to estimate from min/max if available
        hr_min = health_data.get('hr_min_1h')
        hr_max = health_data.get('hr_max_1h')
        if hr_min and hr_max:
            range_val = hr_max - hr_min
            return f"Based on your heart rate range ({hr_min:.0f}-{hr_max:.0f} BPM), you show {'good' if range_val > 20 else 'limited'} heart rate variability."
            
        return "HRV data is not currently available. This metric requires continuous heart rate monitoring."
        
    def _handle_temperature_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle temperature queries"""
        core_temp = health_data.get('core_temp')
        ambient = health_data.get('ambient_temp')
        
        if core_temp is None and ambient is None:
            return "Temperature data is not currently available."
            
        parts = []
        
        if core_temp is not None:
            temp_f = core_temp * 9/5 + 32
            
            if 'fever' in query_lower:
                if core_temp > 38:
                    parts.append(f"Yes, your temperature of {core_temp:.1f}°C ({temp_f:.1f}°F) indicates a fever")
                elif core_temp > 37.5:
                    parts.append(f"Your temperature is slightly elevated at {core_temp:.1f}°C ({temp_f:.1f}°F) - borderline fever")
                else:
                    parts.append(f"No, your temperature is {core_temp:.1f}°C ({temp_f:.1f}°F), which is normal")
            elif 'normal' in query_lower or 'elevated' in query_lower:
                if core_temp < 36:
                    parts.append(f"Your core temperature is {core_temp:.1f}°C ({temp_f:.1f}°F), below normal range")
                elif core_temp > 37.5:
                    parts.append(f"Your core temperature is {core_temp:.1f}°C ({temp_f:.1f}°F), elevated")
                else:
                    parts.append(f"Your core temperature is {core_temp:.1f}°C ({temp_f:.1f}°F), normal")
            elif 'skin' in query_lower:
                parts.append(f"Core temperature is {core_temp:.1f}°C ({temp_f:.1f}°F). Skin temperature is typically 1-2°C lower.")
            else:
                parts.append(f"Your body temperature is {core_temp:.1f}°C ({temp_f:.1f}°F)")
                if core_temp > 37.5:
                    parts.append("which is elevated")
                elif core_temp < 36:
                    parts.append("which is below normal")
                else:
                    parts.append("which is normal")
                    
        if ambient is not None and ('ambient' in query_lower or 'room' in query_lower or len(parts) == 0):
            parts.append(f"Ambient temperature is {ambient:.1f}°C")
            
        heat_index = health_data.get('heat_index')
        if heat_index is not None and heat_index > 30:
            parts.append(f"Heat index is {heat_index:.1f}°C - take precautions")
            
        return ". ".join(parts) + "."
        
    def _handle_oxygen_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle oxygen/respiratory queries"""
        spo2 = health_data.get('spo2')
        
        # Breathing rate is not directly measured
        if 'breathing rate' in query_lower or 'respiratory rate' in query_lower or 'how fast am i breathing' in query_lower:
            return "Breathing rate is not directly measured by BeAST sensors. However, your blood oxygen is monitored which relates to respiratory function."
            
        if spo2 is None:
            return "Blood oxygen (SpO2) data is not currently available."
            
        if spo2 >= 98:
            status = "excellent"
            advice = "Your oxygen saturation is optimal."
        elif spo2 >= 95:
            status = "normal"
            advice = "Your oxygen levels are healthy."
        elif spo2 >= 90:
            status = "slightly low"
            advice = "Consider taking deep breaths and monitoring. If persistent, consult a healthcare provider."
        else:
            status = "low"
            advice = "This may need attention. If you feel unwell, seek medical advice."
            
        if 'level' in query_lower or 'spo2' in query_lower or 'saturation' in query_lower:
            return f"Your blood oxygen (SpO2) is {spo2:.0f}%, which is {status}. {advice}"
        else:
            return f"Your oxygen saturation is {spo2:.0f}%, which is {status}. {advice}"
            
    def _handle_stress_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle stress related queries"""
        # Derive stress from multiple factors
        cognitive = health_data.get('cognitive_load')
        hrv = health_data.get('hr_variability') or health_data.get('hrv')
        hr = health_data.get('heart_rate')
        
        # Calculate composite stress score
        stress_score = None
        factors = []
        
        if cognitive is not None:
            stress_score = cognitive * 0.5  # Cognitive load contributes 50%
            factors.append(f"cognitive load at {cognitive:.0f}")
            
        if hr is not None and hr > 80:
            hr_stress = min((hr - 80) * 2, 50)  # Up to 50 points from elevated HR
            stress_score = (stress_score or 0) + hr_stress * 0.3
            if hr > 90:
                factors.append(f"elevated heart rate at {hr:.0f} BPM")
                
        if stress_score is None:
            return "I don't have enough data to assess your stress level. Cognitive and heart rate metrics are needed."
            
        # Normalize to 0-100
        stress_score = min(stress_score, 100)
        
        if 'high' in query_lower or 'low' in query_lower or 'normal' in query_lower:
            if stress_score < 30:
                return f"Your stress level is low ({stress_score:.0f}/100). You appear calm and relaxed."
            elif stress_score < 50:
                return f"Your stress level is moderate ({stress_score:.0f}/100). This is manageable."
            elif stress_score < 70:
                return f"Your stress level is elevated ({stress_score:.0f}/100). Consider taking a break or practicing relaxation techniques."
            else:
                return f"Your stress level is high ({stress_score:.0f}/100). Factors: {', '.join(factors)}. Taking a break is recommended."
        elif 'stressed' in query_lower or 'how stressed' in query_lower:
            if stress_score < 30:
                return f"You don't appear very stressed. Your stress index is {stress_score:.0f}/100."
            elif stress_score < 60:
                return f"You're moderately stressed. Stress index: {stress_score:.0f}/100."
            else:
                return f"Yes, you appear stressed. Stress index: {stress_score:.0f}/100. {', '.join(factors).capitalize()}."
        else:
            return f"Your stress index is {stress_score:.0f}/100. " + ("You appear relaxed." if stress_score < 40 else "Consider stress management techniques." if stress_score > 60 else "Moderate stress detected.")
            
    def _handle_cognitive_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle cognitive load queries"""
        cognitive = health_data.get('cognitive_load')
        alertness = health_data.get('alertness')
        
        if cognitive is None:
            return "Cognitive load data is not currently available."
            
        if cognitive < 30:
            status = "low"
            advice = "You have plenty of mental capacity available."
        elif cognitive < 50:
            status = "moderate"
            advice = "Your mental workload is manageable."
        elif cognitive < 70:
            status = "elevated"
            advice = "You're handling a significant mental workload. Consider prioritizing tasks."
        else:
            status = "high"
            advice = "Your cognitive load is high. Performance may be affected. Consider reducing tasks or taking a break."
            
        response = f"Your cognitive load is {cognitive:.0f}/100, which is {status}. {advice}"
        
        if alertness is not None:
            response += f" Alertness is at {alertness:.0f}%."
            
        if 'performance' in query_lower or 'score' in query_lower:
            performance = max(0, 100 - cognitive)
            response += f" Estimated cognitive performance capacity: {performance:.0f}%."
            
        return response
        
    def _handle_alertness_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle alertness queries"""
        alertness = health_data.get('alertness')
        fatigue = health_data.get('fatigue')
        
        if alertness is None:
            return "Alertness data is not currently available."
            
        if alertness > 80:
            status = "highly alert"
            advice = "You're in an optimal state of alertness."
        elif alertness > 60:
            status = "alert"
            advice = "Your alertness is good."
        elif alertness > 40:
            status = "moderately alert"
            advice = "Your alertness is declining. Consider a break or caffeine."
        else:
            status = "low alertness"
            advice = "Alertness is low. Rest or stimulation recommended."
            
        if 'vigilance' in query_lower or 'vigilant' in query_lower:
            return f"Your vigilance level is {alertness:.0f}%. {advice}"
        elif 'readiness' in query_lower or 'ready' in query_lower:
            readiness = alertness
            if fatigue is not None:
                readiness = max(0, alertness - fatigue * 0.3)
            return f"Your readiness score is {readiness:.0f}%. " + ("You're ready for demanding tasks." if readiness > 70 else "Consider recovery before high-demand activities." if readiness < 40 else "Adequate for normal activities.")
        else:
            response = f"Your alertness is {alertness:.0f}%, which is {status}. {advice}"
            if fatigue is not None:
                response += f" Fatigue level: {fatigue:.0f}/100."
            return response
            
    def _handle_fatigue_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle fatigue queries"""
        fatigue = health_data.get('fatigue')
        alertness = health_data.get('alertness')
        
        if fatigue is None:
            return "Fatigue data is not currently available."
            
        if fatigue < 20:
            status = "fresh"
            advice = "You appear well-rested and energized."
        elif fatigue < 40:
            status = "mild"
            advice = "Minor fatigue. You're doing fine."
        elif fatigue < 60:
            status = "moderate"
            advice = "Moderate fatigue detected. A short break would help."
        elif fatigue < 80:
            status = "significant"
            advice = "Significant fatigue. Rest is recommended."
        else:
            status = "severe"
            advice = "Severe fatigue detected. You should rest soon for safety and performance."
            
        if 'drowsy' in query_lower or 'sleepy' in query_lower:
            drowsy = fatigue > 50 or (alertness is not None and alertness < 40)
            if drowsy:
                return f"Yes, you appear drowsy. Fatigue is at {fatigue:.0f}/100. Rest is advised."
            else:
                return f"You don't appear drowsy. Fatigue level is {fatigue:.0f}/100."
        elif 'too tired' in query_lower:
            if fatigue > 70:
                return f"Yes, you appear too tired to continue at full capacity. Fatigue: {fatigue:.0f}/100. Rest recommended."
            else:
                return f"Your fatigue level ({fatigue:.0f}/100) is manageable, though rest is always beneficial."
        elif 'score' in query_lower or 'level' in query_lower:
            return f"Your fatigue score is {fatigue:.0f}/100, which indicates {status} fatigue. {advice}"
        elif 'energy' in query_lower:
            energy = 100 - fatigue
            return f"Your energy level is approximately {energy:.0f}%. {advice}"
        else:
            return f"Your fatigue level is {fatigue:.0f}/100 ({status}). {advice}"
            
    def _handle_hydration_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle hydration queries"""
        hydration = health_data.get('hydration')
        
        if hydration is None:
            return "Hydration data is not currently available. This metric requires bioimpedance sensing."
            
        if hydration > 80:
            status = "well hydrated"
            advice = "Hydration status is optimal."
        elif hydration > 60:
            status = "adequately hydrated"
            advice = "Your hydration is acceptable."
        elif hydration > 40:
            status = "mildly dehydrated"
            advice = "Consider drinking water soon."
        else:
            status = "dehydrated"
            advice = "You need to hydrate. Drink water now."
            
        if 'dehydrated' in query_lower:
            if hydration < 50:
                return f"Yes, you appear to be dehydrated. Hydration level: {hydration:.0f}%. Please drink water."
            else:
                return f"No, your hydration level is {hydration:.0f}%, which is adequate."
        elif 'sweat' in query_lower:
            return "Sweat rate is estimated based on activity and temperature. With hydration at {hydration:.0f}%, ensure you're replacing fluids during activity."
        elif 'electrolyte' in query_lower:
            return f"Electrolyte balance is related to hydration. Your hydration is {hydration:.0f}%. If exercising heavily, consider electrolyte replacement."
        else:
            return f"Your hydration level is {hydration:.0f}%, which is {status}. {advice}"
            
    def _handle_metabolic_query(self, health_data: Dict) -> str:
        """Handle metabolic/calorie queries"""
        # These are estimated values based on activity
        hr = health_data.get('heart_rate')
        motion = health_data.get('motion_magnitude')
        
        if hr is None:
            return "Metabolic data requires heart rate monitoring, which is not currently available."
            
        # Very rough estimation
        # Basal metabolic rate varies, using simple estimate
        bmr_per_min = 1.2  # kcal per minute at rest
        
        if motion and motion > 11:
            multiplier = 1.5  # Active
        elif motion and motion > 10:
            multiplier = 1.2  # Light activity
        else:
            multiplier = 1.0  # Rest
            
        if hr > 100:
            multiplier *= 1.3
        elif hr > 80:
            multiplier *= 1.1
            
        estimated_rate = bmr_per_min * multiplier
        
        return f"Estimated metabolic rate: ~{estimated_rate:.1f} kcal/min. Heart rate: {hr:.0f} BPM. Note: This is a rough estimate. Actual values depend on individual factors."
        
    def _handle_activity_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle activity/motion queries"""
        motion = health_data.get('motion_magnitude')
        
        if motion is None:
            return "Activity data is not currently available. IMU sensors needed."
            
        # Motion magnitude interpretation (based on accelerometer magnitude)
        # ~9.8 = stationary (just gravity), higher = moving
        if motion < 9.9:
            activity = "stationary"
            description = "You appear to be sitting or lying still."
        elif motion < 10.2:
            activity = "minimal"
            description = "Very light movement detected, possibly sitting with small movements."
        elif motion < 11:
            activity = "light"
            description = "Light activity detected, possibly walking slowly or fidgeting."
        elif motion < 13:
            activity = "moderate"
            description = "Moderate activity detected, possibly walking or light exercise."
        else:
            activity = "vigorous"
            description = "High activity level detected."
            
        if 'steps' in query_lower:
            return "Step counting requires continuous motion tracking. Current activity level: " + activity + ". For accurate step count, use a dedicated step tracker."
        elif 'exertion' in query_lower:
            hr = health_data.get('heart_rate')
            if hr:
                if hr > 120:
                    return f"Your exertion level is high. Heart rate: {hr:.0f} BPM, Activity: {activity}."
                elif hr > 90:
                    return f"Your exertion level is moderate. Heart rate: {hr:.0f} BPM, Activity: {activity}."
                else:
                    return f"Your exertion level is low. Heart rate: {hr:.0f} BPM, Activity: {activity}."
            return f"Activity level is {activity}, but heart rate data is needed for accurate exertion assessment."
        elif 'still' in query_lower or 'moving' in query_lower:
            if activity == "stationary":
                return "Yes, you appear to be still/stationary."
            else:
                return f"You appear to be moving. Activity level: {activity}. {description}"
        else:
            return f"Your activity level is {activity}. {description}"
            
    def _handle_trend_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle time-based and trend queries"""
        import re
        
        # Extract time mentions
        minutes_match = re.search(r'(\d+)\s*minutes?\s*ago', query_lower)
        if minutes_match and self.database:
            try:
                minutes = int(minutes_match.group(1))
                
                # Heart rate queries
                if 'heart' in query_lower or 'hr' in query_lower or 'pulse' in query_lower:
                    result = self.database.get_sensor_data_at('ppg', minutes_ago=minutes)
                    if result and result.get('heart_rate') is not None:
                        ts = result.get('timestamp')
                        hr = result['heart_rate']
                        return f"Your heart rate {minutes} minutes ago was {hr:.0f} BPM (timestamp: {ts})."
                    else:
                        return "I couldn't find a heart rate reading for that time window in the local data."
                
                # Temperature queries
                elif 'temp' in query_lower or 'temperature' in query_lower:
                    result = self.database.get_sensor_data_at('temp', minutes_ago=minutes)
                    if result and result.get('core_temp') is not None:
                        ts = result.get('timestamp')
                        temp = result['core_temp']
                        return f"Your core temperature {minutes} minutes ago was {temp:.1f}°F (timestamp: {ts})."
                    else:
                        return "I couldn't find a temperature reading for that time window."
                
                # Oxygen/SpO2 queries
                elif 'oxygen' in query_lower or 'spo2' in query_lower or 'saturation' in query_lower:
                    result = self.database.get_sensor_data_at('ppg', minutes_ago=minutes)
                    if result and result.get('spo2') is not None:
                        ts = result.get('timestamp')
                        spo2 = result['spo2']
                        return f"Your blood oxygen (SpO2) {minutes} minutes ago was {spo2:.0f}% (timestamp: {ts})."
                    else:
                        return "I couldn't find an oxygen saturation reading for that time window."
                
            except Exception as e:
                logger.error(f"Trend query heart rate lookup failed: {e}")
                # Fall through to existing trend responses
        
        # For now, we have 1-hour aggregates
        hr = health_data.get('heart_rate')
        hr_avg = health_data.get('hr_avg_1h')
        hr_min = health_data.get('hr_min_1h')
        hr_max = health_data.get('hr_max_1h')
        
        if 'highest' in query_lower and 'heart' in query_lower:
            if hr_max:
                return f"Your highest heart rate in the last hour was {hr_max:.0f} BPM."
            return "Historical heart rate data is not available."
            
        if 'lowest' in query_lower and 'heart' in query_lower:
            if hr_min:
                return f"Your lowest heart rate in the last hour was {hr_min:.0f} BPM."
            return "Historical heart rate data is not available."
            
        if 'average' in query_lower and 'heart' in query_lower:
            if hr_avg:
                return f"Your average heart rate over the last hour is {hr_avg:.0f} BPM (range: {hr_min:.0f}-{hr_max:.0f})."
            return "Historical heart rate data is not available."
            
        if 'vitals' in query_lower and ('last hour' in query_lower or 'hour' in query_lower):
            parts = []
            if hr_avg:
                parts.append(f"Heart rate avg: {hr_avg:.0f} BPM (range: {hr_min:.0f}-{hr_max:.0f})")
            spo2_avg = health_data.get('spo2_avg_1h')
            if spo2_avg:
                parts.append(f"SpO2 avg: {spo2_avg:.0f}%")
            if parts:
                return "Last hour vitals: " + ", ".join(parts) + "."
            return "Historical vital data is not available."
            
        if 'changed' in query_lower or 'trend' in query_lower:
            if hr and hr_avg:
                diff = hr - hr_avg
                if abs(diff) < 5:
                    return f"Your heart rate is stable. Current: {hr:.0f} BPM, 1-hour average: {hr_avg:.0f} BPM."
                elif diff > 0:
                    return f"Your heart rate has increased. Current: {hr:.0f} BPM vs 1-hour average: {hr_avg:.0f} BPM (+{diff:.0f})."
                else:
                    return f"Your heart rate has decreased. Current: {hr:.0f} BPM vs 1-hour average: {hr_avg:.0f} BPM ({diff:.0f})."
                    
        # Current vital signs summary
        if 'vital' in query_lower or 'current' in query_lower:
            return self._generate_status_summary(health_data)
            
        return "Trend data is limited to 1-hour summaries. Ask about average, highest, or lowest values over the last hour."
        
    def _handle_threshold_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle threshold/alert status queries"""
        alerts = []
        normal = []
        
        # Check heart rate
        hr = health_data.get('heart_rate')
        if hr is not None:
            if hr < 50 or hr > 110:
                alerts.append(f"Heart rate ({hr:.0f} BPM) is outside normal range")
            else:
                normal.append("heart rate")
                
        # Check SpO2
        spo2 = health_data.get('spo2')
        if spo2 is not None:
            if spo2 < 94:
                alerts.append(f"Blood oxygen ({spo2:.0f}%) is low")
            else:
                normal.append("oxygen")
                
        # Check cognitive load
        cognitive = health_data.get('cognitive_load')
        if cognitive is not None:
            if cognitive > 80:
                alerts.append(f"Cognitive load ({cognitive:.0f}/100) is very high")
            else:
                normal.append("cognitive load")
                
        # Check fatigue
        fatigue = health_data.get('fatigue')
        if fatigue is not None:
            if fatigue > 70:
                alerts.append(f"Fatigue ({fatigue:.0f}/100) is high")
            else:
                normal.append("fatigue levels")
                
        # Check alertness
        alertness = health_data.get('alertness')
        if alertness is not None:
            if alertness < 30:
                alerts.append(f"Alertness ({alertness:.0f}%) is very low")
            else:
                normal.append("alertness")
                
        # Check hydration
        hydration = health_data.get('hydration')
        if hydration is not None:
            if hydration < 40:
                alerts.append(f"Hydration ({hydration:.0f}%) is low")
            else:
                normal.append("hydration")
                
        # Check temperature
        core_temp = health_data.get('core_temp')
        if core_temp is not None:
            if core_temp > 38 or core_temp < 35.5:
                alerts.append(f"Body temperature ({core_temp:.1f}°C) is abnormal")
            else:
                normal.append("temperature")
                
        # Format response based on query type
        if 'abnormal' in query_lower or 'attention' in query_lower or 'any metrics' in query_lower:
            if alerts:
                return f"⚠️ Metrics needing attention: {'; '.join(alerts)}."
            else:
                return "All monitored metrics are within normal ranges."
                
        if 'normal' in query_lower or 'safe' in query_lower:
            if not alerts:
                return f"✓ Yes, all your vitals are in normal/safe ranges. Monitoring: {', '.join(normal)}."
            else:
                return f"Most metrics are normal ({', '.join(normal)}), but some need attention: {'; '.join(alerts)}."
                
        if 'everything' in query_lower:
            if not alerts:
                return "Yes, everything looks normal. All monitored vitals are within healthy ranges."
            else:
                return f"Not quite - some metrics need attention: {'; '.join(alerts)}."
                
        # Overall health status
        if len(alerts) == 0:
            return f"✓ Overall health status: Good. All {len(normal)} monitored metrics are within normal ranges."
        elif len(alerts) <= 2:
            return f"⚠️ Overall status: Mostly good, but {len(alerts)} metric(s) need attention: {'; '.join(alerts)}."
        else:
            return f"⚠️ Overall status: Multiple metrics need attention: {'; '.join(alerts)}. Consider taking a break."
            
    def _handle_yesno_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle yes/no threshold-based queries"""
        
        # Am I dehydrated?
        if 'dehydrated' in query_lower:
            hydration = health_data.get('hydration')
            if hydration is not None:
                if hydration < 40:
                    return f"Yes, you appear dehydrated. Hydration level: {hydration:.0f}%. Please drink water."
                elif hydration < 60:
                    return f"You're borderline. Hydration: {hydration:.0f}%. Drinking water would be beneficial."
                else:
                    return f"No, your hydration level ({hydration:.0f}%) is adequate."
            return "I don't have hydration data to assess this."
            
        # Is my heart rate elevated?
        if 'heart rate elevated' in query_lower or 'hr elevated' in query_lower:
            hr = health_data.get('heart_rate')
            if hr is not None:
                if hr > 100:
                    return f"Yes, your heart rate ({hr:.0f} BPM) is elevated above the normal resting range."
                else:
                    return f"No, your heart rate ({hr:.0f} BPM) is within normal range."
            return "Heart rate data is not available."
            
        # Am I too tired?
        if 'too tired' in query_lower:
            fatigue = health_data.get('fatigue')
            if fatigue is not None:
                if fatigue > 70:
                    return f"Yes, your fatigue level ({fatigue:.0f}/100) is high. Rest is recommended."
                elif fatigue > 50:
                    return f"You're getting tired (fatigue: {fatigue:.0f}/100). A break would help."
                else:
                    return f"No, your fatigue level ({fatigue:.0f}/100) is manageable."
            return "Fatigue data is not available."
            
        # Do I need rest?
        if 'need rest' in query_lower or 'need a break' in query_lower:
            fatigue = health_data.get('fatigue')
            cognitive = health_data.get('cognitive_load')
            alertness = health_data.get('alertness')
            
            need_rest = False
            reasons = []
            
            if fatigue is not None and fatigue > 60:
                need_rest = True
                reasons.append(f"fatigue at {fatigue:.0f}")
            if cognitive is not None and cognitive > 70:
                need_rest = True
                reasons.append(f"cognitive load at {cognitive:.0f}")
            if alertness is not None and alertness < 40:
                need_rest = True
                reasons.append(f"low alertness at {alertness:.0f}%")
                
            if need_rest:
                return f"Yes, rest is recommended. Factors: {', '.join(reasons)}."
            elif len(reasons) == 0 and (fatigue is None and cognitive is None):
                return "I don't have enough data to assess if you need rest."
            else:
                return "Not urgently, but rest is always beneficial. Your metrics look manageable."
                
        # Am I overexerted?
        if 'overexert' in query_lower:
            hr = health_data.get('heart_rate')
            fatigue = health_data.get('fatigue')
            
            if hr is not None and hr > 150:
                return f"Yes, your heart rate ({hr:.0f} BPM) suggests high exertion. Monitor closely."
            elif hr is not None and hr > 120 and fatigue is not None and fatigue > 60:
                return f"Possibly. Heart rate is {hr:.0f} BPM with fatigue at {fatigue:.0f}/100. Consider reducing intensity."
            elif hr is not None:
                return f"No, your heart rate ({hr:.0f} BPM) doesn't indicate overexertion."
            return "I need heart rate data to assess exertion levels."
            
        # Should I take a break?
        if 'take a break' in query_lower:
            fatigue = health_data.get('fatigue')
            cognitive = health_data.get('cognitive_load')
            
            if (fatigue is not None and fatigue > 50) or (cognitive is not None and cognitive > 60):
                score = max(fatigue or 0, cognitive or 0)
                return f"Yes, a break would be beneficial. Your load is at {score:.0f}/100."
            else:
                return "You don't urgently need a break, but regular breaks improve performance."
                
        # Am I ready to continue?
        if 'ready to continue' in query_lower or 'ready' in query_lower:
            alertness = health_data.get('alertness')
            fatigue = health_data.get('fatigue')
            
            if alertness is not None and alertness > 60 and (fatigue is None or fatigue < 50):
                return f"Yes, you appear ready. Alertness: {alertness:.0f}%."
            elif alertness is not None and alertness < 40:
                return f"Not optimally. Your alertness is low at {alertness:.0f}%. Consider a short break."
            elif fatigue is not None and fatigue > 60:
                return f"You could continue, but fatigue is elevated at {fatigue:.0f}/100. A break would help."
            return "Based on available data, you appear ready to continue."
            
        # Am I fit for duty?
        if 'fit for duty' in query_lower:
            alerts = []
            hr = health_data.get('heart_rate')
            if hr is not None and (hr < 50 or hr > 110):
                alerts.append("heart rate")
            fatigue = health_data.get('fatigue')
            if fatigue is not None and fatigue > 70:
                alerts.append("high fatigue")
            alertness = health_data.get('alertness')
            if alertness is not None and alertness < 40:
                alerts.append("low alertness")
            cognitive = health_data.get('cognitive_load')
            if cognitive is not None and cognitive > 80:
                alerts.append("cognitive overload")
                
            if alerts:
                return f"Caution advised. Concerns: {', '.join(alerts)}. Consider assessment before demanding tasks."
            else:
                return "Yes, based on available metrics, you appear fit for duty."
                
        return "I couldn't determine a yes/no answer for that question. Try asking more specifically."
        
    def _handle_device_query(self, query_lower: str, health_data: Dict) -> str:
        """Handle device/system status queries"""
        
        # Battery (would need to be added to data)
        if 'battery' in query_lower or 'power' in query_lower:
            battery = health_data.get('battery_level')
            if battery is not None:
                if battery > 50:
                    return f"Battery level is {battery:.0f}%. Good charge remaining."
                elif battery > 20:
                    return f"Battery level is {battery:.0f}%. Consider charging soon."
                else:
                    return f"Battery level is {battery:.0f}%. Low battery - charge soon."
            return "Battery level information is not available from current sensors."
            
        # Sensor connection status
        if 'connected' in query_lower or 'sensor' in query_lower or 'working' in query_lower:
            connected_sensors = []
            if health_data.get('heart_rate') is not None:
                connected_sensors.append("PPG (heart/SpO2)")
            if health_data.get('cognitive_load') is not None:
                connected_sensors.append("EEG (cognitive)")
            if health_data.get('core_temp') is not None:
                connected_sensors.append("Temperature")
            if health_data.get('hydration') is not None:
                connected_sensors.append("Bioimpedance")
            if health_data.get('motion_magnitude') is not None:
                connected_sensors.append("IMU (motion)")
                
            if connected_sensors:
                return f"Active sensors: {', '.join(connected_sensors)}. Receiving data normally."
            else:
                return "No sensor data is being received. Check connections."
                
        # Last reading
        if 'last reading' in query_lower or 'when' in query_lower:
            timestamps = []
            for key in ['ppg_timestamp', 'cognitive_timestamp', 'temp_timestamp', 'imu_timestamp']:
                ts = health_data.get(key)
                if ts:
                    timestamps.append(ts)
            if timestamps:
                latest = max(timestamps)
                return f"Last sensor reading received at {latest}."
            return "No recent sensor readings available."
            
        # Signal quality (would need to be added)
        if 'signal' in query_lower or 'quality' in query_lower:
            return "Signal quality metrics are not currently available. Check physical sensor placement for best results."
            
        return "Device status information is limited. Check physical connections and sensor placement."
        
    def _handle_eeg_query(self, health_data: Dict) -> str:
        """Handle EEG/brain wave queries"""
        eeg_data = []
        if health_data.get('eeg_alpha') is not None:
            eeg_data.append(f"Alpha: {health_data['eeg_alpha']:.1f}")
        if health_data.get('eeg_beta') is not None:
            eeg_data.append(f"Beta: {health_data['eeg_beta']:.1f}")
        if health_data.get('eeg_theta') is not None:
            eeg_data.append(f"Theta: {health_data['eeg_theta']:.1f}")
        if health_data.get('eeg_gamma') is not None:
            eeg_data.append(f"Gamma: {health_data['eeg_gamma']:.1f}")
        if health_data.get('eeg_delta') is not None:
            eeg_data.append(f"Delta: {health_data['eeg_delta']:.1f}")
            
        if eeg_data:
            interpretation = ""
            alpha = health_data.get('eeg_alpha', 0)
            beta = health_data.get('eeg_beta', 0)
            theta = health_data.get('eeg_theta', 0)
            
            if beta > alpha:
                interpretation = "Higher beta suggests active focus or alertness."
            elif alpha > beta:
                interpretation = "Higher alpha suggests relaxation."
            if theta > alpha:
                interpretation += " Elevated theta may indicate drowsiness or deep relaxation."
                
            return f"EEG band powers - {', '.join(eeg_data)}. {interpretation}"
        return "EEG data is not currently available."
        
    def _handle_environment_query(self, health_data: Dict) -> str:
        """Handle environment queries"""
        parts = []
        if health_data.get('ambient_temp') is not None:
            parts.append(f"Ambient temperature: {health_data['ambient_temp']:.1f}°C")
        if health_data.get('humidity') is not None:
            parts.append(f"Humidity: {health_data['humidity']:.0f}%")
        if health_data.get('heat_index') is not None:
            hi = health_data['heat_index']
            parts.append(f"Heat index: {hi:.1f}°C")
            if hi > 32:
                parts.append("⚠️ High heat index - take precautions")
                
        if parts:
            return "Environment conditions: " + ", ".join(parts) + "."
        return "Environment data is not currently available."
        
    def _handle_zone_query(self, health_data: Dict) -> str:
        """Handle zone duration queries"""
        zones = health_data.get('zone_durations', {})
        if zones:
            parts = []
            for metric, zone_data in zones.items():
                zone_str = ", ".join([f"{z}: {d//60}m {d%60}s" for z, d in zone_data.items()])
                parts.append(f"{metric}: {zone_str}")
            return "Time in zones: " + "; ".join(parts) + "."
        return "Zone duration data is not available for this session."
        
    def _generate_status_summary(self, health_data: Dict) -> str:
        """Generate overall status summary"""
        parts = []
        
        # Heart rate
        hr = health_data.get('heart_rate')
        if hr:
            status = "normal" if 60 <= hr <= 100 else ("low" if hr < 60 else "elevated")
            parts.append(f"heart rate is {hr:.0f} BPM ({status})")
            
        # SpO2
        spo2 = health_data.get('spo2')
        if spo2:
            status = "good" if spo2 >= 95 else "low"
            parts.append(f"oxygen is {spo2:.0f}% ({status})")
            
        # Cognitive load
        cognitive = health_data.get('cognitive_load')
        if cognitive is not None:
            if cognitive < 40:
                parts.append("mental load is low")
            elif cognitive < 70:
                parts.append("moderate cognitive load")
            else:
                parts.append("high cognitive load")
                
        # Alertness
        alertness = health_data.get('alertness')
        if alertness is not None:
            if alertness > 70:
                parts.append("you're alert")
            elif alertness > 40:
                parts.append("moderate alertness")
            else:
                parts.append("low alertness")
                
        # Fatigue
        fatigue = health_data.get('fatigue')
        if fatigue is not None:
            if fatigue < 30:
                parts.append("well-rested")
            elif fatigue < 70:
                parts.append("some fatigue detected")
            else:
                parts.append("significant fatigue, rest recommended")
                
        # Hydration
        hydration = health_data.get('hydration')
        if hydration is not None:
            if hydration < 50:
                parts.append("need to hydrate")
            elif hydration < 70:
                parts.append("could use some water")
                
        # Temperature
        core_temp = health_data.get('core_temp')
        if core_temp is not None:
            if core_temp > 37.5:
                parts.append(f"temperature elevated at {core_temp:.1f}°C")
            elif core_temp < 36:
                parts.append(f"temperature low at {core_temp:.1f}°C")
                
        # Environment
        heat_index = health_data.get('heat_index')
        if heat_index is not None and heat_index > 32:
            parts.append(f"heat index is high ({heat_index:.0f}°C)")
                
        if parts:
            return "Here's your status: " + ", ".join(parts) + "."
        return "I don't have enough data for a status summary. Make sure sensors are connected."
    
    def close(self):
        """Close database connections"""
        if self.pg_conn:
            try:
                self.pg_conn.close()
                logger.info("PostgreSQL connection closed")
            except:
                pass
        
    def query(self, question: str) -> str:
        """
        Main query method - process user question and generate response.
        
        Args:
            question: User's question
            
        Returns:
            Generated response string
        """
        logger.info(f"Processing query: {question}")
        start_time = time.time()
        
        try:
            # Get user health data
            health_data = self._get_user_health_context()
            
            # Retrieve relevant knowledge
            knowledge = self._retrieve_knowledge(question)
            context = " ".join(knowledge)
            
            # Generate response
            response = self._generate_response(question, context, health_data)
            
            elapsed = time.time() - start_time
            logger.info(f"Query processed in {elapsed:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return "I'm sorry, I encountered an error processing your question. Please try again."


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Health RAG System - Comprehensive Query Coverage")
    print("=" * 60)
    
    # Test configuration
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'beast',
            'user': 'beast',
            'password': ''
        },
        'device_id': None,
        'session_id': None
    }
    
    # Initialize RAG
    rag = HealthRAG(config=config)
    
    # Use mock data for testing
    if not rag.pg_conn:
        print("\nNo PostgreSQL connection - using mock data for demo")
    
    rag.user_context = {
        'heart_rate': 78,
        'spo2': 97,
        'cognitive_load': 55,
        'alertness': 68,
        'fatigue': 42,
        'hydration': 62,
        'core_temp': 36.8,
        'ambient_temp': 24.5,
        'humidity': 55,
        'heat_index': 25.2,
        'eeg_alpha': 12.5,
        'eeg_beta': 8.3,
        'eeg_theta': 6.1,
        'eeg_gamma': 4.2,
        'eeg_delta': 15.0,
        'motion_magnitude': 10.2,
        'hr_avg_1h': 75,
        'hr_min_1h': 62,
        'hr_max_1h': 92,
        'hr_variability': 45,
        'spo2_avg_1h': 97,
    }
    rag._get_user_health_context = lambda: rag.user_context
    
    # Comprehensive test queries organized by category
    test_categories = {
        "Heart Rate/Cardiovascular": [
            "What is my heart rate?",
            "What's my current heart rate?",
            "What is my pulse?",
            "Is my heart rate normal?",
            "What's my resting heart rate?",
            "What's my average heart rate this hour?",
            "How fast is my heart beating?",
            "What is my heart rate variability?",
        ],
        "Blood Pressure": [
            "What is my blood pressure?",
        ],
        "Body Temperature": [
            "What is my body temperature?",
            "What's my core temperature?",
            "Is my temperature normal?",
            "Do I have a fever?",
        ],
        "Respiratory/Oxygen": [
            "What is my breathing rate?",
            "What is my oxygen saturation?",
            "What's my SpO2 level?",
        ],
        "Stress": [
            "What is my stress level?",
            "How stressed am I?",
            "Is my stress level high?",
        ],
        "Cognitive Load": [
            "What is my cognitive load?",
            "What's my mental load level?",
            "What's my cognitive performance score?",
        ],
        "Alertness & Fatigue": [
            "What is my alertness level?",
            "How alert am I?",
            "What's my fatigue level?",
            "How tired am I?",
            "Am I drowsy?",
            "What's my readiness score?",
        ],
        "Hydration": [
            "What's my hydration level?",
            "Am I dehydrated?",
        ],
        "Activity": [
            "What's my activity level?",
            "What's my current exertion level?",
        ],
        "Trends": [
            "What's my average heart rate over the last hour?",
            "What's my highest heart rate today?",
            "What's my lowest heart rate today?",
            "What are my current vital signs?",
        ],
        "Threshold/Alerts": [
            "Are any of my vitals abnormal?",
            "Are my vitals in normal range?",
            "What's my overall health status?",
            "Is everything normal?",
        ],
        "Yes/No Assessments": [
            "Am I dehydrated?",
            "Is my heart rate elevated?",
            "Am I too tired?",
            "Do I need rest?",
            "Should I take a break?",
            "Am I ready to continue?",
            "Am I fit for duty?",
        ],
        "Device Status": [
            "What's the battery level?",
            "Are all sensors working?",
            "When was the last reading?",
        ],
    }
    
    for category, queries in test_categories.items():
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print("-" * 60)
        
        for query in queries:
            print(f"\nQ: {query}")
            response = rag.query(query)
            print(f"A: {response}")
            
    # Clean up
    rag.close()
    print("\n" + "=" * 60)
    print("Test complete! All query categories covered.")
    print("=" * 60)
