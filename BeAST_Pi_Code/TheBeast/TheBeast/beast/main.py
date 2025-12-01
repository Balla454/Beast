#!/usr/bin/env python3
"""
beast Main Entry Point
======================
Raspberry Pi 5 Edge Deployment

Boot sequence:
1. Initialize all systems
2. Announce "beast is ready"
3. Enter wake word listening loop
4. Process voice commands
5. Return to listening

Power button triggers backup + shutdown (handled by triggerhappy)

NOTE: This system runs fully OFFLINE. All AI models (Whisper, embeddings, Ollama)
must be pre-downloaded. No internet connection is required during operation.
"""

import os

# Force offline mode BEFORE importing any ML libraries
# This prevents HuggingFace from trying to download models
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import time
import signal
import logging
import yaml
from pathlib import Path
from typing import Optional

# Add beast module to path
BEAST_DIR = Path(__file__).parent
sys.path.insert(0, str(BEAST_DIR))

from voice.wake_word import WakeWordDetector
from voice.speech_to_text import SpeechToText
from voice.text_to_speech import TextToSpeech
from rag.health_rag import HealthRAG
from processing.database_manager import DatabaseManager
from utils.announcer import Announcer

# Configure logging
LOG_DIR = Path(os.environ.get('BEAST_LOG_DIR', BEAST_DIR / 'logs'))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'beast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('beast')


class beastSystem:
    """Main beast system controller"""
    
    def __init__(self, config_path: str = None):
        self.running = False
        self.config = self._load_config(config_path)
        
        # Components (initialized in startup)
        self.wake_detector: Optional[WakeWordDetector] = None
        self.stt: Optional[SpeechToText] = None
        self.tts: Optional[TextToSpeech] = None
        self.rag: Optional[HealthRAG] = None
        self.db: Optional[DatabaseManager] = None
        self.announcer: Optional[Announcer] = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = BEAST_DIR / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return minimal default config with dynamic paths
            data_dir = Path(os.environ.get('BEAST_DATA_DIR', BEAST_DIR / 'data'))
            return {
                'system': {'name': 'beast'},
                'voice': {
                    'wake_word': 'beast',
                    'stt': {'engine': 'whisper', 'model': 'tiny.en'},
                    'tts': {'engine': 'piper'},
                    'audio': {'sample_rate': 16000, 'silence_duration': 1.5}
                },
                'paths': {
                    'database': str(data_dir / 'beast_local.db'),
                    'data_dir': str(data_dir)
                }
            }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize all system components"""
        logger.info("=" * 60)
        logger.info("beast System Initializing...")
        logger.info("=" * 60)
        
        try:
            # Ensure data directories exist
            data_dir = Path(self.config['paths']['data_dir'])
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize TTS first (for announcements)
            logger.info("Initializing Text-to-Speech...")
            self.tts = TextToSpeech(self.config['voice']['tts'])
            self.announcer = Announcer(self.tts)
            
            # Announce startup
            self.announcer.announce("beast system starting up. Please wait.")
            
            # Initialize Speech-to-Text
            logger.info("Initializing Speech-to-Text...")
            self.stt = SpeechToText(self.config['voice']['stt'])
            
            # Initialize Wake Word Detector
            logger.info("Initializing Wake Word Detection...")
            self.wake_detector = WakeWordDetector(
                wake_word=self.config['voice']['wake_word'],
                sensitivity=self.config['voice'].get('wake_word_sensitivity', 0.5),
                input_device=self.config['voice']['stt'].get('input_device', None)
            )
            
            # Initialize Database
            logger.info("Initializing Database...")
            self.db = DatabaseManager(self.config['paths']['database'])
            
            # Initialize RAG System
            logger.info("Initializing RAG System...")
            self.rag = HealthRAG(
                config=self.config.get('rag', {}),
                database=self.db
            )
            
            logger.info("=" * 60)
            logger.info("beast System Ready!")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            if self.announcer:
                self.announcer.announce("beast initialization failed. Please check logs.")
            return False
    
    def run(self):
        """Main interaction loop"""
        if not self.initialize():
            logger.error("Failed to initialize. Exiting.")
            return
        
        # Announce ready
        self.announcer.announce("beast is ready. Say Beast to ask a question.")
        self.running = True
        
        logger.info("Entering main interaction loop...")
        
        while self.running:
            try:
                # Step 1: Listen for wake word
                logger.debug("Listening for wake word...")
                
                if self.wake_detector.listen_for_wake_word():
                    # Wake word detected - close stream to release mic (but don't terminate PyAudio)
                    logger.info("Wake word detected!")
                    if self.wake_detector.stream:
                        self.wake_detector.stream.stop_stream()
                        self.wake_detector.stream.close()
                        self.wake_detector.stream = None
                    
                    self.announcer.play_sound("listening")  # Short beep
                    
                    # Step 2: Record user's question
                    logger.info("Recording user question...")
                    audio_data = self.stt.record_until_silence(
                        silence_duration=self.config['voice']['audio'].get('silence_duration', 1.5)
                    )
                    
                    # Close STT stream after recording
                    if hasattr(self.stt, 'stream') and self.stt.stream:
                        try:
                            self.stt.stream.stop_stream()
                            self.stt.stream.close()
                            self.stt.stream = None
                        except Exception as e:
                            logger.debug(f"Error closing STT stream: {e}")
                    
                    if audio_data is None or len(audio_data) == 0:
                        logger.warning("No audio recorded")
                        self.announcer.announce("I didn't hear anything. Try again.")
                        continue
                    
                    # Step 3: Transcribe audio to text
                    logger.info("Transcribing audio...")
                    question = self.stt.transcribe(audio_data)
                    
                    if not question or len(question.strip()) < 2:
                        logger.warning(f"Empty or invalid transcription: {question}")
                        self.announcer.announce("I didn't understand that. Please try again.")
                        continue
                    
                    logger.info(f"User asked: {question}")
                    
                    # Step 4: Process with RAG
                    logger.info("Processing with RAG...")
                    response = self.rag.query(question)
                    
                    logger.info(f"Response: {response[:100]}...")
                    
                    # Step 5: Speak response
                    self.tts.speak(response)
                    logger.info("TTS speak() returned")
                    
                    # Log interaction to database (optional)
                    if self.config.get('logging', {}).get('log_interactions', False):
                        self.db.log_interaction(question, response)
                    
                    # Wait a moment for TTS to finish and audio to settle
                    time.sleep(0.5)
                    
                    logger.info("Interaction complete. Returning to listening...")
                    
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.running = False
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                # Brief pause before retrying
                time.sleep(1)
        
        # Cleanup
        self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of all components"""
        logger.info("Shutting down beast system...")
        
        if self.announcer:
            self.announcer.announce("beast shutting down. Goodbye.")
        
        if self.wake_detector:
            self.wake_detector.stop()
            
        if self.db:
            self.db.close()
            
        logger.info("beast shutdown complete.")


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='beast Voice Assistant')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to config.yaml')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (no wake word required)')
    args = parser.parse_args()
    
    # Create and run system
    beast = beastSystem(config_path=args.config)
    
    if args.test:
        # Test mode: direct interaction without wake word
        logger.info("Running in TEST mode")
        if beast.initialize():
            beast.announcer.announce("Test mode active. Speak your question now.")
            while True:
                try:
                    audio = beast.stt.record_until_silence()
                    if audio:
                        text = beast.stt.transcribe(audio)
                        if text:
                            print(f"You said: {text}")
                            response = beast.rag.query(text)
                            print(f"Response: {response}")
                            beast.tts.speak(response)
                except KeyboardInterrupt:
                    break
            beast.shutdown()
    else:
        # Normal mode
        beast.run()


if __name__ == "__main__":
    main()
