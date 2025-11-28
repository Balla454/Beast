#!/usr/bin/env python3
"""
Wake Word Detection for BeAST
==============================
Listens continuously for the wake word "Beast" to trigger voice interaction.

Supports multiple backends:
- pvporcupine (Picovoice Porcupine) - recommended for Pi
- vosk - open source alternative
- simple - basic energy-based detection (fallback)
"""

import logging
import os
import time
import struct
import numpy as np
from typing import Optional, Callable
from pathlib import Path

logger = logging.getLogger('BeAST.WakeWord')

# Try to import wake word engines
PORCUPINE_AVAILABLE = False
VOSK_AVAILABLE = False

try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
    logger.info("Porcupine wake word engine available")
except ImportError:
    logger.warning("Porcupine not available (pip install pvporcupine)")

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
    logger.info("Vosk wake word engine available")
except ImportError:
    logger.warning("Vosk not available (pip install vosk)")

# Audio capture
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available (pip install pyaudio)")


class WakeWordDetector:
    """
    Wake word detector that listens for "Beast" trigger word.
    
    Usage:
        detector = WakeWordDetector(wake_word="beast")
        
        # Blocking listen
        if detector.listen_for_wake_word():
            print("Wake word detected!")
            
        # Or with callback
        detector.start_continuous(callback=on_wake_word)
    """
    
    def __init__(self, 
                 wake_word: str = "beast",
                 sensitivity: float = 0.5,
                 engine: str = "auto"):
        """
        Initialize wake word detector.
        
        Args:
            wake_word: The word to listen for
            sensitivity: Detection sensitivity (0.0-1.0)
            engine: "porcupine", "vosk", "simple", or "auto"
        """
        self.wake_word = wake_word.lower()
        self.sensitivity = sensitivity
        self.running = False
        self.audio = None
        self.stream = None
        
        # Audio parameters
        self.sample_rate = 16000
        self.frame_length = 512  # ~32ms at 16kHz
        
        # Select engine
        if engine == "auto":
            if PORCUPINE_AVAILABLE:
                engine = "porcupine"
            elif VOSK_AVAILABLE:
                engine = "vosk"
            else:
                engine = "simple"
        
        self.engine = engine
        self._detector = None
        
        logger.info(f"Wake word detector using engine: {self.engine}")
        
        # Initialize selected engine
        self._init_engine()
        
    def _init_engine(self):
        """Initialize the selected wake word engine"""
        if self.engine == "porcupine":
            self._init_porcupine()
        elif self.engine == "vosk":
            self._init_vosk()
        else:
            self._init_simple()
            
    def _init_porcupine(self):
        """Initialize Porcupine wake word engine"""
        try:
            # Porcupine requires an access key for custom wake words
            # For "computer" or other built-in words, no key needed
            # For custom "beast", you'd need to train a model
            
            # Try built-in keywords first
            keywords = ["computer"]  # Fallback to "computer" if "beast" not available
            
            self._detector = pvporcupine.create(
                keywords=keywords,
                sensitivities=[self.sensitivity]
            )
            self.frame_length = self._detector.frame_length
            self.sample_rate = self._detector.sample_rate
            
            logger.info(f"Porcupine initialized with keywords: {keywords}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            logger.info("Falling back to simple detector")
            self.engine = "simple"
            self._init_simple()
            
    def _init_vosk(self):
        """Initialize Vosk for wake word detection"""
        try:
            # Vosk model path - check env var, then common locations
            model_path = os.environ.get('VOSK_MODEL_PATH')
            
            if not model_path or not Path(model_path).exists():
                search_paths = [
                    Path.home() / "models" / "vosk-model-small-en-us-0.15",
                    Path(__file__).parent.parent / "models" / "vosk-model-small-en-us-0.15",
                    Path("/usr/share/vosk-models/vosk-model-small-en-us-0.15"),
                ]
                for p in search_paths:
                    if p.exists():
                        model_path = str(p)
                        break
                else:
                    model_path = None
            
            if not model_path or not Path(model_path).exists():
                logger.warning(f"Vosk model not found")
                logger.info("Falling back to simple detector")
                self.engine = "simple"
                self._init_simple()
                return
                
            self._vosk_model = VoskModel(model_path)
            self._detector = KaldiRecognizer(self._vosk_model, self.sample_rate)
            
            logger.info("Vosk wake word detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vosk: {e}")
            self.engine = "simple"
            self._init_simple()
            
    def _init_simple(self):
        """Initialize simple energy-based detection (always listens)"""
        # This is a fallback that just detects when someone speaks
        # Not true wake word detection, but works as placeholder
        self._energy_threshold = 500  # Adjust based on environment
        logger.info("Using simple energy-based voice detection")
        
    def _init_audio(self):
        """Initialize PyAudio stream"""
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio not available")
            
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
            
        if self.stream is None or not self.stream.is_active():
            self.stream = self.audio.open(
                rate=self.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.frame_length
            )
            
    def listen_for_wake_word(self, timeout: float = None) -> bool:
        """
        Block and listen for wake word.
        
        Args:
            timeout: Maximum seconds to listen (None = forever)
            
        Returns:
            True if wake word detected, False if timeout
        """
        self._init_audio()
        self.running = True
        
        start_time = time.time()
        
        try:
            while self.running:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    return False
                
                # Read audio frame
                pcm = self.stream.read(self.frame_length, exception_on_overflow=False)
                pcm_data = struct.unpack_from(f"{self.frame_length}h", pcm)
                
                # Check for wake word based on engine
                if self._check_wake_word(pcm_data):
                    return True
                    
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            return False
            
        return False
        
    def _check_wake_word(self, pcm_data) -> bool:
        """Check if wake word is present in audio frame"""
        if self.engine == "porcupine" and self._detector:
            result = self._detector.process(pcm_data)
            return result >= 0
            
        elif self.engine == "vosk" and self._detector:
            # Convert to bytes for Vosk
            audio_bytes = struct.pack(f"{len(pcm_data)}h", *pcm_data)
            
            if self._detector.AcceptWaveform(audio_bytes):
                result = self._detector.Result()
                # Check if wake word is in the result
                if self.wake_word in result.lower():
                    return True
            else:
                partial = self._detector.PartialResult()
                if self.wake_word in partial.lower():
                    return True
            return False
            
        else:  # Simple energy detection
            # Calculate RMS energy
            energy = np.sqrt(np.mean(np.array(pcm_data, dtype=np.float32) ** 2))
            
            if energy > self._energy_threshold:
                # Voice activity detected - treat as wake word for simple mode
                # Wait a bit to ensure it's sustained speech
                time.sleep(0.1)
                return True
            return False
            
    def start_continuous(self, callback: Callable[[], None]):
        """
        Start continuous wake word detection with callback.
        
        Args:
            callback: Function to call when wake word detected
        """
        import threading
        
        def _listen_loop():
            while self.running:
                if self.listen_for_wake_word(timeout=1.0):
                    callback()
                    
        self.running = True
        self._thread = threading.Thread(target=_listen_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop wake word detection"""
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        if self.audio:
            self.audio.terminate()
            self.audio = None
            
        if self.engine == "porcupine" and self._detector:
            self._detector.delete()
            self._detector = None


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing wake word detection...")
    print("Say 'Beast' (or speak loudly for simple mode)...")
    
    detector = WakeWordDetector(wake_word="beast")
    
    try:
        while True:
            if detector.listen_for_wake_word(timeout=5.0):
                print(">>> Wake word detected! <<<")
            else:
                print("Timeout, still listening...")
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        detector.stop()
