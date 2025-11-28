#!/usr/bin/env python3
"""
Speech-to-Text for BeAST
========================
Converts spoken audio to text using local models.

Supports:
- Whisper (via transformers) - recommended for accuracy
- Moonshine - lightweight alternative
- Vosk - fully offline, fast
"""

import logging
import os
import time
import struct
import wave
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Union
import io

logger = logging.getLogger('BeAST.STT')

# Try to import STT engines
WHISPER_AVAILABLE = False
MOONSHINE_AVAILABLE = False
VOSK_AVAILABLE = False

try:
    from transformers import pipeline
    WHISPER_AVAILABLE = True
    logger.info("Whisper (transformers) available")
except ImportError:
    logger.warning("transformers not available (pip install transformers)")

try:
    from moonshine_stt import MoonshineSTT
    MOONSHINE_AVAILABLE = True
    logger.info("Moonshine STT available")
except ImportError:
    pass

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    import json
    VOSK_AVAILABLE = True
    logger.info("Vosk STT available")
except ImportError:
    pass

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available")


class SpeechToText:
    """
    Speech-to-text engine for BeAST.
    
    Usage:
        stt = SpeechToText(config={'engine': 'whisper', 'model': 'openai/whisper-tiny'})
        
        # Record and transcribe
        audio = stt.record_until_silence()
        text = stt.transcribe(audio)
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize STT engine.
        
        Args:
            config: Configuration dict with engine, model settings
        """
        config = config or {}
        self.engine = config.get('engine', 'whisper')
        self.model_name = config.get('model', 'openai/whisper-tiny')
        self.language = config.get('language', 'en')
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        
        # Audio device (None = default, or specify device index)
        self.input_device = config.get('input_device', None)
        
        # Initialize engine
        self._model = None
        self._init_engine()
        
        # PyAudio
        self.audio = None
        
        # Find USB microphone if not specified
        if self.input_device is None:
            self.input_device = self._find_usb_microphone()
        
    def _init_engine(self):
        """Initialize the selected STT engine"""
        if self.engine == "whisper" and WHISPER_AVAILABLE:
            self._init_whisper()
        elif self.engine == "moonshine" and MOONSHINE_AVAILABLE:
            self._init_moonshine()
        elif self.engine == "vosk" and VOSK_AVAILABLE:
            self._init_vosk()
        else:
            # Fallback
            if WHISPER_AVAILABLE:
                self.engine = "whisper"
                self._init_whisper()
            elif VOSK_AVAILABLE:
                self.engine = "vosk"
                self._init_vosk()
            else:
                raise RuntimeError("No STT engine available!")
                
    def _init_whisper(self):
        """Initialize Whisper via transformers"""
        logger.info(f"Loading Whisper model: {self.model_name}")
        
        try:
            self._model = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device="cpu"  # Pi5 uses CPU
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise
            
    def _init_moonshine(self):
        """Initialize Moonshine STT"""
        logger.info("Initializing Moonshine STT")
        try:
            self._model = MoonshineSTT(model_name="moonshine/tiny")
            logger.info("Moonshine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Moonshine: {e}")
            raise
            
    def _init_vosk(self):
        """Initialize Vosk STT"""
        # Check environment variable first, then common locations
        model_path = os.environ.get('VOSK_MODEL_PATH')
        
        if not model_path or not Path(model_path).exists():
            # Try common locations
            search_paths = [
                Path.home() / "models" / "vosk-model-small-en-us-0.15",
                Path(__file__).parent.parent / "models" / "vosk-model-small-en-us-0.15",
                Path("/usr/share/vosk-models/vosk-model-small-en-us-0.15"),
            ]
            for p in search_paths:
                if p.exists():
                    model_path = p
                    break
            else:
                model_path = search_paths[0]  # Default to home directory
            
        logger.info(f"Loading Vosk model: {model_path}")
        
        try:
            self._vosk_model = VoskModel(str(model_path))
            logger.info("Vosk model loaded")
        except Exception as e:
            logger.error(f"Failed to load Vosk: {e}")
            raise
            
    def _find_usb_microphone(self) -> Optional[int]:
        """
        Find USB microphone device.
        
        Scans audio devices for USB audio input devices.
        Returns device index or None for default.
        """
        if not PYAUDIO_AVAILABLE:
            return None
            
        try:
            audio = pyaudio.PyAudio()
            
            usb_keywords = ['usb', 'USB', 'Headset', 'headset', 'Microphone', 'microphone']
            
            for i in range(audio.get_device_count()):
                try:
                    info = audio.get_device_info_by_index(i)
                    name = info.get('name', '')
                    max_inputs = info.get('maxInputChannels', 0)
                    
                    # Must have input channels
                    if max_inputs > 0:
                        # Check if it's a USB device
                        if any(kw in name for kw in usb_keywords):
                            logger.info(f"Found USB microphone: '{name}' (index {i})")
                            audio.terminate()
                            return i
                            
                except Exception:
                    continue
                    
            # If no USB device found, try to find any non-default input
            default_input = audio.get_default_input_device_info()
            logger.info(f"Using default input device: '{default_input.get('name', 'Unknown')}'")
            audio.terminate()
            return None
            
        except Exception as e:
            logger.error(f"Error scanning audio devices: {e}")
            return None
            
    @staticmethod
    def list_audio_devices() -> list:
        """
        List all available audio input devices.
        
        Returns:
            List of dicts with device info
        """
        if not PYAUDIO_AVAILABLE:
            return []
            
        devices = []
        audio = pyaudio.PyAudio()
        
        for i in range(audio.get_device_count()):
            try:
                info = audio.get_device_info_by_index(i)
                if info.get('maxInputChannels', 0) > 0:
                    devices.append({
                        'index': i,
                        'name': info.get('name', 'Unknown'),
                        'channels': info.get('maxInputChannels'),
                        'sample_rate': int(info.get('defaultSampleRate', 16000)),
                        'is_default': i == audio.get_default_input_device_info().get('index')
                    })
            except Exception:
                continue
                
        audio.terminate()
        return devices
            
    def record_until_silence(self, 
                            silence_threshold: int = 500,
                            silence_duration: float = 1.5,
                            max_duration: float = 30.0) -> Optional[np.ndarray]:
        """
        Record audio until silence is detected.
        
        Args:
            silence_threshold: RMS energy threshold for silence detection
            silence_duration: Seconds of silence to stop recording
            max_duration: Maximum recording duration
            
        Returns:
            Audio data as numpy array, or None if no speech
        """
        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio not available for recording")
            return None
            
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
            
        # Open stream with specific device if found
        stream_kwargs = {
            'format': pyaudio.paInt16,
            'channels': self.channels,
            'rate': self.sample_rate,
            'input': True,
            'frames_per_buffer': self.chunk_size
        }
        
        if self.input_device is not None:
            stream_kwargs['input_device_index'] = self.input_device
            logger.info(f"Using audio input device index: {self.input_device}")
            
        stream = self.audio.open(**stream_kwargs)
        
        logger.info("Recording started...")
        
        frames = []
        silence_start = None
        speech_detected = False
        start_time = time.time()
        
        try:
            while True:
                # Check max duration
                if time.time() - start_time > max_duration:
                    logger.warning("Max recording duration reached")
                    break
                    
                # Read audio chunk
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # Calculate RMS energy
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                
                if rms > silence_threshold:
                    # Speech detected
                    speech_detected = True
                    silence_start = None
                else:
                    # Silence
                    if speech_detected:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > silence_duration:
                            logger.info("Silence detected, stopping recording")
                            break
                            
        except Exception as e:
            logger.error(f"Recording error: {e}")
            
        finally:
            stream.stop_stream()
            stream.close()
            
        if not speech_detected or len(frames) < 5:
            logger.warning("No speech detected in recording")
            return None
            
        # Convert to numpy array
        audio_bytes = b''.join(frames)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        logger.info(f"Recorded {len(audio_array) / self.sample_rate:.1f} seconds")
        
        return audio_array
        
    def transcribe(self, audio: Union[np.ndarray, str, bytes]) -> Optional[str]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array, file path, or bytes
            
        Returns:
            Transcribed text, or None on failure
        """
        if audio is None:
            return None
            
        try:
            if self.engine == "whisper":
                return self._transcribe_whisper(audio)
            elif self.engine == "moonshine":
                return self._transcribe_moonshine(audio)
            elif self.engine == "vosk":
                return self._transcribe_vosk(audio)
            else:
                logger.error(f"Unknown engine: {self.engine}")
                return None
                
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return None
            
    def _transcribe_whisper(self, audio: Union[np.ndarray, str]) -> str:
        """Transcribe using Whisper"""
        if isinstance(audio, np.ndarray):
            # Normalize to float32 [-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
                
            result = self._model(
                {"raw": audio, "sampling_rate": self.sample_rate},
                return_timestamps=False
            )
        else:
            # File path
            result = self._model(audio, return_timestamps=False)
            
        text = result.get("text", "").strip()
        logger.info(f"Whisper transcription: {text}")
        return text
        
    def _transcribe_moonshine(self, audio: np.ndarray) -> str:
        """Transcribe using Moonshine"""
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
            
        text = self._model.transcribe(audio)
        logger.info(f"Moonshine transcription: {text}")
        return text.strip()
        
    def _transcribe_vosk(self, audio: np.ndarray) -> str:
        """Transcribe using Vosk"""
        recognizer = KaldiRecognizer(self._vosk_model, self.sample_rate)
        
        # Convert to bytes
        if audio.dtype != np.int16:
            audio = (audio * 32768).astype(np.int16)
        audio_bytes = audio.tobytes()
        
        # Process in chunks
        chunk_size = 4000
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            recognizer.AcceptWaveform(chunk)
            
        result = json.loads(recognizer.FinalResult())
        text = result.get("text", "").strip()
        
        logger.info(f"Vosk transcription: {text}")
        return text


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing Speech-to-Text...")
    print("Speak something when prompted...")
    
    stt = SpeechToText(config={
        'engine': 'whisper',
        'model': 'openai/whisper-tiny'
    })
    
    print("\n>>> Recording... speak now! <<<")
    audio = stt.record_until_silence(silence_duration=1.5)
    
    if audio is not None:
        print("Transcribing...")
        text = stt.transcribe(audio)
        print(f"\nYou said: {text}")
    else:
        print("No speech detected")
