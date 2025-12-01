#!/usr/bin/env python3
"""
Wake Word Detection for beast
==============================
Listens continuously for the wake word "Beast" to trigger voice interaction.

Supports multiple backends:
- whisper - uses Faster Whisper for accurate word detection (recommended)
- pvporcupine (Picovoice Porcupine) - dedicated wake word
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
from scipy import signal as scipy_signal

logger = logging.getLogger('beast.WakeWord')

# Try to import wake word engines
PORCUPINE_AVAILABLE = False
VOSK_AVAILABLE = False
WHISPER_AVAILABLE = False

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

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
    logger.info("Faster Whisper wake word engine available")
except ImportError:
    logger.warning("Faster Whisper not available")

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
                 engine: str = "auto",
                 input_device: int = None):
        """
        Initialize wake word detector.
        
        Args:
            wake_word: The word to listen for
            sensitivity: Detection sensitivity (0.0-1.0)
            engine: "porcupine", "vosk", "simple", or "auto"
            input_device: Audio input device index (None = auto-detect)
        """
        self.wake_word = wake_word.lower()
        self.sensitivity = sensitivity
        self.running = False
        self.audio = None
        self.stream = None
        self.input_device = input_device  # Can be explicitly set or will be auto-detected
        
        # Audio parameters (defaults, will be updated by _find_usb_microphone)
        self.sample_rate = 16000  # Target sample rate for processing
        self.hardware_sample_rate = 16000  # Hardware recording rate (auto-detected)
        self.channels = 1  # Mono by default (auto-detected)
        self.frame_length = 512  # ~32ms at 16kHz
        self.hardware_frame_length = self.frame_length  # Will be adjusted if resampling needed
        
        # Find USB microphone and detect its capabilities (only if not explicitly set)
        if self.input_device is None:
            self._find_usb_microphone()
        else:
            # Device explicitly set - detect its capabilities
            self._detect_device_capabilities(self.input_device)
        
        # Select engine
        if engine == "auto":
            if WHISPER_AVAILABLE:
                engine = "whisper"
            elif PORCUPINE_AVAILABLE:
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
        if self.engine == "whisper":
            self._init_whisper()
        elif self.engine == "porcupine":
            self._init_porcupine()
        elif self.engine == "vosk":
            self._init_vosk()
        else:
            self._init_simple()
            
    def _init_whisper(self):
        """Initialize Faster Whisper for wake word detection"""
        try:
            from faster_whisper import WhisperModel
            
            # Use tiny model for fast wake word detection
            self._detector = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            
            # Audio buffer settings for wake word detection
            self._audio_buffer = []
            self._buffer_duration = 2.0  # seconds of audio to accumulate
            self._energy_threshold = 1400  # Require voice activity before transcribing
            self._check_interval = 0.5  # Check every 0.5 seconds
            self._last_check_time = time.time()
            
            logger.info("Faster Whisper wake word detector initialized (tiny.en)")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper wake word: {e}")
            logger.info("Falling back to simple detector")
            self.engine = "simple"
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
        # Higher threshold = less sensitive (requires louder speech)
        # Derive threshold from sensitivity (0.0-1.0)
        # sensitivity=1.0 -> lower threshold (more sensitive)
        # sensitivity=0.0 -> higher threshold (less sensitive)
        base_min = 600
        base_max = 2200
        # Clamp sensitivity
        s = max(0.0, min(1.0, float(self.sensitivity)))
        self._energy_threshold = int(base_max - (base_max - base_min) * s)
        self._consecutive_frames = 8  # Require multiple loud frames in a row (increased to reduce false triggers)
        self._loud_frame_count = 0
        logger.info(f"Using simple energy-based voice detection (threshold: {self._energy_threshold}, sensitivity: {s})")
    
    def _find_usb_microphone(self):
        """Find USB microphone device index and detect its capabilities"""
        if not PYAUDIO_AVAILABLE:
            return
            
        try:
            audio = pyaudio.PyAudio()
            usb_keywords = ['usb', 'USB', 'Headset', 'headset', 'Microphone', 'microphone', 'JLAB', 'Logi']
            
            for i in range(audio.get_device_count()):
                try:
                    info = audio.get_device_info_by_index(i)
                    name = info.get('name', '')
                    max_inputs = info.get('maxInputChannels', 0)
                    default_rate = int(info.get('defaultSampleRate', 44100))
                    
                    if max_inputs > 0:
                        if any(kw in name for kw in usb_keywords):
                            logger.info(f"Found USB microphone: '{name}' (index {i})")
                            logger.info(f"  Channels: {max_inputs}, Default rate: {default_rate}")
                            self.input_device = i
                            
                            # Auto-detect capabilities
                            # Logitech headsets support 16kHz mono directly
                            # JLAB requires 44100Hz stereo
                            if 'Logi' in name or 'logi' in name:
                                # Logitech supports 16kHz mono - optimal!
                                self.hardware_sample_rate = 16000
                                self.channels = 1
                                self.hardware_frame_length = self.frame_length
                                logger.info(f"  Using direct 16kHz mono (no resampling needed)")
                            elif max_inputs == 1:
                                # Mono mic - try to detect supported sample rate
                                self.channels = 1
                                # Test common sample rates in order of preference
                                for test_rate in [16000, 48000, 44100, 32000]:
                                    try:
                                        test_stream = audio.open(
                                            format=pyaudio.paInt16,
                                            channels=1,
                                            rate=test_rate,
                                            input=True,
                                            input_device_index=i,
                                            frames_per_buffer=512,
                                            start=False
                                        )
                                        test_stream.close()
                                        self.hardware_sample_rate = test_rate
                                        if test_rate == 16000:
                                            self.hardware_frame_length = self.frame_length
                                            logger.info(f"  Using direct 16kHz mono (no resampling)")
                                        else:
                                            self.hardware_frame_length = int(self.frame_length * test_rate / 16000)
                                            logger.info(f"  Using {test_rate}Hz mono with resampling to 16kHz")
                                        break
                                    except Exception as e:
                                        logger.debug(f"  {test_rate}Hz not supported: {e}")
                                        continue
                                else:
                                    # Fallback if no rate worked
                                    self.hardware_sample_rate = default_rate
                                    self.hardware_frame_length = int(self.frame_length * default_rate / 16000)
                                    logger.info(f"  Using {default_rate}Hz mono with resampling (fallback)")
                            else:
                                # Stereo mic (like JLAB)
                                self.channels = 2
                                self.hardware_sample_rate = 44100
                                self.hardware_frame_length = int(self.frame_length * 44100 / 16000)
                                logger.info(f"  Using 44100Hz stereo with resampling")
                            
                            audio.terminate()
                            return
                except Exception as e:
                    logger.debug(f"Error checking device {i}: {e}")
                    continue
            
            audio.terminate()
            logger.warning("No USB microphone found, using default")
        except Exception as e:
            logger.error(f"Error scanning audio devices: {e}")
    
    def _detect_device_capabilities(self, device_index: int):
        """Detect capabilities of a specific audio device"""
        if not PYAUDIO_AVAILABLE:
            return
        
        try:
            audio = pyaudio.PyAudio()
            info = audio.get_device_info_by_index(device_index)
            name = info.get('name', 'Unknown')
            max_inputs = info.get('maxInputChannels', 0)
            default_rate = int(info.get('defaultSampleRate', 44100))
            
            logger.info(f"Using device {device_index}: '{name}' ({max_inputs}ch, {default_rate}Hz)")
            
            if max_inputs == 0:
                logger.warning(f"Device {device_index} has no input channels!")
                audio.terminate()
                return
            
            self.channels = min(max_inputs, 1)  # Use mono if available
            
            # Test common sample rates
            for test_rate in [16000, 48000, 44100, 32000]:
                try:
                    test_stream = audio.open(
                        format=pyaudio.paInt16,
                        channels=self.channels,
                        rate=test_rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=512,
                        start=False
                    )
                    test_stream.close()
                    self.hardware_sample_rate = test_rate
                    if test_rate == 16000:
                        self.hardware_frame_length = self.frame_length
                        logger.info(f"  Using {test_rate}Hz (no resampling needed)")
                    else:
                        self.hardware_frame_length = int(self.frame_length * test_rate / 16000)
                        logger.info(f"  Using {test_rate}Hz with resampling to 16kHz")
                    break
                except Exception as e:
                    logger.debug(f"  {test_rate}Hz not supported: {e}")
                    continue
            else:
                # Fallback
                self.hardware_sample_rate = default_rate
                self.hardware_frame_length = int(self.frame_length * default_rate / 16000)
                logger.info(f"  Using {default_rate}Hz (fallback)")
            
            audio.terminate()
        except Exception as e:
            logger.error(f"Error detecting device capabilities: {e}")
        
    def _init_audio(self):
        """Initialize PyAudio stream"""
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio not available")
            
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
            
        if self.stream is None or not self.stream.is_active():
            # Use auto-detected sample rate and channels
            stream_kwargs = {
                'rate': self.hardware_sample_rate,
                'channels': self.channels,
                'format': pyaudio.paInt16,
                'input': True,
                'frames_per_buffer': self.hardware_frame_length
            }
            
            if self.input_device is not None:
                stream_kwargs['input_device_index'] = self.input_device
                logger.info(f"Using audio device {self.input_device}: {self.hardware_sample_rate}Hz, {self.channels}ch")
            
            self.stream = self.audio.open(**stream_kwargs)
            
    def _resample_audio(self, pcm_data):
        """Resample audio from hardware rate to processing rate"""
        # Convert to numpy array if not already
        if isinstance(pcm_data, np.ndarray):
            audio_np = pcm_data.astype(np.float32)
        else:
            audio_np = np.array(pcm_data, dtype=np.float32)
        # Resample from 44100 to 16000 Hz
        num_samples = int(len(audio_np) * self.sample_rate / self.hardware_sample_rate)
        resampled = scipy_signal.resample(audio_np, num_samples)
        return resampled.astype(np.int16)
            
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
                pcm = self.stream.read(self.hardware_frame_length, exception_on_overflow=False)
                
                # Handle based on channels and sample rate
                if self.channels == 2:
                    # Stereo: unpack 2x samples (left + right interleaved)
                    pcm_data = struct.unpack_from(f"{self.hardware_frame_length * 2}h", pcm)
                    # Convert stereo to mono by averaging channels
                    pcm_array = np.array(pcm_data, dtype=np.float32)
                    pcm_mono = (pcm_array[0::2] + pcm_array[1::2]) / 2
                else:
                    # Mono: unpack directly
                    pcm_data = struct.unpack_from(f"{self.hardware_frame_length}h", pcm)
                    pcm_mono = np.array(pcm_data, dtype=np.float32)
                
                # Resample if needed (skip if already at 16kHz)
                if self.hardware_sample_rate != self.sample_rate:
                    pcm_data = self._resample_audio(pcm_mono)
                else:
                    pcm_data = pcm_mono.astype(np.int16)
                
                # Check for wake word based on engine
                if self._check_wake_word(pcm_data):
                    return True
                    
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            return False
            
        return False
        
    def _check_wake_word(self, pcm_data) -> bool:
        """Check if wake word is present in audio frame"""
        if self.engine == "whisper" and self._detector:
            # Accumulate audio in buffer
            self._audio_buffer.extend(pcm_data)
            
            # Check if enough time has passed and we have enough audio
            current_time = time.time()
            buffer_samples = len(self._audio_buffer)
            buffer_seconds = buffer_samples / self.sample_rate
            
            if current_time - self._last_check_time >= self._check_interval and buffer_seconds >= 1.0:
                # Calculate energy to see if there's voice activity
                audio_array = np.array(self._audio_buffer, dtype=np.float32)
                energy = np.sqrt(np.mean(audio_array ** 2))
                
                if energy > self._energy_threshold:
                    # There's voice activity, transcribe it
                    try:
                        # Convert to float32 normalized audio for Whisper
                        audio_float = audio_array / 32768.0
                        
                        # Transcribe the buffer
                        segments, info = self._detector.transcribe(
                            audio_float,
                            language="en",
                            beam_size=1,
                            best_of=1,
                            temperature=0.0,
                            vad_filter=False
                        )
                        
                        # Check if wake word is in transcription
                        text = " ".join([segment.text for segment in segments]).lower().strip()
                        
                        if self.wake_word in text:
                            logger.debug(f"Wake word detected in: '{text}'")
                            self._audio_buffer = []  # Clear buffer
                            self._last_check_time = current_time
                            return True
                    except Exception as e:
                        logger.debug(f"Whisper transcription error: {e}")
                
                # Update last check time and trim buffer to keep last 2 seconds
                self._last_check_time = current_time
                max_buffer_samples = int(self._buffer_duration * self.sample_rate)
                if buffer_samples > max_buffer_samples:
                    self._audio_buffer = self._audio_buffer[-max_buffer_samples:]
            
            return False
        
        elif self.engine == "porcupine" and self._detector:
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
                # Count consecutive loud frames to avoid noise triggers
                self._loud_frame_count += 1
                if self._loud_frame_count >= self._consecutive_frames:
                    # Sustained loud audio detected - treat as wake word
                    self._loud_frame_count = 0
                    logger.debug(f"Wake triggered (energy: {energy:.0f})")
                    return True
            else:
                # Reset counter on quiet frame
                self._loud_frame_count = 0
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
