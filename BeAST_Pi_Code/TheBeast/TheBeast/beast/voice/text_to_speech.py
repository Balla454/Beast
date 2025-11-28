#!/usr/bin/env python3
"""
Text-to-Speech for BeAST
========================
Converts text to spoken audio using local TTS engines.

Supports:
- Piper TTS - recommended for quality + speed on Pi
- eSpeak - lightweight fallback
- Transformers TTS - neural TTS (slower)
"""

import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger('BeAST.TTS')

# Try to import TTS engines
PIPER_AVAILABLE = False
ESPEAK_AVAILABLE = False
TRANSFORMERS_TTS_AVAILABLE = False

try:
    # Check if piper-tts is installed
    result = subprocess.run(['piper', '--help'], capture_output=True, text=True)
    PIPER_AVAILABLE = True
    logger.info("Piper TTS available")
except (FileNotFoundError, subprocess.SubprocessError):
    logger.warning("Piper TTS not available (pip install piper-tts)")

try:
    # Check if espeak is installed
    result = subprocess.run(['espeak', '--version'], capture_output=True, text=True)
    ESPEAK_AVAILABLE = True
    logger.info("eSpeak TTS available")
except (FileNotFoundError, subprocess.SubprocessError):
    logger.warning("eSpeak not available (apt install espeak)")

try:
    from transformers import pipeline
    TRANSFORMERS_TTS_AVAILABLE = True
except ImportError:
    pass

# Audio playback
try:
    import sounddevice as sd
    import soundfile as sf
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice not available (pip install sounddevice soundfile)")


class TextToSpeech:
    """
    Text-to-speech engine for BeAST.
    
    Usage:
        tts = TextToSpeech(config={'engine': 'piper'})
        tts.speak("Hello, I am BeAST")
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize TTS engine.
        
        Args:
            config: Configuration dict with engine, model settings
        """
        config = config or {}
        self.engine = config.get('engine', 'piper')
        self.model = config.get('model', 'en_US-lessac-medium')
        self.rate = config.get('rate', 1.0)
        
        # Model path for Piper - check env var, then relative to module
        default_model_path = os.environ.get(
            'PIPER_MODEL_PATH',
            str(Path(__file__).parent.parent / 'models' / 'tts')
        )
        self.model_path = config.get('model_path', default_model_path)
        
        # Audio output device (None = default, or specify device index)
        self.output_device = config.get('output_device', None)
        
        # Find USB speaker if not specified
        if self.output_device is None:
            self.output_device = self._find_usb_speaker()
        
        # Validate engine availability
        self._validate_engine()
        
        logger.info(f"TTS initialized with engine: {self.engine}")
        
    def _validate_engine(self):
        """Validate and fallback to available engine"""
        if self.engine == "piper" and not PIPER_AVAILABLE:
            logger.warning("Piper not available, falling back")
            if ESPEAK_AVAILABLE:
                self.engine = "espeak"
            else:
                self.engine = "system"
                
        elif self.engine == "espeak" and not ESPEAK_AVAILABLE:
            logger.warning("eSpeak not available, falling back")
            self.engine = "system"
            
    def _find_usb_speaker(self) -> Optional[int]:
        """
        Find USB speaker/audio output device.
        
        Scans audio devices for USB audio output devices.
        Returns device index or None for default.
        """
        if not SOUNDDEVICE_AVAILABLE:
            return None
            
        try:
            devices = sd.query_devices()
            
            usb_keywords = ['usb', 'USB', 'Headset', 'headset', 'Speaker', 'speaker', 'Headphone', 'headphone']
            
            for i, dev in enumerate(devices):
                name = dev.get('name', '')
                max_outputs = dev.get('max_output_channels', 0)
                
                # Must have output channels
                if max_outputs > 0:
                    # Check if it's a USB device
                    if any(kw in name for kw in usb_keywords):
                        logger.info(f"Found USB speaker: '{name}' (index {i})")
                        return i
                        
            # If no USB device found, use default
            default_output = sd.query_devices(kind='output')
            logger.info(f"Using default output device: '{default_output.get('name', 'Unknown')}'")
            return None
            
        except Exception as e:
            logger.error(f"Error scanning audio output devices: {e}")
            return None
            
    @staticmethod
    def list_audio_devices() -> list:
        """
        List all available audio output devices.
        
        Returns:
            List of dicts with device info
        """
        if not SOUNDDEVICE_AVAILABLE:
            return []
            
        devices = []
        all_devs = sd.query_devices()
        default_output = sd.query_devices(kind='output')
        default_idx = default_output.get('index') if default_output else None
        
        for i, dev in enumerate(all_devs):
            if dev.get('max_output_channels', 0) > 0:
                devices.append({
                    'index': i,
                    'name': dev.get('name', 'Unknown'),
                    'channels': dev.get('max_output_channels'),
                    'sample_rate': int(dev.get('default_samplerate', 44100)),
                    'is_default': i == default_idx
                })
                
        return devices
            
    def speak(self, text: str, blocking: bool = True):
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            blocking: Wait for speech to complete
        """
        if not text or not text.strip():
            return
            
        # Clean text for TTS
        text = self._clean_text(text)
        
        logger.info(f"Speaking: {text[:50]}...")
        
        try:
            if self.engine == "piper":
                self._speak_piper(text, blocking)
            elif self.engine == "espeak":
                self._speak_espeak(text, blocking)
            elif self.engine == "transformers":
                self._speak_transformers(text, blocking)
            else:
                self._speak_system(text, blocking)
                
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            # Try fallback
            try:
                self._speak_system(text, blocking)
            except:
                logger.error("All TTS methods failed")
                
    def _clean_text(self, text: str) -> str:
        """Clean text for TTS"""
        # Remove markdown formatting
        text = text.replace("**", "").replace("*", "")
        text = text.replace("#", "").replace("`", "")
        text = text.replace("\n\n", ". ").replace("\n", " ")
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        return text.strip()
        
    def _speak_piper(self, text: str, blocking: bool = True):
        """Speak using Piper TTS"""
        # Find model file
        model_file = self._find_piper_model()
        
        if not model_file:
            logger.warning("Piper model not found, falling back to espeak")
            self._speak_espeak(text, blocking)
            return
            
        # Generate audio with piper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            
        try:
            # Run piper
            cmd = [
                'piper',
                '--model', str(model_file),
                '--output_file', temp_path
            ]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = process.communicate(input=text.encode('utf-8'))
            
            if process.returncode != 0:
                logger.error(f"Piper error: {stderr.decode()}")
                raise RuntimeError("Piper failed")
                
            # Play the audio
            self._play_audio_file(temp_path, blocking)
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def _find_piper_model(self) -> Optional[Path]:
        """Find Piper model file"""
        # Check common locations (no hardcoded paths)
        search_paths = [
            Path(self.model_path) / f"{self.model}.onnx",
            Path.home() / "models" / "piper" / f"{self.model}.onnx",
            Path(__file__).parent.parent / "models" / "tts" / f"{self.model}.onnx",
            Path("/usr/share/piper-voices") / f"{self.model}.onnx",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
                
        # Try to find any .onnx file in model directories
        for base in [Path(self.model_path), Path.home() / "models" / "piper"]:
            if base.exists():
                models = list(base.glob("*.onnx"))
                if models:
                    return models[0]
                    
        return None
        
    def _speak_espeak(self, text: str, blocking: bool = True):
        """Speak using eSpeak"""
        cmd = ['espeak', '-v', 'en', '-s', str(int(150 * self.rate)), text]
        
        # Run with a timeout to prevent hanging
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            logger.debug(f"espeak completed with return code {result.returncode}")
        except subprocess.TimeoutExpired:
            logger.warning("espeak timed out after 60 seconds")
        except Exception as e:
            logger.error(f"espeak error: {e}")
            
    def _speak_transformers(self, text: str, blocking: bool = True):
        """Speak using transformers TTS pipeline"""
        if not hasattr(self, '_tts_pipeline'):
            from transformers import pipeline
            self._tts_pipeline = pipeline(
                "text-to-speech",
                model="microsoft/speecht5_tts"
            )
            
        # Generate audio
        result = self._tts_pipeline(text)
        
        # Play using sounddevice
        if SOUNDDEVICE_AVAILABLE:
            audio = result["audio"]
            sr = result["sampling_rate"]
            sd.play(audio, sr)
            if blocking:
                sd.wait()
                
    def _speak_system(self, text: str, blocking: bool = True):
        """Speak using system TTS (macOS/Linux)"""
        import platform
        
        system = platform.system()
        
        if system == "Darwin":  # macOS
            cmd = ['say', text]
        elif system == "Linux":
            # Try various Linux TTS commands
            if ESPEAK_AVAILABLE:
                cmd = ['espeak', text]
            else:
                # Last resort: festival
                cmd = ['festival', '--tts']
        else:
            logger.error(f"No system TTS for {system}")
            return
            
        if blocking:
            subprocess.run(cmd, capture_output=True)
        else:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
    def _play_audio_file(self, path: str, blocking: bool = True):
        """Play an audio file"""
        if SOUNDDEVICE_AVAILABLE:
            data, sr = sf.read(path)
            # Use specified output device if available
            if self.output_device is not None:
                sd.play(data, sr, device=self.output_device)
            else:
                sd.play(data, sr)
            if blocking:
                sd.wait()
        else:
            # Fall back to aplay (Linux) or afplay (macOS)
            import platform
            
            if platform.system() == "Darwin":
                cmd = ['afplay', path]
            else:
                cmd = ['aplay', '-q', path]
                
            if blocking:
                subprocess.run(cmd, capture_output=True)
            else:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                

# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing Text-to-Speech...")
    
    tts = TextToSpeech(config={'engine': 'piper'})
    
    test_texts = [
        "Hello, I am BeAST, your personal health assistant.",
        "Your current heart rate is 72 beats per minute.",
        "Based on your data, you appear to be well rested today."
    ]
    
    for text in test_texts:
        print(f"\nSpeaking: {text}")
        tts.speak(text)
        print("Done")
