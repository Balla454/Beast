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
        """Clean text for TTS - remove special symbols that cause issues"""
        import re
        
        # Remove markdown formatting
        text = text.replace("**", "").replace("*", "")
        text = text.replace("#", "").replace("`", "")
        text = text.replace("\n\n", ". ").replace("\n", " ")
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove emojis and special unicode characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Remove special symbols that cause TTS issues
        text = re.sub(r'[<>{}[\]|\\^~@#$%&*+=]', ' ', text)
        
        # Replace common symbols with words
        text = text.replace("&", " and ")
        text = text.replace("%", " percent ")
        text = text.replace("+", " plus ")
        text = text.replace("=", " equals ")
        
        # Keep basic punctuation: . , ! ? ' " - : ;
        # Remove other punctuation
        text = re.sub(r'[^\w\s.,!?\'\"-:;()]', ' ', text)
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Limit length to prevent very long responses
        if len(text) > 500:
            text = text[:500] + "..."
        
        return text.strip()
        
    def _speak_piper(self, text: str, blocking: bool = True):
        """Speak using Piper TTS"""
        # Find model file
        model_file = self._find_piper_model()
        
        if not model_file:
            logger.warning("Piper model not found, falling back to espeak")
            self._speak_espeak(text, blocking)
            return
        
        try:
            # Generate audio with Piper to a temp file
            import tempfile
            import os
            
            logger.info("Creating temp file for TTS...")
            tmp_path = tempfile.mktemp(suffix='.wav')
            logger.info(f"Temp file: {tmp_path}")
            
            piper_cmd = [
                'piper',
                '--model', str(model_file),
                '--output_file', tmp_path
            ]
            
            logger.info(f"Running piper command: {' '.join(piper_cmd)}")
            
            # Use subprocess.run instead of Popen for simpler handling
            try:
                result = subprocess.run(
                    piper_cmd,
                    input=text.encode('utf-8'),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=30
                )
                if result.returncode != 0:
                    logger.warning(f"Piper returned code {result.returncode}: {result.stderr.decode()}")
            except subprocess.TimeoutExpired:
                logger.error("Piper generation timed out")
                raise
            
            logger.info(f"Speech generated, playing...")
            
            # Try to play on available devices
            # Get list of devices
            import re
            try:
                result = subprocess.run(['aplay', '-l'], capture_output=True, text=True, timeout=2)
                devices = []
                for line in result.stdout.split('\n'):
                    match = re.search(r'card (\d+).*device (\d+)', line)
                    if match:
                        # Use plughw instead of hw for automatic format conversion
                        devices.append(f"plughw:{match.group(1)},{match.group(2)}")
                
                if not devices:
                    devices = ['default']
            except:
                devices = ['default']
            
            logger.info(f"Trying {len(devices)} device(s)")
            
            # Try each device until one works
            played = False
            for device in devices:
                try:
                    logger.info(f"Attempting playback on {device}")
                    aplay_cmd = ['aplay', '-D', device, '-q', tmp_path]
                    result = subprocess.run(
                        aplay_cmd,
                        timeout=10,
                        capture_output=True
                    )
                    if result.returncode == 0:
                        logger.info(f"Successfully played on {device}")
                        played = True
                        break
                    else:
                        logger.info(f"Failed on {device}: {result.stderr}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Timeout playing on {device}")
                except Exception as e:
                    logger.info(f"Error playing on {device}: {e}")
            
            if not played:
                logger.warning("Failed to play on any device, trying default")
                subprocess.run(['aplay', tmp_path], timeout=10)
            
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            logger.info("Piper TTS completed")
            return  # Explicit return after successful completion
            
        except subprocess.TimeoutExpired:
            logger.warning("Piper TTS timed out")
            if 'piper_proc' in locals():
                try:
                    piper_proc.kill()
                except:
                    pass
            return
        except Exception as e:
            logger.error(f"Piper error: {e}", exc_info=True)
            # Fallback to espeak
            self._speak_espeak(text, blocking)
            return
                
    def _find_piper_model(self) -> Optional[Path]:
        """Find Piper model file"""
        # Get BEAST_HOME directory (where main.py is)
        beast_home = Path(__file__).parent.parent  # /beast/voice -> /beast
        project_root = beast_home.parent  # /beast -> /TheBeast
        
        # Check common locations
        search_paths = [
            # Project root models directory
            project_root / "models" / "tts" / f"{self.model}.onnx",
            # Relative to BEAST_HOME (from config)
            beast_home / self.model_path / f"{self.model}.onnx",
            # Direct path if model_path is absolute
            Path(self.model_path) / f"{self.model}.onnx",
            # Home directory
            Path.home() / "models" / "piper" / f"{self.model}.onnx",
            # System path
            Path("/usr/share/piper-voices") / f"{self.model}.onnx",
        ]
        
        logger.debug(f"Searching for Piper model: {self.model}")
        for path in search_paths:
            logger.debug(f"  Checking: {path}")
            if path.exists():
                logger.info(f"Found Piper model: {path}")
                return path
                
        # Try to find any .onnx file in model directories
        for base in [project_root / "models" / "tts", beast_home / "models" / "tts", Path(self.model_path)]:
            if base.exists():
                models = list(base.glob("*.onnx"))
                if models:
                    logger.info(f"Found Piper model: {models[0]}")
                    return models[0]
        
        logger.warning(f"Piper model not found. Searched: {[str(p) for p in search_paths]}")
        return None
        
    def _speak_espeak(self, text: str, blocking: bool = True):
        """Speak using eSpeak with ALSA device selection"""
        try:
            # Use default device from ~/.asoundrc (multi_out)
            espeak_cmd = ['espeak', '-v', 'en', '-s', str(int(150 * self.rate)), '--stdout', text]
            aplay_cmd = ['aplay']
            
            # Run espeak and pipe to aplay
            espeak_proc = subprocess.Popen(
                espeak_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            aplay_proc = subprocess.Popen(
                aplay_cmd,
                stdin=espeak_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            espeak_proc.stdout.close()
            
            if blocking:
                aplay_proc.wait(timeout=30)
            
        except subprocess.TimeoutExpired:
            logger.warning("eSpeak TTS timed out")
            aplay_proc.kill()
            espeak_proc.kill()
        except Exception as e:
            logger.error(f"eSpeak error: {e}")
            # Final fallback to system TTS
            self._speak_system(text, blocking)
            espeak_proc = subprocess.Popen(espeak_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            aplay_proc = subprocess.Popen(aplay_cmd, stdin=espeak_proc.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            espeak_proc.stdout.close()
            
            if blocking:
                aplay_proc.wait(timeout=60)
                stderr = aplay_proc.stderr.read().decode() if aplay_proc.stderr else ""
                if aplay_proc.returncode != 0:
                    logger.warning(f"aplay returned {aplay_proc.returncode}: {stderr}")
            
            logger.debug(f"espeak+aplay completed")
        except subprocess.TimeoutExpired:
            logger.warning("espeak timed out after 60 seconds")
        except Exception as e:
            logger.warning(f"espeak+aplay failed ({e}), trying fallback")
            # Fallback: try default device
            try:
                cmd = ['espeak', '-v', 'en', '-s', str(int(150 * self.rate)), text]
                subprocess.run(cmd, capture_output=True, timeout=60)
            except Exception as e2:
                logger.error(f"espeak error: {e2}")
            
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
