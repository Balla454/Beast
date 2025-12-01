#!/usr/bin/env python3
"""
Announcer Utility for beast
===========================
Handles system announcements and audio feedback.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger('beast.Announcer')


class Announcer:
    """
    Handles audio announcements and feedback sounds.
    
    Usage:
        announcer = Announcer(tts_engine)
        announcer.announce("beast is ready")
        announcer.play_sound("listening")
    """
    
    # Standard announcements
    ANNOUNCEMENTS = {
        'startup': "beast system starting up. Please wait.",
        'ready': "beast is ready. Say Beast to ask a question.",
        'shutdown': "beast shutting down. Goodbye.",
        'error': "An error occurred. Please check the logs.",
        'listening': "I'm listening.",
        'processing': "Processing your request.",
        'no_speech': "I didn't hear anything. Please try again.",
        'not_understood': "I didn't understand that. Please try again.",
        'calibrating': "Calibrating sensors. Please remain still.",
        'calibration_done': "Calibration complete.",
        'low_battery': "Battery is low. Please charge soon.",
        'syncing': "Syncing data to host computer.",
        'sync_complete': "Data sync complete."
    }
    
    def __init__(self, tts_engine=None, sounds_dir: str = None):
        """
        Initialize announcer.
        
        Args:
            tts_engine: TextToSpeech instance for voice announcements
            sounds_dir: Directory containing sound effect files
        """
        self.tts = tts_engine
        # Use provided path, env var, or default to relative path
        if sounds_dir:
            self.sounds_dir = Path(sounds_dir)
        else:
            self.sounds_dir = Path(os.environ.get(
                'BEAST_SOUNDS_DIR',
                str(Path(__file__).parent.parent / 'sounds')
            ))
        
        # Ensure sounds directory exists
        self.sounds_dir.mkdir(parents=True, exist_ok=True)
        
    def announce(self, message: str, blocking: bool = True):
        """
        Make a voice announcement.
        
        Args:
            message: Text to speak
            blocking: Wait for speech to complete
        """
        if not message:
            return
            
        logger.info(f"Announcing: {message}")
        
        if self.tts:
            try:
                self.tts.speak(message, blocking=blocking)
            except Exception as e:
                logger.error(f"Announcement failed: {e}")
        else:
            logger.warning("No TTS engine available for announcement")
            
    def announce_preset(self, preset_name: str, blocking: bool = True):
        """
        Make a preset announcement.
        
        Args:
            preset_name: Name of preset (e.g., 'ready', 'shutdown')
            blocking: Wait for speech to complete
        """
        if preset_name in self.ANNOUNCEMENTS:
            self.announce(self.ANNOUNCEMENTS[preset_name], blocking)
        else:
            logger.warning(f"Unknown preset: {preset_name}")
            
    def play_sound(self, sound_name: str, blocking: bool = False):
        """
        Play a sound effect.
        
        Args:
            sound_name: Name of sound (e.g., 'listening', 'success', 'error')
            blocking: Wait for sound to complete
        """
        sound_file = self.sounds_dir / f"{sound_name}.wav"
        
        if not sound_file.exists():
            # Generate simple beep instead
            self._play_beep(sound_name)
            return
            
        try:
            import subprocess
            import platform
            
            if platform.system() == "Darwin":
                cmd = ['afplay', str(sound_file)]
            else:
                cmd = ['aplay', '-q', str(sound_file)]
                
            if blocking:
                subprocess.run(cmd, capture_output=True)
            else:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
        except Exception as e:
            logger.error(f"Failed to play sound: {e}")
            
    def _play_beep(self, sound_type: str = "default"):
        """Generate and play a simple beep sound"""
        try:
            import numpy as np
            
            # Sound parameters
            sample_rate = 44100
            duration = 0.15
            
            # Different beep types
            if sound_type == "listening":
                freq = 880  # High beep
            elif sound_type == "success":
                freq = 660  # Medium beep
            elif sound_type == "error":
                freq = 440  # Low beep
            else:
                freq = 660
                
            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration))
            wave = np.sin(2 * np.pi * freq * t)
            
            # Apply envelope
            envelope = np.exp(-3 * t / duration)
            wave = wave * envelope * 0.3  # Scale volume
            
            # Convert to int16
            audio = (wave * 32767).astype(np.int16)
            
            # Play
            try:
                import sounddevice as sd
                sd.play(audio, sample_rate)
                sd.wait()
            except ImportError:
                # Fallback: write temp file and play
                import tempfile
                import wave
                import subprocess
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                    
                wf = wave.open(temp_path, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())
                wf.close()
                
                subprocess.run(['aplay', '-q', temp_path], capture_output=True)
                Path(temp_path).unlink()
                
        except Exception as e:
            logger.debug(f"Beep failed: {e}")
            # Last resort: terminal bell
            print('\a', end='', flush=True)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing Announcer...")
    
    # Test without TTS
    announcer = Announcer(tts_engine=None)
    
    print("Playing beep sounds...")
    announcer.play_sound("listening")
    announcer.play_sound("success")
    announcer.play_sound("error")
    
    print("\nTest complete!")
