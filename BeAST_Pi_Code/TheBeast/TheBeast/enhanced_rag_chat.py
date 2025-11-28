#!/usr/bin/env python3
"""
TheBeast AI - Enhanced Gemma 2B Chat with RAG System
Chat interface with Retrieval Augmented Generation using organized research datasets
"""

import os
import sys
import logging
import threading
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our RAG system
try:
    from rag_system import RAGSystem
    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ RAG system not available: {e}")
    RAG_AVAILABLE = False

# Import TTS
try:
    sys.path.append('organized/applications/data_analysis/core_modules')
    from tts_engine import TTSEngine
    TTS_AVAILABLE = True
    logger.info("✅ Piper TTS available")
except ImportError:
    logger.warning("⚠️ Piper TTS engine not available")
    TTS_AVAILABLE = False

# Import Speech-to-Text
try:
    from moonshine_stt import MoonshineSTT
    STT_AVAILABLE = True
    logger.info("✅ Moonshine STT available")
except ImportError:
    logger.warning("⚠️ Moonshine STT not available")
    STT_AVAILABLE = False

class EnhancedGemmaRAGChat:
    """Enhanced chat system with RAG capabilities and voice input"""
    
    def __init__(self):
        self.dataset_root = "/Users/collinball/Applications/TheBeast/dataset"
        self.model_path = "/Users/collinball/Applications/TheBeast/organized/models/gemma3n"
        
        # Initialize components
        self.rag_system = None
        self.tts_engine = None
        self.stt_engine = None
        
        # Settings
        self.tts_enabled = True
        self.stt_enabled = True
        self.use_rag = True
        self.show_sources = False  # Disabled by default
        self.continuous_listening = False
        self.wake_word_mode = True  # Use wake words by default
        
        # Initialize systems
        self.initialize()
    
    def initialize(self):
        """Initialize all systems"""
        logger.info("🚀 Initializing Enhanced Gemma RAG Chat with Voice Input...")
        
        # Initialize RAG system
        if RAG_AVAILABLE:
            try:
                self.rag_system = RAGSystem(self.dataset_root, self.model_path, preload=True)
                logger.info("✅ RAG system initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize RAG system: {e}")
                self.rag_system = None
        
        # Initialize TTS
        if TTS_AVAILABLE:
            try:
                self.tts_engine = TTSEngine()
                logger.info("✅ Piper TTS engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Piper TTS: {e}")
                self.tts_engine = None
                self.tts_enabled = False
        
        # Initialize Speech-to-Text
        if STT_AVAILABLE:
            try:
                self.stt_engine = MoonshineSTT(model_name="moonshine/tiny")
                logger.info("✅ Moonshine STT initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize STT: {e}")
                self.stt_engine = None
                self.stt_enabled = False
        
        logger.info("✅ Enhanced chat system ready!")
    
    def get_voice_input(self, method: str = "silence") -> str:
        """Get voice input using speech-to-text"""
        if not self.stt_enabled or not self.stt_engine:
            print("❌ Voice input not available")
            return ""
        
        try:
            if method == "silence":
                print("🎤 Speak your question (will auto-stop after silence)...")
                transcription = self.stt_engine.record_until_silence()
            elif method == "timed":
                print("🎤 Speak your question (5 seconds)...")
                transcription = self.stt_engine.record_and_transcribe(duration=5.0)
            else:
                print("❌ Unknown voice input method")
                return ""
            
            if transcription:
                print(f"🗣️ You said: '{transcription}'")
                return transcription
            else:
                print("⚠️ No speech detected")
                return ""
                
        except KeyboardInterrupt:
            print("\n🛑 Voice input cancelled")
            return ""
        except Exception as e:
            print(f"❌ Voice input error: {e}")
            return ""
    
    def start_always_listening(self, wake_word_mode: bool = True):
        """Start always-on voice listening mode"""
        if not self.stt_enabled or not self.stt_engine:
            print("❌ Speech-to-text not available")
            return False
            
        if self.continuous_listening:
            print("⚠️ Already in continuous listening mode")
            return False
            
        self.wake_word_mode = wake_word_mode
        self.continuous_listening = True
        
        # Start continuous listening with callback
        self.stt_engine.start_continuous_listening(
            callback=self._handle_voice_command,
            wake_word_mode=wake_word_mode
        )
        
        mode_text = "wake word" if wake_word_mode else "always active"
        print(f"🎤 Started always-on listening ({mode_text} mode)")
        if wake_word_mode:
            wake_words = ", ".join(self.stt_engine.wake_words)
            print(f"💡 Wake words: {wake_words}")
        print("💡 Say 'stop listening' to disable always-on mode")
        return True
    
    def stop_always_listening(self):
        """Stop always-on voice listening mode"""
        if not self.continuous_listening:
            return False
            
        if self.stt_engine:
            self.stt_engine.stop_continuous_listening()
            
        self.continuous_listening = False
        print("🛑 Stopped always-on listening")
        return True
    
    def _handle_voice_command(self, transcription: str):
        """Handle voice commands from continuous listening"""
        try:
            # Check for stop command
            if "stop listening" in transcription.lower():
                self.stop_always_listening()
                return
                
            # Process the voice command as a regular chat message
            print(f"\n🎤 Voice: '{transcription}'")
            self.process_message(transcription)
            print("\n🎤 Listening...")  # Show we're still listening
            
        except Exception as e:
            print(f"❌ Error processing voice command: {e}")

    def toggle_sources(self):
        """Toggle source citation display"""
        self.show_sources = not self.show_sources
        status = "enabled" if self.show_sources else "disabled"
        return f"📚 Source citations {status}"
    def speak_response(self, text: str):
        """Convert text to speech using Piper TTS"""
        if self.tts_enabled and self.tts_engine and text.strip():
            try:
                # Clean text for TTS
                clean_text = text.replace("**", "").replace("*", "").replace("#", "").strip()
                if clean_text:
                    # Use non-blocking speech so the user can continue interacting
                    self.tts_engine.speak(clean_text, blocking=False)
                    logger.debug(f"Speaking with Piper: {clean_text[:50]}...")
            except Exception as e:
                logger.error(f"❌ Piper TTS error: {e}")
    
    def process_query(self, user_input: str) -> str:
        """Process user query with RAG or standard mode"""
        
        # Check if this is a dataset/research question that would benefit from RAG
        research_keywords = [
            'dataset', 'data', 'research', 'study', 'participant', 'sensor',
            'wesad', 'eeg', 'stress', 'questionnaire', 'physiological',
            'detection', 'brain', 'wearable', 'analysis'
        ]
        
        use_rag_for_query = (
            self.use_rag and 
            self.rag_system and 
            any(keyword in user_input.lower() for keyword in research_keywords)
        )
        
        if use_rag_for_query:
            print("🔍 Searching research datasets...")
            try:
                result = self.rag_system.generate_rag_response(user_input)
                
                # Format response with sources
                response = result['response']
                
                # Only show sources if explicitly enabled
                if self.show_sources and result['sources']:
                    response += "\n\n📚 Sources consulted:\n"
                    for source in result['sources'][:3]:
                        source_name = Path(source).name
                        response += f"  • {source_name}\n"
                
                return response
                
            except Exception as e:
                logger.error(f"❌ RAG processing failed: {e}")
                return "I encountered an error accessing the research data. Let me try a standard response."
        
        else:
            # Use standard Gemma 2B without RAG
            return self._generate_standard_response(user_input)
    
    def _generate_standard_response(self, user_input: str) -> str:
        """Generate standard response without RAG"""
        if not self.rag_system or not self.rag_system.model:
            return "I'm sorry, the language model is not available for standard responses."
        
        try:
            # Create simple prompt
            prompt = f"User: {user_input}\nAssistant:"
            
            # Use the same model from RAG system for consistency
            tokenizer = self.rag_system.tokenizer
            model = self.rag_system.model
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=400,
                truncation=True,
                padding=True,
                return_attention_mask=True
            )
            
            # Generate
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=80,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response if response else "I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"❌ Standard generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def toggle_rag(self):
        """Toggle RAG mode on/off"""
        if self.rag_system:
            self.use_rag = not self.use_rag
            status = "enabled" if self.use_rag else "disabled"
            return f"🧠 RAG mode {status}"
        else:
            return "❌ RAG system not available"
    
    def toggle_sources(self):
        """Toggle source display on/off"""
        self.show_sources = not self.show_sources
        status = "enabled" if self.show_sources else "disabled"
        return f"� Source citations {status}"
    
    def show_status(self):
        """Show system status"""
        status = "🤖 TheBeast AI System Status:\n"
        status += f"📊 RAG System: {'✅ Available' if self.rag_system else '❌ Not available'}\n"
        status += f"🔍 RAG Mode: {'✅ Enabled' if self.use_rag else '❌ Disabled'}\n"
        status += f"🔊 TTS Engine: {'✅ Available' if self.tts_engine else '❌ Not available'}\n"
        status += f"🗣️ TTS Mode: {'✅ Enabled' if self.tts_enabled else '❌ Disabled'}\n"
        status += f"🎤 STT Mode: {'✅ Enabled' if self.stt_enabled else '❌ Disabled'}\n"
        
        # Show continuous listening status
        if self.continuous_listening:
            mode = "Wake Word" if self.wake_word_mode else "Always Active"
            status += f"👂 Continuous Listening: ✅ Enabled ({mode} mode)\n"
        else:
            status += "👂 Continuous Listening: ❌ Disabled\n"
            
        status += f"📚 Dataset Root: {self.dataset_root}\n"
        
        if self.rag_system and self.rag_system.knowledge_base:
            doc_count = len(self.rag_system.knowledge_base.documents)
            status += f"📑 Indexed Documents: {doc_count}\n"
            if hasattr(self.rag_system, '_response_cache'):
                status += f"💾 Cached Responses: {len(self.rag_system._response_cache)}\n"
        
        return status
    
    def show_help(self):
        """Show help information"""
        help_text = """
🤖 TheBeast AI Enhanced Chat Commands:

Chat Commands:
  /rag       - Toggle RAG mode on/off
  /tts       - Toggle text-to-speech on/off
  /sources   - Toggle source citations on/off
  /voice     - Voice input (auto-detect silence)
  /listen    - Voice input (5 second recording)
  /always    - Start always-on listening (wake word mode)
  /active    - Start always-on listening (no wake words)
  /stop      - Stop always-on listening
  /mics      - List available microphone devices
  /mic <n>   - Select microphone device by index
  /testmic   - Test current microphone device
  /status    - Show system status
  /help      - Show this help
  /quit      - Exit chat

Always-On Listening:
  • /always: Listens for wake words: "hey beast", "hey computer", "listen up"
  • /active: Always listening (no wake words needed)
  • Say "stop listening" to disable always-on mode
  • Voice commands are processed automatically

RAG Features:
  • Automatically searches research datasets for relevant information
  • Works with WESAD, EEG, questionnaire, and documentation data
  • Combines retrieved information with AI generation
  • English-only responses

Examples:
  "What datasets do we have for stress detection?"
  "Tell me about the WESAD study participants"
  "What EEG datasets are available?"
  "How many questionnaire responses do we have?"
        """
        return help_text.strip()
    
    def list_microphones(self):
        """List available microphone devices"""
        if not self.stt_enabled or not self.stt_engine:
            print("❌ Speech-to-text not available")
            return
            
        print("\n🎤 Available microphone devices:")
        devices = self.stt_engine.list_microphone_devices()
        
        if not devices:
            print("❌ No microphone devices found")
            return
            
        for device in devices:
            current = " (CURRENT)" if self.stt_engine.device_index == device['index'] else ""
            print(f"  {device['index']}: {device['name']}{current}")
            print(f"       Channels: {device['channels']}, Sample Rate: {device['sample_rate']}Hz")
        
        print("\n💡 Use '/mic <index>' to select a device")
        print("💡 Use '/testmic' to test the current device")
    
    def set_microphone_device(self, device_index: int):
        """Set the microphone device"""
        if not self.stt_enabled or not self.stt_engine:
            print("❌ Speech-to-text not available")
            return
            
        if self.stt_engine.set_microphone_device(device_index):
            print(f"✅ Microphone device set to index {device_index}")
        else:
            print(f"❌ Failed to set microphone device {device_index}")
            print("💡 Use '/mics' to see available devices")
    
    def test_microphone(self):
        """Test the current microphone"""
        if not self.stt_enabled or not self.stt_engine:
            print("❌ Speech-to-text not available")
            return
            
        print("🎤 Testing microphone... speak now!")
        if self.stt_engine.test_microphone():
            print("✅ Microphone test passed")
        else:
            print("❌ Microphone test failed")
            print("💡 Try a different device with '/mic <index>'")
            print("💡 Check microphone permissions in System Preferences")
    
    def run_chat(self):
        """Run the interactive chat loop"""
        print("\n" + "="*60)
        print("🧠 TheBeast AI - Enhanced Chat with RAG System")
        print("="*60)
        print("🚀 Features: Research Dataset Integration + Neural TTS")
        print("💡 Tip: Ask about your research data for RAG-enhanced responses!")
        print("📝 Type /help for commands or /quit to exit")
        print("="*60)
        
        # Initial status
        print(self.show_status())
        
        try:
            while True:
                try:
                    user_input = input("\n🗣️  You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        command = user_input[1:].lower()
                        
                        if command in ['quit', 'exit', 'q']:
                            print("👋 Thanks for using TheBeast AI!")
                            if self.tts_enabled and self.tts_engine:
                                self.tts_engine.speak("Goodbye! Thanks for using TheBeast AI!", blocking=True)
                            break
                        
                        elif command == 'rag':
                            response = self.toggle_rag()
                            print(f"⚙️  {response}")
                            continue
                        
                        elif command == 'tts':
                            response = self.toggle_tts()
                            print(f"⚙️  {response}")
                            continue
                        
                        elif command == 'sources':
                            response = self.toggle_sources()
                            print(f"⚙️  {response}")
                            continue
                        
                        elif command == 'voice':
                            # Voice input with silence detection
                            voice_input = self.get_voice_input(method="silence")
                            if voice_input.strip():
                                user_input = voice_input
                                # Continue to process the voice input as regular query
                            else:
                                print("❌ No voice input detected")
                                continue
                        
                        elif command == 'listen':
                            # Timed voice input (5 seconds)
                            voice_input = self.get_voice_input(method="timed")
                            if voice_input.strip():
                                user_input = voice_input
                                # Continue to process the voice input as regular query
                            else:
                                print("❌ No voice input detected")
                                continue
                        
                        elif command == 'always':
                            # Start always-on listening with wake words
                            if self.start_always_listening(wake_word_mode=True):
                                continue
                            else:
                                print("❌ Failed to start always-on listening")
                                continue
                        
                        elif command == 'active':
                            # Start always-on listening without wake words
                            if self.start_always_listening(wake_word_mode=False):
                                continue
                            else:
                                print("❌ Failed to start always-on listening")
                                continue
                        
                        elif command == 'stop':
                            # Stop always-on listening
                            if self.stop_always_listening():
                                continue
                            else:
                                print("⚠️  Not in always-on listening mode")
                                continue
                        
                        elif command == 'mics':
                            # List available microphone devices
                            self.list_microphones()
                            continue
                        
                        elif command.startswith('mic'):
                            # Set microphone device
                            parts = command.split()
                            if len(parts) == 2 and parts[1].isdigit():
                                device_index = int(parts[1])
                                self.set_microphone_device(device_index)
                            else:
                                print("❓ Usage: /mic <device_index>")
                                print("💡 Use /mics to list available devices")
                            continue
                        
                        elif command == 'testmic':
                            # Test current microphone
                            self.test_microphone()
                            continue
                        
                        elif command == 'status':
                            print(self.show_status())
                            continue
                        
                        elif command == 'help':
                            print(self.show_help())
                            continue
                        
                        else:
                            print("❓ Unknown command. Type /help for available commands.")
                            continue
                    
                    # Process regular query
                    print("🤖 Thinking...")
                    response = self.process_query(user_input)
                    
                    if response:
                        print(f"🤖 {response}")
                        
                        # Speak response
                        self.speak_response(response)
                    else:
                        print("🤖 I'm sorry, I couldn't generate a response.")
                    
                except KeyboardInterrupt:
                    print("\n👋 Chat ended by user.")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    logger.error(f"Chat error: {e}")
                    continue
        finally:
            # Clean up continuous listening
            if self.continuous_listening:
                self.stop_always_listening()

def main():
    """Main function"""
    try:
        chat = EnhancedGemmaRAGChat()
        chat.run_chat()
    except KeyboardInterrupt:
        print("\n🛑 Chat interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to start chat: {e}")
        print("❌ Failed to start enhanced chat system.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
