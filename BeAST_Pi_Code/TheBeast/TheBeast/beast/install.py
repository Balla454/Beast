#!/usr/bin/env python3
"""
beast System Installer
======================
Downloads and installs all necessary dependencies for the beast system.

Usage:
    python install.py              # Full installation
    python install.py --minimal    # Core packages only (no ML models)
    python install.py --dev        # Include development tools
    python install.py --check      # Check what's installed
    python install.py --models     # Download ML models only

Works on: macOS, Linux (including Raspberry Pi)
"""

import subprocess
import sys
import os
import platform
import argparse
import shutil
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

# Core packages (always installed)
CORE_PACKAGES = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pyyaml>=6.0",
    "pathlib",
]

# Audio packages
AUDIO_PACKAGES = [
    "sounddevice>=0.4.6",
    "soundfile>=0.12.0",
    "webrtcvad>=2.0.10",
]

# Database packages
DATABASE_PACKAGES = [
    "psycopg2-binary>=2.9.9",  # PostgreSQL
]

# ML/AI packages
ML_PACKAGES = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "sentence-transformers>=2.2.0",
    "faiss-cpu>=1.7.4",
]

# Speech packages
SPEECH_PACKAGES = [
    "faster-whisper>=0.9.0",
    "piper-tts>=1.2.0",
]

# Development packages
DEV_PACKAGES = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
    "ipython>=8.0.0",
]

# System dependencies by platform
SYSTEM_DEPS = {
    "Darwin": {  # macOS
        "brew": [
            "portaudio",
            "ffmpeg",
            "espeak",
            "espeak-ng",
        ]
    },
    "Linux": {  # Linux / Raspberry Pi
        "apt": [
            "python3-full",
            "python3-pip",
            "python3-venv",
            "portaudio19-dev",
            "libasound2-dev",
            "libsndfile1-dev",
            "ffmpeg",
            "espeak",
            "espeak-ng",
           # "libatlas-base-dev",
            "libopenblas-dev",
        ]
    }
}

# Model URLs (for --models flag)
MODEL_DOWNLOADS = {
    "whisper-tiny-en": {
        "model_id": "Systran/faster-whisper-tiny.en",
        "dest": "models/stt/faster-whisper-tiny.en",
        "description": "Faster Whisper Tiny English (CPU optimized)"
    },
    "whisper-tiny-multi": {
        "model_id": "Systran/faster-whisper-tiny",
        "dest": "models/stt/faster-whisper-tiny",
        "description": "Faster Whisper Tiny Multilingual (CPU optimized)"
    },
    "piper-tts": {
        "pip_package": "piper-tts",
        "voices_url": "https://huggingface.co/rhasspy/piper-voices",
        "description": "Piper TTS (requires espeak-ng)"
    },
    "gemma2-2b": {
        "ollama_model": "gemma2:2b",
        "description": "Google Gemma 2 2B via Ollama (recommended LLM)"
    },
    "embedding": {
        "model": "all-MiniLM-L6-v2",
        "description": "Sentence embeddings for RAG"
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(text: str):
    """Print step indicator."""
    print(f"\n>>> {text}")


def print_success(text: str):
    """Print success message."""
    print(f"✓ {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"⚠ {text}")


def print_error(text: str):
    """Print error message."""
    print(f"✗ {text}")


def run_command(cmd: list, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command."""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        if capture and e.stderr:
            print(e.stderr)
        raise


def check_command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(cmd) is not None


def get_platform() -> str:
    """Get current platform."""
    return platform.system()


def is_raspberry_pi() -> bool:
    """Check if running on Raspberry Pi."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            return "Raspberry Pi" in f.read()
    except:
        return False


def check_python_version():
    """Ensure Python version is compatible."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        print_error(f"Python 3.9+ required, found {major}.{minor}")
        sys.exit(1)
    print_success(f"Python {major}.{minor} detected")


# =============================================================================
# Installation Functions
# =============================================================================

def install_system_dependencies():
    """Install system-level dependencies."""
    print_step("Installing system dependencies...")
    
    system = get_platform()
    
    if system == "Darwin":  # macOS
        if not check_command_exists("brew"):
            print_warning("Homebrew not found. Please install from https://brew.sh")
            print("Then run: brew install portaudio ffmpeg espeak")
            return False
        
        packages = SYSTEM_DEPS["Darwin"]["brew"]
        for pkg in packages:
            print(f"  Installing {pkg}...")
            try:
                run_command(["brew", "install", pkg], check=False)
            except:
                print_warning(f"Could not install {pkg}")
        
    elif system == "Linux":
        if check_command_exists("apt"):
            packages = SYSTEM_DEPS["Linux"]["apt"]
            print("  Updating package list...")
            run_command(["sudo", "apt", "update"], check=False)
            
            print("  Installing packages...")
            run_command(
                ["sudo", "apt", "install", "-y"] + packages,
                check=False
            )
        else:
            print_warning("apt not found. Please install dependencies manually:")
            print("  portaudio, ffmpeg, espeak, libasound, libsndfile")
            return False
    
    print_success("System dependencies installed")
    return True


def install_pip_packages(packages: list, name: str):
    """Install pip packages."""
    print_step(f"Installing {name}...")
    
    for pkg in packages:
        pkg_name = pkg.split(">=")[0].split("==")[0]
        print(f"  {pkg_name}...", end=" ", flush=True)
        try:
            run_command(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                capture=True
            )
            print("✓")
        except:
            print("✗")
            print_warning(f"Failed to install {pkg}")


def install_pyaudio():
    """Install PyAudio with platform-specific handling."""
    print_step("Installing PyAudio...")
    
    system = get_platform()
    
    try:
        if system == "Darwin":
            # macOS may need special handling
            run_command(
                [sys.executable, "-m", "pip", "install", "pyaudio", "-q"],
                capture=True
            )
        else:
            run_command(
                [sys.executable, "-m", "pip", "install", "pyaudio", "-q"],
                capture=True
            )
        print_success("PyAudio installed")
    except:
        print_warning("PyAudio installation failed")
        print("  Try: pip install --global-option='build_ext' pyaudio")


def install_torch_optimized():
    """Install PyTorch with platform-specific optimizations."""
    print_step("Installing PyTorch...")
    
    system = get_platform()
    is_pi = is_raspberry_pi()
    
    try:
        if is_pi:
            # Raspberry Pi - CPU only, specific wheel
            run_command([
                sys.executable, "-m", "pip", "install",
                "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
                "-q"
            ], capture=True)
        elif system == "Darwin":
            # macOS - MPS support for Apple Silicon
            run_command([
                sys.executable, "-m", "pip", "install", "torch", "-q"
            ], capture=True)
        else:
            # Linux - CPU version
            run_command([
                sys.executable, "-m", "pip", "install",
                "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
                "-q"
            ], capture=True)
        print_success("PyTorch installed")
    except:
        print_warning("PyTorch installation failed - try manually")


def download_models():
    """Download ML models for offline use."""
    print_step("Downloading ML models...")
    
    # Create models directory relative to this script
    # This ensures models end up in TheBeast/TheBeast/models regardless of CWD
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"

    models_dir.mkdir(exist_ok=True)
    (models_dir / "stt").mkdir(exist_ok=True)
    (models_dir / "tts").mkdir(exist_ok=True)
    (models_dir / "embeddings").mkdir(exist_ok=True)
    
    # =========================================================================
    # 1. Install Ollama and pull gemma2:2b
    # =========================================================================
    print("\n  Setting up Ollama with gemma2:2b...")
    
    if not check_command_exists("ollama"):
        print_warning("Ollama not installed")
        system = get_platform()
        if system == "Darwin":
            print("  Install Ollama from: https://ollama.com/download")
            print("  Or run: brew install ollama")
        elif system == "Linux":
            print("  Install with: curl -fsSL https://ollama.com/install.sh | sh")
        print("\n  After installing, run:")
        print("    ollama serve  (in background)")
        print("    ollama pull gemma2:2b")
    else:
        print_success("Ollama installed")
        print("  Pulling gemma2:2b model (this may take a few minutes)...")
        try:
            result = run_command(["ollama", "pull", "gemma2:2b"], check=False, capture=True)
            if result.returncode == 0:
                print_success("gemma2:2b model downloaded")
            else:
                print_warning("Could not pull model. Make sure 'ollama serve' is running.")
                print("  Start Ollama: ollama serve")
                print("  Then pull: ollama pull gemma2:2b")
        except Exception as e:
            print_warning(f"Could not pull gemma2:2b: {e}")
            print("  Make sure Ollama is running: ollama serve")
    
    # =========================================================================
    # 2. Download faster-whisper-tiny.en (English, CPU optimized)
    # =========================================================================
    print("\n  Downloading faster-whisper-tiny.en (English, CPU)...")
    try:
        from faster_whisper import WhisperModel
        # This downloads the model to the default cache and loads it
        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        print_success("faster-whisper-tiny.en downloaded")
        del model  # Free memory
    except ImportError:
        print_warning("faster-whisper not installed")
        print("  Run: pip install faster-whisper")
    except Exception as e:
        print_warning(f"Could not download Whisper model: {e}")
    
    # =========================================================================
    # 3. Pre-download sentence-transformers embedding model
    # =========================================================================
    print("\n  Downloading embedding model (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        cache_dir = models_dir / "embeddings"
        model.save(str(cache_dir / "all-MiniLM-L6-v2"))
        print_success("Embedding model downloaded")
        del model
    except ImportError:
        print_warning("sentence-transformers not installed, skipping")
    except Exception as e:
        print_warning(f"Could not download embedding model: {e}")
    
    # =========================================================================
    # 4. Download Piper Voice (en_US-lessac-medium)
    # =========================================================================
    print("\n  Downloading Piper voice (en_US-lessac-medium)...")
    piper_dir = models_dir / "tts"
    voice_name = "en_US-lessac-medium"
    onnx_file = piper_dir / f"{voice_name}.onnx"
    json_file = piper_dir / f"{voice_name}.onnx.json"
    
    # URLs for the voice model
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
    onnx_url = f"{base_url}/{voice_name}.onnx"
    json_url = f"{base_url}/{voice_name}.onnx.json"
    
    import urllib.request
    
    if not onnx_file.exists():
        print(f"  Downloading {onnx_file.name}...")
        try:
            urllib.request.urlretrieve(onnx_url, onnx_file)
            print_success("Voice model downloaded")
        except Exception as e:
            print_warning(f"Failed to download voice model: {e}")
    else:
        print_success("Voice model already exists")
        
    if not json_file.exists():
        print(f"  Downloading {json_file.name}...")
        try:
            urllib.request.urlretrieve(json_url, json_file)
            print_success("Voice config downloaded")
        except Exception as e:
            print_warning(f"Failed to download voice config: {e}")
    else:
        print_success("Voice config already exists")

    # =========================================================================
    # 5. Check espeak for Piper TTS
    # =========================================================================
    print("\n  Checking espeak (required for Piper TTS)...")
    if check_command_exists("espeak") or check_command_exists("espeak-ng"):
        print_success("espeak found")
    else:
        print_warning("espeak NOT found - required for Piper TTS")
        system = get_platform()
        if system == "Darwin":
            print("  Install with: brew install espeak espeak-ng")
        else:
            print("  Install with: sudo apt install espeak-ng")
    
    # =========================================================================
    # 6. Check piper-tts
    # =========================================================================
    print("\n  Checking Piper TTS...")
    try:
        import piper
        print_success("piper-tts installed")
        print("\n  Download Piper voices from:")
        print("    https://huggingface.co/rhasspy/piper-voices")
        print("  Recommended: en_US-lessac-medium.onnx")
        print("  Place .onnx and .onnx.json files in models/tts/")
    except ImportError:
        print_warning("piper-tts not installed")
        print("  Run: pip install piper-tts")
    
    print("\n" + "-" * 50)
    print("Model Summary:")
    print("  LLM: gemma2:2b via Ollama")
    print("  STT: faster-whisper-tiny.en (English, CPU optimized)")
    print("  TTS: Piper TTS (requires espeak-ng)")
    print("  Embeddings: all-MiniLM-L6-v2")
    print("\nTo start Ollama (required for LLM):")
    print("  ollama serve")
    print("\nTo verify Ollama models:")
    print("  ollama list")


def check_installation():
    """Check what's installed and working."""
    print_header("Installation Check")
    
    checks = [
        ("numpy", "Core numerics"),
        ("scipy", "Scientific computing"),
        ("yaml", "Configuration (pyyaml)"),
        ("sounddevice", "Audio I/O"),
        ("soundfile", "Audio files"),
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face"),
        ("sentence_transformers", "Embeddings"),
        ("faiss", "Vector search (faiss-cpu)"),
        ("faster_whisper", "Speech-to-text"),
        ("piper", "Text-to-speech (piper-tts)"),
        ("psycopg2", "PostgreSQL"),
        ("webrtcvad", "Voice activity detection"),
    ]
    
    installed = 0
    for module, description in checks:
        try:
            __import__(module)
            print_success(f"{description}")
            installed += 1
        except ImportError:
            print_error(f"{description} - NOT INSTALLED")
    
    print(f"\n{installed}/{len(checks)} packages installed")
    
    # Check system tools
    print("\nSystem tools:")
    tools = ["ffmpeg", "espeak", "espeak-ng", "ollama"]
    for tool in tools:
        if check_command_exists(tool):
            print_success(tool)
        else:
            print_error(f"{tool} - NOT FOUND")
    
    # Check Ollama and gemma2:2b model
    print("\nOllama LLM (gemma2:2b):")
    if check_command_exists("ollama"):
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                models = [m.get('name', '') for m in resp.json().get('models', [])]
                if any('gemma2:2b' in m for m in models):
                    print_success("gemma2:2b model available")
                else:
                    print_warning("gemma2:2b not found. Pull it with: ollama pull gemma2:2b")
                    print(f"  Available models: {', '.join(models) if models else 'none'}")
            else:
                print_warning("Ollama not responding. Start with: ollama serve")
        except Exception:
            print_warning("Ollama not running. Start with: ollama serve")
    else:
        print_error("Ollama not installed")
        print("  Install from: https://ollama.com/download")
    
    # Check STT model
    print("\nSTT Model (faster-whisper-tiny.en):")
    try:
        from faster_whisper import WhisperModel
        # Check if model is cached
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(cache_dir):
            print_success("faster-whisper cache directory exists")
        print("  Model will download on first use if not cached")
    except ImportError:
        print_error("faster-whisper not installed")
    
    # Check audio devices
    print("\nAudio devices:")
    try:
        import sounddevice as sd
        inputs = [d for d in sd.query_devices() if d.get('max_input_channels', 0) > 0]
        outputs = [d for d in sd.query_devices() if d.get('max_output_channels', 0) > 0]
        print(f"  Input devices: {len(inputs)}")
        print(f"  Output devices: {len(outputs)}")
        
        # Check for USB devices
        usb_inputs = [d for d in inputs if 'usb' in d.get('name', '').lower()]
        usb_outputs = [d for d in outputs if 'usb' in d.get('name', '').lower()]
        if usb_inputs:
            print_success(f"USB microphone found: {usb_inputs[0].get('name', 'Unknown')}")
        if usb_outputs:
            print_success(f"USB speaker found: {usb_outputs[0].get('name', 'Unknown')}")
    except Exception as e:
        print_warning(f"Could not check audio devices: {e}")


def create_directories():
    """Create necessary directories."""
    print_step("Creating directories...")
    
    dirs = [
        "models/stt",
        "models/tts",
        "models/embeddings",
        "logs",
        "data",
        "cache",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print_success("Directory structure created")


# =============================================================================
# Main Installation Flow
# =============================================================================

def full_install(include_dev: bool = False, minimal: bool = False):
    """Run full installation."""
    print_header("beast System Installation")
    print(f"Platform: {get_platform()}")
    print(f"Python: {sys.version}")
    if is_raspberry_pi():
        print("Device: Raspberry Pi detected")
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # System dependencies
    install_system_dependencies()
    
    # Upgrade pip
    print_step("Upgrading pip...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    
    # Core packages
    install_pip_packages(CORE_PACKAGES, "core packages")
    
    # Audio packages
    install_pip_packages(AUDIO_PACKAGES, "audio packages")
    install_pyaudio()
    
    # Database
    install_pip_packages(DATABASE_PACKAGES, "database packages")
    
    if not minimal:
        # ML packages (with optimized torch)
        install_torch_optimized()
        install_pip_packages(
            [p for p in ML_PACKAGES if "torch" not in p.lower()],
            "ML packages"
        )
        
        # Speech packages
        install_pip_packages(SPEECH_PACKAGES, "speech packages")
    
    if include_dev:
        install_pip_packages(DEV_PACKAGES, "development packages")
    
    # Final check
    check_installation()
    
    print_header("Installation Complete!")
    print("\nNext steps:")
    print("  1. Run: python install.py --models  (to download ML models)")
    print("  2. Configure: edit config.yaml")
    print("  3. Test: python test_e2e.py --scenario normal")
    print("  4. Run: python main.py")


def main():
    parser = argparse.ArgumentParser(
        description="beast System Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py              # Full installation
  python install.py --minimal    # Core only (no ML)
  python install.py --dev        # Include dev tools
  python install.py --check      # Check installation
  python install.py --models     # Download models
        """
    )
    
    parser.add_argument(
        "--minimal", "-m",
        action="store_true",
        help="Install core packages only (no ML/AI)"
    )
    parser.add_argument(
        "--dev", "-d",
        action="store_true",
        help="Include development packages"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check what's installed"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Download ML models for offline use"
    )
    parser.add_argument(
        "--system-only",
        action="store_true",
        help="Install system dependencies only"
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_installation()
    elif args.models:
        download_models()
    elif args.system_only:
        install_system_dependencies()
    else:
        full_install(include_dev=args.dev, minimal=args.minimal)


if __name__ == "__main__":
    main()
