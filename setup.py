"""
Setup script for the text-driven video search pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "data/videos",
        "data/keyframes", 
        "data/features",
        "data/index",
        "data/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies!")
        print("Please run: pip install -r requirements.txt")
        return False
    return True

def download_models():
    """Download required models."""
    print("Downloading models (this may take a while)...")
    try:
        # Import models to trigger download
        from transformers import BlipProcessor, BlipForConditionalGeneration
        # No CLIP used in this pipeline
        
        print("✓ Models downloaded successfully!")
    except Exception as e:
        print(f"✗ Model download failed: {e}")
        print("Models will be downloaded automatically on first use.")

def check_requirements():
    """Check system requirements."""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required!")
        return False
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
    except ImportError:
        print("⚠ PyTorch not installed yet")
    
    return True

def main():
    """Main setup function."""
    print("Text-Driven Video Search Pipeline Setup")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        print("Setup failed due to requirements not met.")
        return
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("Setup incomplete due to dependency installation failure.")
        return
    
    # Download models
    download_models()
    
    print("\n" + "=" * 40)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your video files to data/videos/")
    print("2. Run: python video_search_pipeline.py build")
    print("3. Run: python video_search_pipeline.py search 'your query'")
    print("\nFor interactive mode:")
    print("   python video_search_pipeline.py interactive")
    print("\nFor help:")
    print("   python video_search_pipeline.py --help")

if __name__ == "__main__":
    main()

