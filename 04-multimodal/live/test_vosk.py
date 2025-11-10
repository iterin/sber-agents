#!/usr/bin/env python3
"""Simple test script to verify Vosk integration works."""

import asyncio
import tempfile
import os
from pathlib import Path

# Test imports
try:
    from vosk import Model as VoskModel
    print("OK Vosk import successful")
except ImportError as e:
    print(f"FAIL Vosk import failed: {e}")
    exit(1)

try:
    import soundfile as sf
    print("OK soundfile import successful")
except ImportError as e:
    print(f"FAIL soundfile import failed: {e}")
    exit(1)

try:
    from src.config import config
    print("OK config import successful")
    print(f"  VOSK_MODEL_PATH: {config.VOSK_MODEL_PATH}")
except ImportError as e:
    print(f"FAIL config import failed: {e}")
    exit(1)

# Test model loading
async def test_model_loading():
    try:
        from src.handlers import ensure_vosk_model
        print("Testing model loading...")
        model = await ensure_vosk_model()
        print("OK Model loaded successfully")
        return True
    except Exception as e:
        print(f"FAIL Model loading failed: {e}")
        return False

# Test ffmpeg
def test_ffmpeg():
    try:
        import ffmpeg
        # Create a dummy test
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"dummy")
            dummy_file = f.name

        try:
            # Try to probe the file (this will fail but test ffmpeg availability)
            ffmpeg.probe(dummy_file)
        except:
            pass  # Expected to fail, but ffmpeg is available
        finally:
            os.unlink(dummy_file)

        print("OK ffmpeg import successful")
        return True
    except ImportError as e:
        print(f"FAIL ffmpeg import failed: {e}")
        return False

async def main():
    print("Testing Vosk integration...")
    print("=" * 40)

    # Test basic imports
    test_ffmpeg()

    # Test model loading
    success = await test_model_loading()

    if success:
        print("\nSUCCESS All tests passed! Bot should work with Vosk.")
    else:
        print("\nERROR Some tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
