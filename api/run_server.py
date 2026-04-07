"""
Script to start the Inflation Prediction API
Run this to start the FastAPI server
"""

import subprocess
import sys
from pathlib import Path

def start_api():
    """Start the FastAPI server"""
    api_dir = Path(__file__).parent

    print("=" * 70)
    print(" INFLATION PREDICTION API - STARTING SERVER")
    print("=" * 70)
    print()
    print("[INFO] Starting FastAPI server...")
    print("[INFO] API will be available at: http://localhost:8000")
    print("[INFO] Documentation at: http://localhost:8000/docs")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    print("=" * 70)
    print()

    try:
        # Start uvicorn server with improved dashboard
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "app_improved:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
            cwd=str(api_dir),
            check=False
        )
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped")
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_api()
