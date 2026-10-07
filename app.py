"""
CliNexa Healthcare Intelligence Platform
Universal Application Entry Point & Hugging Face Runner
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    port = os.environ.get("PORT", "7860")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/main.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true"
    ]
    print(f"Starting CliNexa on port {port}...")
    subprocess.run(cmd)
