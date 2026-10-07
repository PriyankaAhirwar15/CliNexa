"""
CliNexa — AI-Powered Healthcare Intelligence Platform
Streamlit Entry Point
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Launch the Streamlit app
from app.main import main

if __name__ == "__main__":
    main()
