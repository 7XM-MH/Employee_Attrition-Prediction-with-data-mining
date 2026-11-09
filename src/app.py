"""Entry point for the Streamlit application.

Usage:
    streamlit run src/app.py
"""
import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit app from the Home page."""
    home = Path(__file__).resolve().parent.parent / "app" / "Home.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(home)], check=True)


if __name__ == "__main__":
    main()
