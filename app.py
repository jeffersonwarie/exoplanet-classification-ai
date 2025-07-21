"""
Main entry point for Streamlit Exoplanet Classification App
"""

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
from streamlit_exoplanet_app import main

if __name__ == "__main__":
    main()
