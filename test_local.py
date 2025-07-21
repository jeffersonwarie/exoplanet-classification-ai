#!/usr/bin/env python3
"""
Test script to verify Streamlit app works locally
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def test_app():
    print("🧪 Testing Streamlit app locally...")
    
    # Check if required files exist
    required_files = [
        'app.py',
        'streamlit_exoplanet_app.py',
        'requirements.txt',
        'exoplanet_model.pth'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        if 'streamlit_exoplanet_app.py' in missing_files:
            print("   📋 Copy the main app code from the Streamlit artifact")
        if 'exoplanet_model.pth' in missing_files:
            print("   🤖 Add your trained model file")
        return False
    
    print("✅ All required files present")
    
    # Try to install dependencies
    try:
        print("📦 Installing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    
    # Test import
    try:
        print("🔍 Testing imports...")
        import streamlit_exoplanet_app
        print("✅ App imports successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("🚀 Starting Streamlit app...")
    print("   Opening http://localhost:8501 in your browser...")
    print("   Press Ctrl+C to stop the server")
    
    # Open browser after a delay
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:8501')
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Streamlit
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])

if __name__ == "__main__":
    test_app()
