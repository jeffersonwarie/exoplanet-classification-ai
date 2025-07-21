#!/usr/bin/env python3
"""
Streamlit Cloud Setup Automation Script
Creates all necessary files for Streamlit deployment
"""

import os
import sys
from pathlib import Path

def create_file(filepath, content):
    """Create a file with given content"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Created: {filepath}")

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      🪐 STREAMLIT CLOUD SETUP AUTOMATION 🪐                  ║
║                                                              ║
║  Automatically creates all files needed for deployment       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Setting up your Streamlit exoplanet classification app...
""")

    # Check if we're in the right directory
    if not os.path.exists('README.md'):
        print("⚠️  This doesn't look like a GitHub repository directory.")
        print("   Please run this script in your cloned GitHub repository.")
        sys.exit(1)

    # 1. Create requirements.txt
    requirements_content = """torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
streamlit>=1.28.0
plotly>=5.0.0
"""
    create_file('requirements.txt', requirements_content)

    # 2. Create .streamlit/config.toml
    streamlit_config = """[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
"""
    create_file('.streamlit/config.toml', streamlit_config)

    # 3. Create app.py (main entry point)
    app_py_content = '''"""
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
'''
    create_file('app.py', app_py_content)

    # 4. Create .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Logs and databases
*.log
*.db
streamlit_monitoring.db

# Jupyter Notebook
.ipynb_checkpoints

# Model files (uncomment if you don't want to include model in repo)
# exoplanet_model.pth
"""
    create_file('.gitignore', gitignore_content)

    # 5. Update README.md
    readme_content = """# Exoplanet Classification AI

An advanced machine learning application for classifying exoplanets with real-time monitoring and analytics.

## Features

- **AI-Powered Classification**: Neural network trained on 5,000+ exoplanets
- **Interactive Interface**: User-friendly Streamlit web application  
- **Real-time Monitoring**: Built-in prediction tracking and analytics
- **Multiple Planet Types**: Classify Terrestrial, Super Earth, Neptune-like, Gas Giant
- **Comprehensive Analytics**: Confidence analysis and temporal trends

## Live Demo

**[Try the App](https://your-app-name.streamlit.app)** *(Update this URL after deployment)*

## How to Use

1. **Enter Planet Data**: Input observational and physical characteristics
2. **Get Prediction**: AI classifies the planet type with confidence scores
3. **View Analytics**: Monitor prediction trends and model performance
4. **Explore Results**: Detailed probability breakdowns for each classification

## Planet Types Classified

| Type | Description | Examples |
|------|-------------|----------|
| Terrestrial | Rocky planets like Earth | Earth, Mars, Venus |
| Super Earth | Large rocky planets | Kepler-442b, K2-18b |  
| Neptune-like | Gas planets with atmospheres | Neptune, Uranus |
| Gas Giant | Large gas planets | Jupiter, Saturn |

## Technology Stack

- **ML Framework**: PyTorch Neural Networks
- **Frontend**: Streamlit 
- **Visualization**: Plotly Interactive Charts
- **Database**: SQLite for monitoring
- **Deployment**: Streamlit Cloud

## Model Performance

- **Training Data**: 5,250 confirmed exoplanets
- **Accuracy**: 90%+ on test set
- **Features**: 11 astronomical characteristics
- **Architecture**: Deep neural network with dropout and batch normalization

## Local Development

```bash
# Clone repository
git clone https://github.com/jeffersonwarie/exoplanet-classification-ai.git
cd exoplanet-classification-ai

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Data Input Format

The model expects these features:
- Distance (light-years)
- Stellar magnitude
- Discovery year
- Mass multiplier and reference
- Radius multiplier and reference  
- Orbital radius (AU)
- Orbital period (days)
- Eccentricity
- Detection method

## Configuration

The app includes comprehensive monitoring and can be customized through:
- Streamlit configuration (`.streamlit/config.toml`)
- Model parameters (in training script)
- UI themes and styling

## Monitoring Features

- **Prediction Tracking**: All classifications logged with timestamps
- **Confidence Analysis**: Model performance metrics and trends
- **Usage Analytics**: Daily/weekly prediction volumes
- **Performance Monitoring**: Response times and system health

---

"""
    create_file('README.md', readme_content)

    # 6. Create a test script
    test_script = '''#!/usr/bin/env python3
"""
Test script to verify Streamlit app works locally
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def test_app():
    print("Testing Streamlit app locally...")
    
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
        print(f"Missing files: {', '.join(missing_files)}")
        if 'streamlit_exoplanet_app.py' in missing_files:
            print("Copy the main app code from the Streamlit artifact")
        if 'exoplanet_model.pth' in missing_files:
            print("Add your trained model file")
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
        print("Testing imports...")
        import streamlit_exoplanet_app
        print("App imports successfully")
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    print("Starting Streamlit app...")
    print("Opening http://localhost:8501 in your browser...")
    print("Press Ctrl+C to stop the server")
    
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
'''
    create_file('test_local.py', test_script)

    # Make test script executable
    os.chmod('test_local.py', 0o755)

    print(f"""
Setup complete! Files created:

Project Structure:
├── app.py                          # Main entry point
├── requirements.txt               # Dependencies
├── README.md                      # Updated documentation
├── test_local.py                  # Local testing script
├── .gitignore                     # Git ignore rules
└── .streamlit/
    └── config.toml                # Streamlit configuration

⚠️  Still needed:
├── streamlit_exoplanet_app.py     # Copy from Streamlit artifact
└── exoplanet_model.pth           # Your trained model

🚀 Next steps:

1. Copy the main app code:
   - Copy code from Streamlit artifact → streamlit_exoplanet_app.py

2. Add your trained model:
   - Copy exoplanet_model.pth to this directory

3. Test locally:
   python test_local.py

4. Commit and push:
   git add .
   git commit -m "Add Streamlit app files"
   git push origin main

5. Deploy on Streamlit Cloud:
   - Go to share.streamlit.io
   - Connect your GitHub repository
   - Deploy with main file: app.py

Your app will be live at: https://your-repo-name.streamlit.app
""")

if __name__ == "__main__":
    main()
