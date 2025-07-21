"""
Streamlit Exoplanet Classification App with Built-in Monitoring
Deployable on Streamlit Cloud
"""

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import os
from typing import Dict, Any, List
import logging

# Configure page
st.set_page_config(
    page_title="🪐 Exoplanet Classification AI",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Neural Network Model Definition
class ExoplanetClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, num_classes: int, dropout_rate: float = 0.3):
        super(ExoplanetClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ExoplanetPredictor:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.target_encoder = None
        self.load_model(model_path)
    
    @st.cache_resource
    def load_model(_self, model_path: str):
        """Load the trained model with sklearn compatibility"""
        try:
            # This tells PyTorch to trust sklearn objects in your checkpoint
            # Only do this for models you created yourself or trust completely
            checkpoint = torch.load(model_path, map_location=_self.device, weights_only=False)
            
            _self.scaler = checkpoint['scaler']
            _self.label_encoders = checkpoint['label_encoders']
            _self.target_encoder = checkpoint['target_encoder']
            config = checkpoint['config']
            
            input_size = len(_self.scaler.mean_)
            num_classes = len(_self.target_encoder.classes_)
            
            _self.model = ExoplanetClassifier(
                input_size=input_size,
                hidden_sizes=config['hidden_sizes'],
                num_classes=num_classes,
                dropout_rate=config['dropout_rate']
            ).to(_self.device)
            
            _self.model.load_state_dict(checkpoint['model_state_dict'])
            _self.model.eval()
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def preprocess_input(self, data: Dict[str, Any]) -> np.ndarray:
        """Preprocess input data for prediction"""
        df = pd.DataFrame([data])
        
        # Handle missing values and encoding
        categorical_cols = ['mass_wrt', 'radius_wrt', 'detection_method']
        for col in categorical_cols:
            if col in self.label_encoders and col in df.columns:
                # Handle unknown categories
                df[col] = df[col].apply(
                    lambda x: x if x in self.label_encoders[col].classes_ else self.label_encoders[col].classes_[0]
                )
                df[col] = self.label_encoders[col].transform(df[col])
        
        # Scale features
        X_scaled = self.scaler.transform(df.values.astype(np.float32))
        return X_scaled
    
    def predict(self, data: Dict[str, Any]):
        """Make prediction"""
        if self.model is None:
            return None, None
        
        X_processed = self.preprocess_input(data)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted_indices = torch.max(outputs, 1)
        
        predicted_class = self.target_encoder.inverse_transform(predicted_indices.cpu().numpy())[0]
        probs = probabilities.cpu().numpy()[0]
        
        prob_dict = {
            class_name: float(prob)
            for class_name, prob in zip(self.target_encoder.classes_, probs)
        }
        
        return predicted_class, prob_dict

# Simple monitoring system for Streamlit
class StreamlitMonitoring:
    def __init__(self):
        self.db_path = "streamlit_monitoring.db"
        self.init_db()
    
    def init_db(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                prediction TEXT,
                confidence REAL,
                input_data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metric_type TEXT,
                value REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, prediction: str, confidence: float, input_data: Dict[str, Any]):
        """Log a prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (timestamp, prediction, confidence, input_data)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            prediction,
            confidence,
            json.dumps(input_data)
        ))
        
        conn.commit()
        conn.close()
    
    def get_prediction_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get prediction statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent predictions
        df = pd.read_sql_query('''
            SELECT * FROM predictions 
            WHERE timestamp > datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days), conn)
        
        conn.close()
        
        if df.empty:
            return {
                'total_predictions': 0,
                'avg_confidence': 0,
                'prediction_distribution': {},
                'recent_predictions': []
            }
        
        stats = {
            'total_predictions': len(df),
            'avg_confidence': df['confidence'].mean(),
            'prediction_distribution': df['prediction'].value_counts().to_dict(),
            'recent_predictions': df.head(10).to_dict('records')
        }
        
        return stats

# Initialize monitoring
@st.cache_resource
def get_monitoring():
    return StreamlitMonitoring()

# Initialize predictor
@st.cache_resource
def get_predictor():
    model_path = "exoplanet_model.pth"
    if os.path.exists(model_path):
        predictor = ExoplanetPredictor(model_path)
        return predictor
    else:
        st.error(f"Model file '{model_path}' not found. Please upload your trained model.")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🪐 Exoplanet Classification AI</h1>', unsafe_allow_html=True)
    st.markdown("**Classify exoplanets using advanced machine learning with real-time monitoring**")
    
    # Initialize components
    monitoring = get_monitoring()
    predictor = get_predictor()
    
    # Sidebar navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Choose a page:", [
        "🔬 Classify Exoplanet", 
        "📊 Monitoring Dashboard",
        "📈 Analytics",
        "ℹ️ About"
    ])
    
    if page == "🔬 Classify Exoplanet":
        show_classification_page(predictor, monitoring)
    elif page == "📊 Monitoring Dashboard":
        show_monitoring_dashboard(monitoring)
    elif page == "📈 Analytics":
        show_analytics_page(monitoring)
    elif page == "ℹ️ About":
        show_about_page()

def show_classification_page(predictor, monitoring):
    """Main classification interface"""
    
    if predictor is None:
        st.error("⚠️ Model not loaded. Please check that 'exoplanet_model.pth' is in the repository root.")
        st.info("Upload your trained model file to the repository root and redeploy the app.")
        return
    
    st.header("🔬 Exoplanet Classification")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 Observational Data")
        distance = st.number_input("Distance (light-years)", min_value=0.1, value=150.5, step=0.1)
        stellar_magnitude = st.number_input("Stellar Magnitude", value=12.5, step=0.1)
        discovery_year = st.number_input("Discovery Year", min_value=1990, max_value=2030, value=2020, step=1)
        
        st.subheader("🪨 Physical Properties")
        mass_multiplier = st.number_input("Mass Multiplier", min_value=0.01, value=5.2, step=0.01)
        mass_wrt = st.selectbox("Mass Reference", ["Earth", "Jupiter"])
        
        radius_multiplier = st.number_input("Radius Multiplier", min_value=0.01, value=2.1, step=0.01)
        radius_wrt = st.selectbox("Radius Reference", ["Earth", "Jupiter"])
    
    with col2:
        st.subheader("🌌 Orbital Characteristics")
        orbital_radius = st.number_input("Orbital Radius (AU)", min_value=0.001, value=0.1, step=0.001, format="%.3f")
        orbital_period = st.number_input("Orbital Period (days)", min_value=0.001, value=10.5, step=0.001, format="%.3f")
        eccentricity = st.slider("Eccentricity", min_value=0.0, max_value=1.0, value=0.05, step=0.001)
        
        st.subheader("🔍 Detection Method")
        detection_method = st.selectbox("Detection Method", [
            "Transit", "Radial Velocity", "Gravitational Microlensing",
            "Direct Imaging", "Transit Timing Variations", "Eclipse Timing Variations",
            "Pulsar Timing", "Orbital Brightness Modulation", "Astrometry",
            "Pulsation Timing Variations", "Disk Kinematics"
        ])
    
    # Prediction button
    if st.button("🚀 Classify Exoplanet", type="primary"):
        # Prepare input data
        input_data = {
            'distance': distance,
            'stellar_magnitude': stellar_magnitude,
            'discovery_year': discovery_year,
            'mass_multiplier': mass_multiplier,
            'mass_wrt': mass_wrt,
            'radius_multiplier': radius_multiplier,
            'radius_wrt': radius_wrt,
            'orbital_radius': orbital_radius,
            'orbital_period': orbital_period,
            'eccentricity': eccentricity,
            'detection_method': detection_method
        }
        
        # Make prediction
        with st.spinner("🤖 Analyzing exoplanet characteristics..."):
            prediction, probabilities = predictor.predict(input_data)
            
            if prediction is not None:
                confidence = max(probabilities.values())
                
                # Log prediction
                monitoring.log_prediction(prediction, confidence, input_data)
                
                # Display results
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>🎯 Classification Result</h2>
                    <h3>Planet Type: {prediction}</h3>
                    <h4>Confidence: {confidence:.1%}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution
                st.subheader("📊 Classification Probabilities")
                
                prob_df = pd.DataFrame(list(probabilities.items()), columns=['Planet Type', 'Probability'])
                prob_df = prob_df.sort_values('Probability', ascending=True)
                
                fig = px.bar(prob_df, x='Probability', y='Planet Type', orientation='h',
                           title="Classification Confidence by Planet Type",
                           color='Probability', color_continuous_scale='viridis')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed probabilities
                st.subheader("🔢 Detailed Probabilities")
                for planet_type, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(planet_type)
                    with col2:
                        st.progress(prob)
                    with col3:
                        st.write(f"{prob:.1%}")
    
    # Example data button
    if st.button("📋 Load Example Data"):
        st.session_state.update({
            'example_loaded': True
        })
        st.rerun()

def show_monitoring_dashboard(monitoring):
    """Monitoring dashboard"""
    st.header("📊 Monitoring Dashboard")
    
    # Get statistics
    stats = monitoring.get_prediction_stats(days=30)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", stats['total_predictions'])
    
    with col2:
        st.metric("Average Confidence", f"{stats['avg_confidence']:.1%}")
    
    with col3:
        most_common = max(stats['prediction_distribution'].items(), key=lambda x: x[1]) if stats['prediction_distribution'] else ("N/A", 0)
        st.metric("Most Common Type", most_common[0])
    
    with col4:
        st.metric("Predictions Today", len([p for p in stats['recent_predictions'] 
                                          if datetime.fromisoformat(p['timestamp']).date() == datetime.now().date()]))
    
    if stats['total_predictions'] > 0:
        # Prediction distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🪐 Planet Type Distribution")
            if stats['prediction_distribution']:
                fig = px.pie(
                    values=list(stats['prediction_distribution'].values()),
                    names=list(stats['prediction_distribution'].keys()),
                    title="Classification Results Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Confidence Trends")
            if stats['recent_predictions']:
                recent_df = pd.DataFrame(stats['recent_predictions'])
                recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp'])
                
                fig = px.line(recent_df, x='timestamp', y='confidence',
                            title="Prediction Confidence Over Time",
                            labels={'confidence': 'Confidence', 'timestamp': 'Time'})
                fig.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent predictions table
        st.subheader("🕒 Recent Predictions")
        if stats['recent_predictions']:
            recent_df = pd.DataFrame(stats['recent_predictions'])
            recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            recent_df['confidence'] = recent_df['confidence'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(
                recent_df[['timestamp', 'prediction', 'confidence']].head(10),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("No predictions yet. Make some classifications to see monitoring data!")

def show_analytics_page(monitoring):
    """Advanced analytics page"""
    st.header("📈 Advanced Analytics")
    
    stats = monitoring.get_prediction_stats(days=30)
    
    if stats['total_predictions'] > 0:
        # Time-based analysis
        st.subheader("⏰ Temporal Analysis")
        
        # Daily predictions
        recent_df = pd.DataFrame(stats['recent_predictions'])
        if not recent_df.empty:
            recent_df['date'] = pd.to_datetime(recent_df['timestamp']).dt.date
            daily_counts = recent_df.groupby('date').size().reset_index(name='predictions')
            
            fig = px.bar(daily_counts, x='date', y='predictions',
                        title="Daily Prediction Volume")
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence analysis
        st.subheader("🎯 Confidence Analysis")
        
        if not recent_df.empty:
            confidence_bins = pd.cut(recent_df['confidence'], bins=[0, 0.5, 0.7, 0.9, 1.0], 
                                   labels=['Low (0-50%)', 'Medium (50-70%)', 'High (70-90%)', 'Very High (90-100%)'])
            confidence_dist = confidence_bins.value_counts()
            
            fig = px.bar(x=confidence_dist.index, y=confidence_dist.values,
                        title="Confidence Score Distribution",
                        labels={'x': 'Confidence Range', 'y': 'Number of Predictions'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for analytics. Make some predictions first!")

def show_about_page():
    """About page with information"""
    st.header("ℹ️ About Exoplanet Classification AI")
    
    st.markdown("""
    ## 🌟 What is this app?
    
    This application uses advanced machine learning to classify exoplanets based on their physical and orbital characteristics. 
    It's trained on a dataset of over 5,000 confirmed exoplanets and can identify different planet types with high accuracy.
    
    ## 🪐 Planet Types
    
    The system can classify planets into these categories:
    
    - **🌍 Terrestrial**: Rocky planets similar to Earth
    - **🌎 Super Earth**: Rocky planets larger than Earth but smaller than Neptune
    - **🌀 Neptune-like**: Gas planets similar to Neptune with thick atmospheres
    - **🪨 Gas Giant**: Large gas planets like Jupiter and Saturn
    - **❓ Unknown**: Planets with uncertain or unique characteristics
    
    ## 🔬 How it works
    
    The model uses a deep neural network trained on features including:
    - Distance from Earth
    - Host star brightness
    - Planet mass and radius
    - Orbital characteristics
    - Detection method used
    
    ## 📊 Built-in Monitoring
    
    This app includes comprehensive monitoring features:
    - Real-time prediction tracking
    - Confidence analysis
    - Performance metrics
    - Temporal trends analysis
    
    ## 🚀 Technology Stack
    
    - **Machine Learning**: PyTorch neural networks
    - **Frontend**: Streamlit
    - **Visualization**: Plotly
    - **Monitoring**: SQLite database
    - **Deployment**: Streamlit Cloud
    
    ## 📝 Data Sources
    
    Based on the NASA Exoplanet Archive and other astronomical databases containing 
    confirmed exoplanet discoveries through 2023.
    
    ---
    
    **Made with ❤️ for space exploration and machine learning**
    """)

if __name__ == "__main__":
    main()
