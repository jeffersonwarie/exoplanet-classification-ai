# Exoplanet Classification AI

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

## 🚀 Local Development

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