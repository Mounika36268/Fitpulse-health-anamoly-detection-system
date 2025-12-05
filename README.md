🩺 FitPulse – Health Anomaly Detection System

A Machine Learning–powered application that detects health anomalies using clinical and sensor-based data. The system preprocesses data, performs feature engineering, applies clustering & classification models, and provides anomaly detection insights through an interactive dashboard.

🚀 Features
✔️ Data Processing

Missing value handling

Outlier detection

Feature scaling (MinMax / Standard scaling)

✔️ Feature Engineering

PCA for dimensionality reduction

Feature clustering

Correlation heatmaps

Visual analysis using Plotly

✔️ Machine Learning Models

KMeans clustering

Isolation Forest / LOF for anomaly detection

Classification models (RandomForest, XGBoost, etc.)

✔️ Interactive Application (Streamlit)

Upload your dataset

Visualize data instantly

Run anomaly detection

Generate detailed analysis reports

📂 Project Structure
Fitpulse-health-anamoly-detection-system/
Dataset/                 # Sample and training datasets
Structure/               # Project modular structure
 app.py                   # Main Streamlit application
code_file.py             # Additional ML utility code
create.py                # Model creation logic
 data_create.py           # Data preprocessing scripts
 mile_stone.py            # Progress tracking script
 librarie.txt             # Required libraries list
 README.md                # Project documentation
 .gitignore               # Git ignore file


 📊 Output Examples

Anomaly Detection Results

PCA Clustering Visualization

Feature Correlation Map

Data Quality Report

🤖 Machine Learning Workflow

Load and clean data

Perform EDA and feature engineering

Train ML models

Detect anomalies and score the data

Present results via interactive UI

📝 Requirements

Python 3.8+

Streamlit

Scikit-Learn

Pandas

NumPy

Plotly

Matplotlib / Seaborn