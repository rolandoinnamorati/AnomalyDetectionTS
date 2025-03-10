# 🚀 Anomaly Detection for Corporate Fleets  

A machine learning system for **real-time anomaly detection** in fleet data using **autoencoders and forecasting models**.

## 📌 Overview  
This project detects anomalies in **streaming vehicle data**, leveraging **unsupervised learning** techniques. The system is designed to work within the **GeoSat** infrastructure, analyzing **GPS and sensor data** from corporate fleets.  

## 🏗️ Architecture  
The system combines two approaches for anomaly detection:  
- **Autoencoder** – Learns a compressed representation of normal data and detects anomalies via **reconstruction error**.  
- **Forecasting Model** (coming soon) – Predicts future values and detects anomalies via **prediction error**.  

## 📊 Data Processing  
- **Input:** Rolling windows of **20 time steps**, with **4 temporal features + 10 vehicle features**.  
- **Preprocessing:** Standardization (Z-score), feature extraction, and temporal windowing.  

## 🏢 Repository Structure  
- anomaly-detection-ts/ 
  - │── data/ # Dataset and preprocessing scripts
  - │── models/ # Machine learning models
  - │── notebooks/ # Jupyter Notebooks for analysis
  - │── utils/ # Helper functions
  - │── config/ # Configuration files
  - │── scripts/ # Execution scripts
  - │── tests/ # Unit tests
  - │── requirements.txt # Dependencies
  - │── README.md # Documentation
  - │── .gitignore # Ignore unnecessary files

## 🚀 Quick Start  
1️⃣ **Clone the repository**  
```bash
git clone https://github.com/your-username/anomaly-detection-ts.git
cd anomaly-detection-ts
pip install -r requirements.txt
python models/train_autoencoder.py
python scripts/detect_anomalies.py
