# Melate Predictor 🎲

A lottery number predictor for the Mexican lottery *Melate* using Machine Learning.  
This project combines **Random Forest** (sklearn) and a **Neural Network** (Keras) 
with feature engineering from historical data to generate top-K number predictions.

## Features
- Historical data parsing from Excel
- Multi-label classification (numbers 1..N)
- Feature engineering:
  - Calendar features (month, weekday, year)
  - Frequency counts
  - Rolling window recency
- Models:
  - Random Forest (multi-output classifier)
  - Neural Network (sigmoid output per number)
- Combined prediction strategy

## Installation
```bash
git clone https://github.com/TU_USUARIO/melate-predictor.git
cd melate-predictor
pip install -r requirements.txt
