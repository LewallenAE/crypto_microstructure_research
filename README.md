# Cryptocurrency Market Microstructure Analysis

**A quantitative research project exploring order flow patterns in cryptocurrency markets using statistical learning methods.**

## Overview
This project analyzes high-frequency order book data from cryptocurrency exchanges to identify patterns in market microstructure and develop predictive models for short-term price movements.

## Key Features
- Real-time order book data collection from Binance API
- Statistical analysis of bid-ask spread dynamics
- Feature engineering for market microstructure signals
- Machine learning models for price prediction
- Backtesting framework with performance metrics

## Technologies
- **Python 3.9+**
- **Data**: Binance API (free, no authentication required)
- **Analysis**: pandas, numpy, scipy
- **ML**: scikit-learn
- **Visualization**: matplotlib, seaborn

## Project Structure
[Will fill this in later]

## Installation
```bash
git clone https://github.com/yourusername/crypto-microstructure-research.git
cd crypto-microstructure-research
pip install -r requirements.txt
```

## Quick Start
[Will fill this in later]

## Results
[Will fill this in at the end]


## Technical Notes

### API Endpoint
This project uses `api.binance.us` instead of `api.binance.com` due to US regulatory restrictions. The endpoints are functionally identical but `binance.us` has slightly fewer trading pairs available.

**Why this matters for reproducibility:**
- Non-US users should use `api.binance.com`
- US users must use `api.binance.us`
- The code is designed to work with both

## Author
Anthony Eugene Lewallen - [LinkedIn] - [Email]