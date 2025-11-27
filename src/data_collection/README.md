# Data Collection Module

## Overview
This module handles all interactions with the Binance.US API for collecting cryptocurrency market data.

## Components

### `api_client.py`
Real-time price fetching for individual trading pairs.

**Key Functions:**
- `get_crypto_price(symbol)`: Get current price

**Usage:**
```python
from src.data_collection.api_client import get_crypto_price

price = get_crypto_price('BTCUSDT')
print(f"Bitcoin: ${price:,.2f}")
```

### `historical.py`
Historical OHLCV (Open, High, Low, Close, Volume) data collection.

**Key Functions:**
- `get_historical_prices(symbol, interval, limit)`: Fetch raw candlestick data
- `parse_candlesticks(raw_data)`: Convert to clean DataFrame
- `fetch_historical_data(symbol, interval, limit)`: Combined fetch+parse

**Usage:**
```python
from src.data_collection.historical import fetch_historical_data

# Get 200 hours of BTC data
df = fetch_historical_data('BTCUSDT', interval='1h', limit=200)
print(df.head())
```

## Data Storage

All collected data is stored in `src/data_collection/raw/` as CSV files:
- Format: `{symbol.lower()}_1h_1week.csv`
- Columns: timestamp, open, high, low, close, volume

## API Information

**Endpoint:** https://api.binance.us/api/v3/
**Authentication:** None required (public endpoints)
**Rate Limits:** 
- Weight: 1 per request
- Limit: 1200 requests per minute

## Dependencies
- requests==2.31.0
- pandas==2.1.0

## Future Enhancements
- [ ] Async data collection (Week 1, Day 2)
- [ ] WebSocket real-time streaming (Week 15)
- [ ] Error retry logic with exponential backoff
- [ ] Data validation and cleaning pipeline
```

---

## 📝 STUDY (15 minutes)

**Probability Review - Zhou Ch 1-2**

**Problem 1: Coin Flip Expected Value**
```
You flip a fair coin. If heads, you win $2. If tails, you lose $1.
What is the expected value of this game?
```

**Solution:**
```
E[X] = P(Heads) × Payoff(Heads) + P(Tails) × Payoff(Tails)
E[X] = 0.5 × $2 + 0.5 × (-$1)
E[X] = $1 - $0.50
E[X] = $0.50

Expected value: $0.50 per flip (positive EV game)