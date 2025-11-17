import os
import requests
import pandas as pd
from datetime import datetime, timedelta


"""
═══════════════════════════════════════════════════════════════
DAY 2: HISTORICAL PRICE DATA - Building Your Dataset
═══════════════════════════════════════════════════════════════

LEARNING OBJECTIVES:
- Fetch historical price data (not just current price)
- Understand timestamps and time intervals
- Work with lists of data
- Save data to CSV file

NEW PACKAGE NEEDED:
    pip install pandas

TIME: 1-2 hours
═══════════════════════════════════════════════════════════════
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════
# CONCEPT: Candlestick Data (OHLCV)
# ═══════════════════════════════════════════════════════════════
# In trading, we use "candlesticks" to represent price over time intervals
# Each candle has:
# - Open: price at start of interval
# - High: highest price during interval
# - Low: lowest price during interval
# - Close: price at end of interval
# - Volume: how much was traded
#
# For example, a "1 hour candle" tells you what happened in 1 hour


# ═══════════════════════════════════════════════════════════════
# STEP 1: Fetch Historical Data
# ═══════════════════════════════════════════════════════════════

def get_historical_prices(symbol, interval='1h', limit=100):
    """
    Fetch historical candlestick data from Binance
    
    Parameters:
        symbol (str): Trading pair (e.g., 'BTCUSDT')
        interval (str): Time interval - '1m', '5m', '1h', '1d', etc.
        limit (int): Number of candles to fetch (max 1000)
    
    Returns:
        list: List of candlesticks, each is [timestamp, open, high, low, close, volume, ...]
    """
    
    # TODO 1.1: Construct the URL
    # Base URL: https://api.binance.us/api/v3/klines
    # Parameters: symbol, interval, limit
    # Example: https://api.binance.us/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=100
    
    url_01 = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    
    try:
        
        response_01 = requests.get(url_01)         
        data = response_01.json()          
        return data
        
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# STEP 2: Convert Raw Data to Clean DataFrame
# ═══════════════════════════════════════════════════════════════

def parse_candlesticks(raw_data):
    """
    Convert raw Binance candlestick data to clean pandas DataFrame
    
    Parameters:
        raw_data (list): Raw data from Binance API
    
    Returns:
        pd.DataFrame: Cleaned data with columns: timestamp, open, high, low, close, volume
    """
    
   
    candles = []
    
    for candle in raw_data:
        # Each candle looks like:
        # [timestamp, open, high, low, close, volume, close_time, ...]
        # We only need the first 6 values

        candles.append({

            # convert to miliseconds
            'timestamp': datetime.fromtimestamp(candle[0] / 1000),
            # convert from str to float
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low':  float(candle[3]),
            'close': float(candle[4]),
            'volume': float(candle[5])
        })
    df = pd.DataFrame(candles)     
    return df





# ═══════════════════════════════════════════════════════════════
# STEP 3: Combine into One Function
# ═══════════════════════════════════════════════════════════════

def fetch_and_parse(symbol, interval='1h', limit=100):
    """
    Fetch historical data and return as clean DataFrame
    
    This combines get_historical_prices() and parse_candlesticks()
    """
    raw_data = get_historical_prices(symbol, interval = interval, limit = limit)  
    
    if raw_data is None or len(raw_data) == 0 or not isinstance(raw_data, list):
        print(f"Skipping {symbol} - Binance returned invalid data")
        return None
    
    df = parse_candlesticks(raw_data)  
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 4: Save to CSV
# ═══════════════════════════════════════════════════════════════

def save_to_csv(df, filename):
    """
    Save DataFrame to CSV file in data/raw/ folder
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "raw")
    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, filename)

    print(f"Saving to: {filepath}")    
    df.to_csv(filepath, index = False)    
    print(f"Data saved to {filepath}")


symbols = ['AAVEUSDT', 'ALGOUSDT', 'AVAXUSDT', 'BTCUSDT', 'BCHUSDT', 'BNBUSDT', 'BONKUSDT', 'ADAUSDT', 'LINKUSDT', 'ATOMUSDT', 'CRVUSDT', 'DGBUSDT', 'DOGEUSDT', 'ETHUSDT', 'ETCUSDT', 'ENSUSDT', 'FETUSDT', 'FLOKIUSDT', 'GALAUSDT', 'ONEUSDT', 'HBARUSDT', 'HYPEUSDT', 'ICPUSDT', 'IOTAUSDT', 'JUPUSDT', 'LTCUSDT', 'LPTUSDT', 'MEUSDT', 'NEARUSDT', 'TRUMPUSDT', 'OPUSDT', 'PEPEUSDT', 'DOTUSDT', 'POLUSDT', 'RVNUSDT', 'RENDERUSDT', 'SHIBUSDT', 'SOLUSDT', 'SUSDT', 'USDCUSDT', 'XLMUSDT', 'SUIUSDT', 'SUSHIUSDT', 'GRTUSDT', 'SANDUSDT', 'THETAUSDT', 'UNIUSDT', 'USDUSDT', 'VETUSDT', 'VTHOUSDT', 'XRPUSDT', 'ZILUSDT']

# ═══════════════════════════════════════════════════════════════
# STEP 5: Test Everything
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    for sym in symbols:
        print(f"\nFetching {sym} historical data...")
        data = fetch_and_parse(sym, interval='1h', limit = 200)
    
        if data is not None:
            print("\nFirst 5 candles:")
            print(f"{data.head()}")
            print("\nBasic Statistics:")
            print(f"{data.describe()}")
            filename = f"{sym.lower()}_1h_1week.csv"
            save_to_csv(data,filename)


# ═══════════════════════════════════════════════════════════════
# WHAT TO SUBMIT:
# ═══════════════════════════════════════════════════════════════
# 1. Your completed code
# 2. Screenshot showing:
#    - First 5 rows printed
#    - Basic statistics printed
#    - Confirmation that CSV was saved
# 3. The CSV file (or screenshot of it opened in Excel/text editor)
#
# BONUS: Try fetching data for ETH and SOL too, save as separate CSV files
# ═══════════════════════════════════════════════════════════════