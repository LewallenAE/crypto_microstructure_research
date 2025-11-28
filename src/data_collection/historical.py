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

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

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

symbols = ['AAVEUSDT', 'ALGOUSDT', 'AVAXUSDT', 'BTCUSDT', 'BCHUSDT', 'BNBUSDT', 
           'BONKUSDT', 'ADAUSDT', 'LINKUSDT', 'ATOMUSDT', 'CRVUSDT', 'DGBUSDT', 
           'DOGEUSDT', 'ETHUSDT', 'ETCUSDT', 'ENSUSDT', 'FETUSDT', 'FLOKIUSDT', 
           'GALAUSDT', 'ONEUSDT', 'HBARUSDT', 'HYPEUSDT', 'ICPUSDT', 'IOTAUSDT', 
           'JUPUSDT', 'LTCUSDT', 'LPTUSDT', 'MEUSDT', 'NEARUSDT', 'TRUMPUSDT', 
           'OPUSDT', 'PEPEUSDT', 'DOTUSDT', 'POLUSDT', 'RVNUSDT', 'RENDERUSDT', 
           'SHIBUSDT', 'SOLUSDT', 'SUSDT', 'USDCUSDT', 'XLMUSDT', 'SUIUSDT', 
           'SUSHIUSDT', 'GRTUSDT', 'SANDUSDT', 'THETAUSDT', 'UNIUSDT', 'USDUSDT', 
           'VETUSDT', 'VTHOUSDT', 'XRPUSDT', 'ZILUSDT']



def get_historical_prices_extended(symbol, interval='1h', total_hours=2000):
    """
    Fetch extended historical data by making multiple API calls
    
    Binance limits us to 1000 candles per request, so we make 2 calls
    to get 2000 hours of data.
    
    Parameters:
        symbol (str): Trading pair (e.g., 'BTCUSDT')
        interval (str): Time interval - '1m', '5m', '1h', '1d', etc.
        total_hours (int): Total hours to fetch (we'll do 2000)
    
    Returns:
        list: Combined list of candlesticks from multiple calls
    """
    
    # TODO 1.1: Construct the URL
    # Base URL: https://api.binance.us/api/v3/klines
    # Parameters: symbol, interval, limit
    # Example: https://api.binance.us/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=100
    all_data = []
    hours_per_call = 1000
    num_calls = (total_hours // hours_per_call) 

    end_time = int(datetime.now().timestamp() * 1000)

    print(f" Fetching {total_hours} hours in {num_calls} batches...")
    
    for batch in range(num_calls):

        hours_back = hours_per_call * (batch + 1)
        start_time = end_time - (hours_back * 3600 * 1000)

        url_01 = (f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}&limit=1000")

        try:
            print(f" Batch {batch + 1} / {num_calls}...", end=" ")
            response_01 = requests.get(url_01)         
            data = response_01.json()

            if isinstance(data, list) and len(data) > 0:
                all_data.extend(data)
                print(f' Got {len(data)} candles')
            else:
                print(f'No data, {symbol} might be too new')
                break
            end_time = start_time
            time.sleep(0.3)

        except Exception as e:
            print(f"x Error: {e}")
            break

    if len(all_data) == 0:
        return None
    
    
        
    unique_data = {candle[0]: candle for candle in all_data}

    sorted_data = sorted(unique_data.values(), key=lambda x: x[0])

    print(f" Total: {len(sorted_data)} unqiue candles collected")

    return sorted_data


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

def fetch_and_parse(symbol, interval='1h', total_hours = 2000):
    """
    Fetch historical data and return as clean DataFrame
    
    This combines get_historical_prices() and parse_candlesticks()
    """
    raw_data = get_historical_prices_extended(symbol, interval = interval, total_hours = total_hours)  
    
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




# ═══════════════════════════════════════════════════════════════
# STEP 5: Test Everything
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 70)
    print("COLLECTION 2,000 HOURS of HISTORICAL DATA")
    print("=" * 70)
    print(f"\nThis will take about 5 - 10 minutes for {len(symbols)} symbols....")
    print("2 API calls per symbol with 300ms delay") 

    successful = 0
    failed = 0


    for i, sym in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {sym}")

        data = fetch_and_parse(sym, interval='1h', total_hours = 2000)
    
        if data is not None:
            #print("\nFirst 5 candles:")
            #print(f"{data.head()}")
            #print("\nBasic Statistics:")
            #print(f"{data.describe()}")
            filename = f"{sym.lower()}_1h_2000h.csv"
            save_to_csv(data,filename)
            successful += 1
        else:
            failed += 1

    print("\n" + "="*70)
    print("COLLECTION COMPLETE!")
    print("="*70)
    print(f"✓ Successful: {successful} symbols")
    print(f"✗ Failed: {failed} symbols (likely too new)")
    print(f"\nData saved to: src/data_collection/raw/")
    print("\nNext steps:")
    print("  1. Run: python src/features/basic_analysis.py")
    print("  2. Run: python src/features/visualization.py")
    print("  3. Run: python src/features/correlation.py")
    print("  4. Run: python src/features/cointegration.py")
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