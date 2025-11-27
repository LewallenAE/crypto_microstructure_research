"""
API Client for Binance Cryptocurrency Data
Provides functions to fetch real-time price data and order book data from Binance API

This module provides a simple interface to the Binance.US REST API for fetching real-time and historical cryptocurrency price data.

Key Functions:
    get_crypto_price(symbol): Fetch current price for a trading pair

Dependencies:
    - requests: HTTP client

API documentation:
    https://docs.binance.us/

Author: Anthony Eugene Lewallen

Date: November 2025
Version: 1.0
"""


import requests
import json
from datetime import datetime
import time

# API URL for binance crypto
url_btc = "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT"
url_eth = "https://api.binance.us/api/v3/ticker/price?symbol=ETHUSDT"

# store get url request as response
response_btc = requests.get(url_btc)
response_eth = requests.get(url_eth)

# store the response converted to json as data
data_btc = response_btc.json()
data_eth = response_eth.json()

btc_price = float(data_btc['price'])
eth_price = float(data_eth['price'])


print(f"Bitcoin current price: {btc_price:,.2f}")
print(f"Ethereum current price: {eth_price:,.2f}")


def get_crypto_price(symbol):
    """    
    Fetches the current price of a cryptocurrency from Binance
    Parameters:
    -----------
        symbol (str): The trading pair symbol (e.g., "BTCUSDT", "ETHUSDT")

    Returns: 
    --------
        float: The current price, or None if there was an error.

    Examples
    --------
    >>> price = get_crypto_price('BTCUSDT')
    >>> print(f"BTC Price: ${price:,.2f}")
    BTC Price: $95,234.56 

    Notes
    -----
    - This function does not require authentication
    - Rate limits apply (see Binance.US documentation)
    - Returns None on any error (network, invalid symbol, etc.)
    
    See Also
    --------
    historical.py : For fetching historical OHLCV data   
    """

    url = f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}"
    
    try:
        response = requests.get(url)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None


def get_order_book(symbol, limit=20):
    """
    Fetches the current order book (bids and asks) for a cryptocurrency pair.
    
    This is CRITICAL for microstructure analysis - shows supply/demand at different price levels.
    
    Parameters:
        symbol (str): The trading pair symbol (e.g., "BTCUSDT", "ETHUSDT")
        limit (int): Number of price levels to fetch (5, 10, 20, 50, 100, 500, 1000, 5000)
    
    Returns:
        dict: Order book data with structure:
            {
                'timestamp': Unix timestamp in milliseconds,
                'symbol': Trading pair,
                'bids': List of [price, quantity] for buy orders,
                'asks': List of [price, quantity] for sell orders
            }
        None if there was an error
    
    Example:
        >>> book = get_order_book("BTCUSDT", limit=10)
        >>> print(f"Best bid: ${book['bids'][0][0]}")
        >>> print(f"Best ask: ${book['asks'][0][0]}")
    """
    url = f"https://api.binance.us/api/v3/depth?symbol={symbol}&limit={limit}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Structure the data for analysis
        order_book = {
            'timestamp': data['lastUpdateId'],  # Binance's update ID (acts as timestamp)
            'symbol': symbol,
            'bids': [[float(price), float(qty)] for price, qty in data['bids']],
            'asks': [[float(price), float(qty)] for price, qty in data['asks']],
            'captured_at': datetime.now().isoformat()
        }
        
        return order_book
        
    except Exception as e:
        print(f"Error fetching order book for {symbol}: {e}")
        return None


def get_recent_trades(symbol, limit=100):
    """
    Fetches recent trades (actual executed transactions) for a cryptocurrency pair.
    
    Trade data reveals market dynamics: large trades, aggressive buying/selling, volume patterns.
    
    Parameters:
        symbol (str): The trading pair symbol (e.g., "BTCUSDT", "ETHUSDT")
        limit (int): Number of recent trades to fetch (max 1000)
    
    Returns:
        list: List of trade dictionaries with structure:
            {
                'id': Trade ID,
                'price': Execution price,
                'qty': Quantity traded,
                'time': Unix timestamp in milliseconds,
                'isBuyerMaker': True if buyer was passive (limit order)
            }
        None if there was an error
    """
    url = f"https://api.binance.us/api/v3/trades?symbol={symbol}&limit={limit}"
    
    try:
        response = requests.get(url)
        trades = response.json()
        
        # Clean and structure trade data
        processed_trades = []
        for trade in trades:
            processed_trades.append({
                'id': trade['id'],
                'price': float(trade['price']),
                'qty': float(trade['qty']),
                'time': trade['time'],
                'isBuyerMaker': trade['isBuyerMaker'],
                'symbol': symbol
            })
        
        return processed_trades
        
    except Exception as e:
        print(f"Error fetching trades for {symbol}: {e}")
        return None


