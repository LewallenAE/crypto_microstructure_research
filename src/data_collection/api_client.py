"""
API Client for Binance Cryptocurrency Data
Provides functions to fetch real-time price data from Binance API
"""


import requests
import json
from datetime import datetime

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
        symbol (str): The trading pair symbol (e.g., "BTCUSDT", "ETHUSDT")

    Returns: 
        float: The current price, or None if there was an error.    
    """

    url = f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}"
    
    try:
        response = requests.get(url)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None


# Test the function
if __name__ == "__main__":
    btc = get_crypto_price("BTCUSDT")
    eth = get_crypto_price("ETHUSDT")
    
    if btc:
        print(f"Bitcoin: ${btc:,.2f}")
    if eth:
        print(f"Ethereum: ${eth:,.2f}")