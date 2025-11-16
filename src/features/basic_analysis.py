"""
DAY 3: PANDAS DATA ANALYSIS
Goal: Analyze your historical price data to understand market behavior
Time: 1-2 hours

CONCEPTS:
- Loading CSV files
- Calculating returns (% price changes)
- Computing volatility (risk measure)
- Finding correlations between assets
- Identifying the most/least volatile cryptocurrencies
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# STEP 1: Load Data from CSV
# ═══════════════════════════════════════════════════════════════

def load_price_data(symbol):
    """
    Load historical price data for a given symbol
    
    Parameters:
        symbol (str): Trading pair (e.g., 'BTCUSDT')
    
    Returns:
        pd.DataFrame: Price data with datetime index
    """
    # TODO 1.1: Construct the filepath
    # The file is in src/data_collection/raw/{symbol.lower()}_1h_1week.csv
    # Hint: Use Path(__file__).parent to get current directory
    
    current_dir = Path(__file__).parent  # Path(__file__).parent
    raw_dir = current_dir/ "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{symbol.lower()}_1h_1week.csv"
    filepath = raw_dir / filename   # current_dir / '../data_collection/raw' / f'{symbol.lower()}_1h_1week.csv'
    
    try:
        # TODO 1.2: Read the CSV file
        df = pd.read_csv(filepath)  # pd.read_csv(filepath)
        
        # TODO 1.3: Convert 'timestamp' column to datetime if it's not already
        # Hint: pd.to_datetime(df['timestamp'])
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        
        
        # TODO 1.4: Set timestamp as the index (makes time-series operations easier)
        # Hint: df.set_index('timestamp', inplace=True)
        df.set_index('timestamp', inplace=True)
        
        return df
        
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# STEP 2: Calculate Returns
# ═══════════════════════════════════════════════════════════════

def calculate_returns(df):
    """
    Calculate percentage returns from price data
    
    Parameters:
        df (pd.DataFrame): Price data with 'close' column
    
    Returns:
        pd.DataFrame: Original data with 'returns' column added
    """
    # TODO 2.1: Calculate returns using pct_change()
    # pct_change() calculates (current - previous) / previous
    # Hint: df['returns'] = df['close'].pct_change()
    
    pct_change() = 
    # TODO 2.2: The first row will be NaN (no previous value), that's okay
    # But let's see how many returns we have
    # print(f"Calculated {df['returns'].count()} returns")
    
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 3: Calculate Basic Statistics
# ═══════════════════════════════════════════════════════════════

def calculate_statistics(df):
    """
    Calculate key statistics for the asset
    
    Returns a dictionary with:
    - mean_return: Average hourly return
    - volatility: Standard deviation of returns (risk measure)
    - min_return: Worst hourly return
    - max_return: Best hourly return
    - sharpe_ratio: Return/Risk ratio (higher is better)
    """
    stats = {}
    
    # TODO 3.1: Calculate mean return
    # Hint: df['returns'].mean()
    stats['mean_return'] = None
    
    # TODO 3.2: Calculate volatility (standard deviation of returns)
    # This measures how much returns fluctuate = risk
    # Hint: df['returns'].std()
    stats['volatility'] = None
    
    # TODO 3.3: Calculate min and max returns
    stats['min_return'] = None  # df['returns'].min()
    stats['max_return'] = None  # df['returns'].max()
    
    # TODO 3.4: Calculate a simple Sharpe-like ratio
    # Sharpe ratio = mean_return / volatility
    # Higher means better risk-adjusted returns
    # Handle divide-by-zero case
    if stats['volatility'] > 0:
        stats['sharpe_ratio'] = None  # mean_return / volatility
    else:
        stats['sharpe_ratio'] = 0
    
    return stats


# ═══════════════════════════════════════════════════════════════
# STEP 4: Analyze Multiple Assets
# ═══════════════════════════════════════════════════════════════

def analyze_portfolio(symbols):
    """
    Analyze multiple cryptocurrencies and compare them
    
    Parameters:
        symbols (list): List of trading pairs to analyze
    
    Returns:
        pd.DataFrame: Summary statistics for all assets
    """
    results = []
    
    # TODO 4.1: Loop through each symbol
    for symbol in symbols:
        print(f"Analyzing {symbol}...")
        
        # TODO 4.2: Load the data
        df = None  # load_price_data(symbol)
        
        if df is None:
            continue
        
        # TODO 4.3: Calculate returns
        df = None  # calculate_returns(df)
        
        # TODO 4.4: Get statistics
        stats = None  # calculate_statistics(df)
        
        # TODO 4.5: Add symbol name to stats dict
        stats['symbol'] = symbol
        
        # TODO 4.6: Add to results list
        # results.append(stats)
    
    # TODO 4.7: Convert list of dicts to DataFrame
    summary_df = None  # pd.DataFrame(results)
    
    # TODO 4.8: Sort by volatility (most volatile first)
    # Hint: summary_df.sort_values('volatility', ascending=False)
    
    
    return summary_df


# ═══════════════════════════════════════════════════════════════
# STEP 5: Find Interesting Assets
# ═══════════════════════════════════════════════════════════════

def find_extremes(summary_df):
    """
    Identify the most/least volatile assets and best risk-adjusted returns
    """
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    # TODO 5.1: Most volatile (highest risk)
    # Hint: summary_df.nlargest(5, 'volatility')
    print("\n📈 MOST VOLATILE (Highest Risk):")
    print("TODO: Print top 5 most volatile")
    
    
    # TODO 5.2: Least volatile (lowest risk)
    # Hint: summary_df.nsmallest(5, 'volatility')
    print("\n📉 LEAST VOLATILE (Lowest Risk):")
    print("TODO: Print top 5 least volatile")
    
    
    # TODO 5.3: Best risk-adjusted returns (highest Sharpe ratio)
    print("\n⭐ BEST RISK-ADJUSTED RETURNS:")
    print("TODO: Print top 5 by Sharpe ratio")
    
    
    # TODO 5.4: Overall statistics
    print("\n📊 OVERALL STATISTICS:")
    print(f"Average volatility: TODO")
    print(f"Average return: TODO")
    print(f"Most common return: TODO (hint: use median)")


# ═══════════════════════════════════════════════════════════════
# STEP 6: Save Analysis Results
# ═══════════════════════════════════════════════════════════════

def save_analysis(summary_df, filename='crypto_analysis_summary.csv'):
    """
    Save the analysis results to CSV
    """
    # TODO 6.1: Create results folder if it doesn't exist
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'tables'
    # os.makedirs(results_dir, exist_ok=True)
    
    # TODO 6.2: Save to CSV
    filepath = None  # results_dir / filename
    # summary_df.to_csv(filepath, index=False)
    
    print(f"\n✅ Analysis saved to {filepath}")


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Your 52 symbols from Day 2
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT',
        'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'UNIUSDT',
        'LTCUSDT', 'BCHUSDT', 'ATOMUSDT', 'ALGOUSDT', 'VETUSDT'
        # Add more if you want, or use all 52
    ]
    
    print("Starting portfolio analysis...")
    
    # TODO: Run the analysis
    # summary = analyze_portfolio(symbols)
    
    # TODO: Print the full summary
    # print("\n" + "="*60)
    # print("FULL SUMMARY")
    # print("="*60)
    # print(summary)
    
    # TODO: Find extremes
    # find_extremes(summary)
    
    # TODO: Save results
    # save_analysis(summary)
    
    print("\n✅ Day 3 Complete!")


# ═══════════════════════════════════════════════════════════════
# WHAT TO SUBMIT:
# ═══════════════════════════════════════════════════════════════
# 1. Your completed code
# 2. Screenshot showing:
#    - Full summary table
#    - Most/least volatile assets
#    - Best risk-adjusted returns
# 3. The saved CSV file with analysis results
#
# QUESTIONS TO THINK ABOUT:
# - Which assets are most volatile? Does this match your intuition?
# - Are memecoins (DOGE, SHIB, PEPE) more volatile than BTC/ETH?
# - Which asset has the best Sharpe ratio?
# - Do stablecoins (USDC, USDT) have near-zero volatility?
# ═══════════════════════════════════════════════════════════════