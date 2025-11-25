"""
DAY 4: DATA VISUALIZATION
Goal: Create professional charts to understand and present your analysis
Time: 1-2 hours

CONCEPTS:
- Plotting time series data
- Creating histograms and distributions
- Building comparison charts
- Subplots and multi-panel figures
- Saving publication-quality images

INSTALL:
pip install matplotlib seaborn
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Import your functions from Day 3
from src.features.basic_analysis import load_price_data, calculate_returns


# ═══════════════════════════════════════════════════════════════
# STEP 1: Plot Price Over Time
# ═══════════════════════════════════════════════════════════════

def plot_price_history(symbol, save_fig=True):
    """
    Create a clean time series plot of price history
    
    Parameters:
        symbol (str): Trading pair (e.g., 'BTCUSDT')
        save_fig (bool): Whether to save the figure
    """
    # TODO 1.1: Load the data
    df = load_price_data(symbol)  # Use load_price_data from Day 3
    
    if df is None:
        return
    
    # TODO 1.2: Create a figure and axis
    # Hint: fig, ax = plt.subplots(figsize=(12, 6))
    fig, ax = plt.subplots(figsize=(12,6))
    
    # TODO 1.3: Plot the close price
    # Hint: ax.plot(df.index, df['close'], linewidth=2, color='blue')
    ax.plot(df.index, df['close'], linewidth=2, color='blue')
    
    
    # TODO 1.4: Add labels and title
    # ax.set_xlabel('Date', fontsize=12)
    # ax.set_ylabel('Price (USDT)', fontsize=12)
    # ax.set_title(f'{symbol} Price History', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USDT)', fontsize=12)
    ax.set_title(f'{symbol} Price History', fontsize=14, fontweight = 'bold')
    
    # TODO 1.5: Add a grid for readability
    # ax.grid(True, alpha=0.3)
    ax.grid(True, alpha = 0.3)
    
    
    # TODO 1.6: Rotate x-axis labels for readability
    # plt.xticks(rotation=45)
    plt.xticks(rotation=45)
    
    
    # TODO 1.7: Tight layout (prevents label cutoff)
    # plt.tight_layout()
    plt.tight_layout()
    
    
    if save_fig:
        # TODO 1.8: Create results/figures directory if it doesn't exist
        figures_dir = PROJECT_ROOT / 'results' / 'figures'
        # os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        # TODO 1.9: Save the figure
        filepath = figures_dir / f'{symbol.lower()}_price_history.png'  # figures_dir / f'{symbol.lower()}_price_history.png'
        # plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        # print(f"Saved: {filepath}")
        print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════
# STEP 2: Plot Returns Distribution (Histogram)
# ═══════════════════════════════════════════════════════════════

def plot_returns_distribution(symbol, save_fig=True):
    """
    Create a histogram showing the distribution of returns
    This shows if returns are normally distributed (they usually aren't!)
    """
    # TODO 2.1: Load data and calculate returns
    df = load_price_data(symbol)  # load_price_data(symbol)
    
    if df is None:
        return
    
    df = calculate_returns(df)  # calculate_returns(df)
    
    # TODO 2.2: Remove NaN values
    returns = df['returns'].dropna()  # df['returns'].dropna()
    
    # TODO 2.3: Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # TODO 2.4: Create histogram
    # bins=50 creates 50 bars
    # alpha=0.7 makes it semi-transparent
    # Hint: ax.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

    ax.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    
    # TODO 2.5: Add a vertical line at zero (no change)
    # ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Return')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Return')
    
    # TODO 2.6: Add labels
    # ax.set_xlabel('Hourly Returns', fontsize=12)
    # ax.set_ylabel('Frequency', fontsize=12)
    # ax.set_title(f'{symbol} Returns Distribution', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Hourly Returns',  fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{symbol} Returns Distribution', fontsize=14, fontweight='bold')
    
    # TODO 2.7: Add legend
    # ax.legend()
    ax.legend()
    
    # TODO 2.8: Add grid
    # ax.grid(True, alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        figures_dir = PROJECT_ROOT / 'results' / 'figures'
        os.makedirs(figures_dir, exist_ok=True)
        filepath = figures_dir / f'{symbol.lower()}_returns_distribution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════
# STEP 3: Compare Volatility Across Assets (Bar Chart)
# ═══════════════════════════════════════════════════════════════

def plot_volatility_comparison(symbols, save_fig=True):
    """
    Create a bar chart comparing volatility across multiple assets
    """
    volatilities = []
    valid_symbols = []
    
    # TODO 3.1: Calculate volatility for each symbol
    for symbol in symbols:
        df = load_price_data(symbol)  # load_price_data(symbol)
        
        if df is None:
            continue
        
        df = calculate_returns(df)  # calculate_returns(df)
        vol = df['returns'].std()  # df['returns'].std()
        
        volatilities.append(vol)
        valid_symbols.append(symbol)
    
    # TODO 3.2: Sort by volatility (highest first)
    # Create a DataFrame to make sorting easier
    data = pd.DataFrame({
        'symbol': valid_symbols,
        'volatility': volatilities
    })
    data = data.sort_values('volatility', ascending=False)  # data.sort_values('volatility', ascending=False)
    
    # TODO 3.3: Take top 15 for readability
    data = data.head(15)  # data.head(15)
    
    # TODO 3.4: Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # TODO 3.5: Create bar chart
    # Hint: ax.bar(data['symbol'], data['volatility'], color='coral', edgecolor='black')
    ax.bar(data['symbol'], data['volatility'], color='coral', edgecolor='black')
    
    # TODO 3.6: Add labels
    # ax.set_xlabel('Asset', fontsize=12)
    # ax.set_ylabel('Volatility (Std Dev of Returns)', fontsize=12)
    # ax.set_title('Volatility Comparison - Top 15 Most Volatile', fontsize=14, fontweight='bold')

    ax.set_xlabel('Asset', fontsize=12)
    ax.set_ylabel('Volatility (Std Dev of Returns)', fontsize=12)
    ax.set_title('Volatility Comparison - Top 15 Most Volatile', fontsize = 14, fontweight='bold')
    
    # TODO 3.7: Rotate x-axis labels
    # plt.xticks(rotation=45, ha='right')
    plt.xticks(rotation = 45, ha='right' )
    
    # TODO 3.8: Add grid (only horizontal lines)
    # ax.grid(True, alpha=0.3, axis='y')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_fig:
        figures_dir = PROJECT_ROOT / 'results' / 'figures'
        os.makedirs(figures_dir, exist_ok=True)
        filepath = figures_dir / 'volatility_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════
# STEP 4: Multi-Panel Dashboard (Subplots)
# ═══════════════════════════════════════════════════════════════

def create_dashboard(symbol, save_fig=True):
    """
    Create a 2x2 dashboard showing multiple views of the data
    This is what you'd show in an interview
    """
    # TODO 4.1: Load and prepare data
    df = load_price_data(symbol)
    if df is None:
        return
    df = calculate_returns(df)
    
    # TODO 4.2: Create 2x2 subplot layout
    # Hint: fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    
    # Access individual subplots: axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    
    # ---- SUBPLOT 1: Price History (Top Left) ----
    # TODO 4.3: Plot price on axes[0, 0]
    # axes[0, 0].plot(df.index, df['close'], linewidth=2, color='blue')
    # axes[0, 0].set_title('Price History', fontweight='bold')
    # axes[0, 0].set_ylabel('Price (USDT)')
    # axes[0, 0].grid(True, alpha=0.3)
    axes[0,0].plot(df.index, df['close'], linewidth=2, color='blue')
    axes[0,0].set_title('Price History', fontweight = 'bold')
    axes[0,0].set_ylabel('Price (USDT)')
    axes[0,0].grid(True, alpha=0.3)
    
    # ---- SUBPLOT 2: Returns Over Time (Top Right) ----
    # TODO 4.4: Plot returns on axes[0, 1]
    # axes[0, 1].plot(df.index, df['returns'], linewidth=1, color='green', alpha=0.7)
    # axes[0, 1].axhline(y=0, color='red', linestyle='--')
    # axes[0, 1].set_title('Returns Over Time', fontweight='bold')
    # axes[0, 1].set_ylabel('Returns')
    # axes[0, 1].grid(True, alpha=0.3)
    axes[0,1].plot(df.index, df['returns'], linewidth=1, color='green', alpha = 0.7)
    axes[0,1].axhline(y=0, color='red', linestyle='--')
    axes[0,1].set_title('Returns Over Time', fontweight='bold')
    axes[0,1].set_ylabel('Returns')
    axes[0,1].grid(True, alpha=0.3)
    
    # ---- SUBPLOT 3: Returns Distribution (Bottom Left) ----
    # TODO 4.5: Histogram on axes[1, 0]
    returns = df['returns'].dropna()  # df['returns'].dropna()
    # axes[1, 0].hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    # axes[1, 0].axvline(x=0, color='red', linestyle='--')
    # axes[1, 0].set_title('Returns Distribution', fontweight='bold')
    # axes[1, 0].set_xlabel('Returns')
    # axes[1, 0].set_ylabel('Frequency')
    # axes[1, 0].grid(True, alpha=0.3)
    axes[1,0].hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,0].axvline(x=0, color='red', linestyle='--')
    axes[1,0].set_title('Returns Distribution', fontweight='bold')
    axes[1,0].set_xlabel('Returns')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(True, alpha=0.3)
    
    # ---- SUBPLOT 4: Volume Over Time (Bottom Right) ----
    # axes[1, 1].set_ylabel('Volume')
    # axes[1, 1].grid(True, alpha=0.3)
    axes[1,1].bar(df.index, df['volume'], color='orange', alpha = 0.6)
    axes[1,1].set_title('Trading Volume', fontweight='bold')
    axes[1,1].set_xlabel('Date')
    axes[1,1].set_ylabel('Volume')
    axes[1,1].grid(True, alpha=0.3)
    
    # TODO 4.7: Add overall title
    # fig.suptitle(f'{symbol} Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
    fig.suptitle(f'{symbol} Analysis Dashboard', fontsize=16, fontweight = 'bold', y=0.995)
    
    plt.tight_layout()
    
    if save_fig:
        figures_dir = PROJECT_ROOT / 'results' / 'figures'
        os.makedirs(figures_dir, exist_ok=True)
        filepath = figures_dir / f'{symbol.lower()}_dashboard.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════
# STEP 5: Risk-Return Scatter Plot
# ═══════════════════════════════════════════════════════════════

def plot_risk_return_scatter(summary_df, save_fig=True):
    """
    Scatter plot showing risk (volatility) vs return
    This is a classic quant visualization
    """
    # TODO 5.1: Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # TODO 5.2: Create scatter plot
    # x-axis: volatility (risk)
    # y-axis: mean_return
    # Hint: ax.scatter(summary_df['volatility'], summary_df['mean_return'], 
    #                  s=100, alpha=0.6, c='purple', edgecolors='black')
    ax.scatter(summary_df['volatility'], summary_df['mean_return'], s = 100, alpha =0.6, c='purple', edgecolors = 'black')
    
    # TODO 5.3: Add labels for each point
    # This shows which asset is which
    # for idx, row in summary_df.iterrows():
    #     ax.annotate(row['symbol'], 
    #                 (row['volatility'], row['mean_return']),
    #                 fontsize=8, alpha=0.7)
    for idx, row in summary_df.iterrows():
        ax.annotate(row['symbol'],
                    (row['volatility'], row['mean_return']),
                    fontsize = 8, alpha = 0.7)
    
    # TODO 5.4: Add reference lines
    # Vertical line at median volatility
    # Horizontal line at zero return
    # ax.axvline(x=summary_df['volatility'].median(), color='gray', linestyle='--', alpha=0.5)
    # ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=summary_df['volatility'].median(), color = 'gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='red', linestyle = '--', alpha = 0.5 )
    
    # TODO 5.5: Add labels
    # ax.set_xlabel('Volatility (Risk)', fontsize=12)
    # ax.set_ylabel('Mean Return', fontsize=12)
    # ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax.set_xlabel('Volatility (Risk)', fontsize = 12)
    ax.set_ylabel('Mean_Return', fontsize = 12)
    ax.set_title('Risk-REturn Profile', fontsize=14, fontweight='bold')
    
    # TODO 5.6: Add grid
    # ax.grid(True, alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        figures_dir = PROJECT_ROOT / 'results' / 'figures'
        os.makedirs(figures_dir, exist_ok=True)
        filepath = figures_dir / 'risk_return_scatter.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Creating visualizations...")
    
    # TODO: Test individual plots first
    test_symbol = 'BTCUSDT'
    
    print(f"\n1. Plotting price history for {test_symbol}...")
    # plot_price_history(test_symbol)
    plot_price_history(test_symbol)
    print(f"\n2. Plotting returns distribution for {test_symbol}...")
    # plot_returns_distribution(test_symbol)
    plot_returns_distribution(test_symbol)
    print(f"\n3. Creating dashboard for {test_symbol}...")
    # create_dashboard(test_symbol)
    create_dashboard(test_symbol)
    # TODO: Load your summary from Day 3
    # summary_df = pd.read_csv(PROJECT_ROOT / 'results' / 'tables' / 'crypto_analysis_summary.csv')
    summary_df = pd.read_csv(PROJECT_ROOT / 'results' / 'tables' / 'crypto_analysis_summary.csv')

    print("\n4. Plotting risk-return scatter...")
    # plot_risk_return_scatter(summary_df)
    plot_risk_return_scatter(summary_df)

    print("\n5. Comparing volatility across assets...")
    # symbols = summary_df['symbol'].tolist()[:20]  # Top 20
    symbols = summary_df['symbol'].tolist()[:20]
    # plot_volatility_comparison(symbols)
    plot_volatility_comparison(symbols)
    
    print("\n✅ Day 4 Complete!")
    print(f"Check your results/figures/ folder for all charts")


# ═══════════════════════════════════════════════════════════════
# WHAT TO SUBMIT:
# ═══════════════════════════════════════════════════════════════
# 1. Your completed code
# 2. Screenshots of at least 3 different charts
# 3. The dashboard for BTC or ETH
# 4. The risk-return scatter plot
#
# BONUS CHALLENGES:
# - Add color-coding to scatter plot (green = positive return, red = negative)
# - Create a correlation heatmap (see Day 5 preview)
# - Add moving averages to price chart
# - Make the charts look even more professional with custom color schemes
# ═══════════════════════════════════════════════════════════════