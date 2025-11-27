"""
DAY 5: CORRELATION ANALYSIS
Goal: Find which crypto pairs move together (candidates for stat arb)
Time: 1-2 hours

CONCEPTS:
- Correlation measures linear relationship between two variables
- High correlation (>0.7) = assets move together
- These become candidates for cointegration testing (Week 7)

WHY THIS MATTERS:
- Pairs trading requires assets that move together
- When correlated assets diverge, we bet on convergence
- This is the foundation of statistical arbitrage
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.features.basic_analysis import load_price_data, calculate_returns


# ═══════════════════════════════════════════════════════════════
# STEP 1: Build Returns DataFrame (All Assets)
# ═══════════════════════════════════════════════════════════════

def build_returns_matrix(symbols):
    """
    Create a DataFrame where each column is returns for one asset
    All assets aligned by timestamp
    
    Parameters:
        symbols (list): List of trading pairs
    
    Returns:
        pd.DataFrame: Returns matrix (rows=timestamps, cols=assets)
    """
    returns_dict = {}
    
    for symbol in symbols:
        # TODO 1.1: Load price data
        df = load_price_data(symbol)  # load_price_data(symbol)
        
        if df is None:
            continue
        
        # TODO 1.2: Calculate returns
        df = calculate_returns(df)  # calculate_returns(df)
        
        # TODO 1.3: Extract returns series and rename to symbol
        # We want just the 'returns' column, named by the symbol
        # Hint: df['returns'].rename(symbol)
        returns_series = df['returns'].rename(symbol)
        
        # TODO 1.4: Add to dictionary
        # returns_dict[symbol] = returns_series
        returns_dict[symbol] = returns_series
    
    # TODO 1.5: Combine all series into one DataFrame
    # pd.DataFrame(returns_dict) aligns by index automatically
    returns_matrix = pd.DataFrame(returns_dict)  # pd.DataFrame(returns_dict)
    
    # TODO 1.6: Drop rows with NaN (first row has NaN from pct_change)
    returns_matrix = returns_matrix.dropna()  # returns_matrix.dropna()
    
    print(f"Built returns matrix: {returns_matrix.shape[0]} timestamps x {returns_matrix.shape[1]} assets")
    
    return returns_matrix


# ═══════════════════════════════════════════════════════════════
# STEP 2: Calculate Correlation Matrix
# ═══════════════════════════════════════════════════════════════

def calculate_correlation_matrix(returns_matrix):
    """
    Calculate pairwise correlations between all assets
    
    Parameters:
        returns_matrix (pd.DataFrame): Returns for all assets
    
    Returns:
        pd.DataFrame: Correlation matrix (N x N)
    """
    # TODO 2.1: Calculate correlation matrix
    # Pandas makes this easy: df.corr()
    # This calculates Pearson correlation between all column pairs
    corr_matrix = returns_matrix.corr() # returns_matrix.corr()
    
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    
    return corr_matrix


# ═══════════════════════════════════════════════════════════════
# STEP 3: Visualize Correlation Heatmap
# ═══════════════════════════════════════════════════════════════

def plot_correlation_heatmap(corr_matrix, save_fig=True):
    """
    Create a heatmap showing correlations between all assets
    Red = high positive correlation, Blue = low/negative correlation
    """
    # TODO 3.1: Create figure (make it big enough to read)
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # TODO 3.2: Create heatmap using seaborn
    # sns.heatmap() creates beautiful heatmaps
    # Parameters:
    #   - annot=False (too many cells for annotations)
    #   - cmap='RdYlBu_r' (red=high, blue=low)
    #   - center=0 (center colormap at zero)
    #   - vmin=-1, vmax=1 (correlation range)
    
    # Hint:
    # sns.heatmap(corr_matrix, annot=False, cmap='RdYlBu_r', 
    #             center=0, vmin=-1, vmax=1, ax=ax,
    #             square=True, linewidths=0.5)
    sns.heatmap(corr_matrix, annot = False, cmap = 'RdYlBu_r', center = 0, vmin = -1, vmax = 1, ax = ax, square = True, linewidths = 0.5 )
    
    # TODO 3.3: Add title
    # ax.set_title('Cryptocurrency Correlation Matrix', fontsize=16, fontweight='bold')
    ax.set_title('Cryptocurrency Correlation Matrix', fontsize = 16, fontweight = 'bold')
    
    # TODO 3.4: Rotate labels for readability
    # plt.xticks(rotation=90)
    # plt.yticks(rotation=0)
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    
    plt.tight_layout()
    
    if save_fig:
        figures_dir = PROJECT_ROOT / 'results' / 'figures'
        os.makedirs(figures_dir, exist_ok=True)
        filepath = figures_dir / 'correlation_heatmap.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════
# STEP 4: Find Highly Correlated Pairs
# ═══════════════════════════════════════════════════════════════

def find_top_correlated_pairs(corr_matrix, min_correlation=0.7, max_pairs=30):
    """
    Extract the most correlated pairs from the correlation matrix
    These are candidates for pairs trading
    
    Parameters:
        corr_matrix (pd.DataFrame): Correlation matrix
        min_correlation (float): Minimum correlation to include
        max_pairs (int): Maximum number of pairs to return
    
    Returns:
        pd.DataFrame: Top correlated pairs with their correlations
    """
    pairs = []
    
    # TODO 4.1: Get list of all symbols
    symbols = corr_matrix.columns.tolist()
    
    # TODO 4.2: Loop through upper triangle of matrix (avoid duplicates)
    # We only want each pair once (BTC-ETH, not also ETH-BTC)
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):  # j > i ensures upper triangle
            # TODO 4.3: Get correlation for this pair
            corr = corr_matrix.iloc[i, j]  # corr_matrix.iloc[i, j]
            
            # TODO 4.4: Skip if below threshold or if it's a stablecoin pair
            # Stablecoins (USDC, USDT) have weird correlations - skip them
            symbol_a = symbols[i]
            symbol_b = symbols[j]
            
            # Skip stablecoin pairs
            stablecoins = ['USDCUSDT', 'USDTUSDT', 'BUSDUSDT']
            if symbol_a in stablecoins or symbol_b in stablecoins:
                continue
            
            # Skip if correlation too low
            # if corr < min_correlation:
            #     continue
            if corr < min_correlation:
                continue

            # TODO 4.5: Add to pairs list
            # pairs.append({
            #     'asset_a': symbol_a,
            #     'asset_b': symbol_b,
            #     'correlation': corr
            # })
            pairs.append({
                'asset_a': symbol_a,
                'asset_b': symbol_b,
                'correlation': corr
            })
    
    # TODO 4.6: Convert to DataFrame and sort by correlation
    pairs_df = pd.DataFrame(pairs)  # pd.DataFrame(pairs)
    
    if pairs_df is None or len(pairs_df) == 0:
        print("No pairs found above correlation threshold")
        return pd.DataFrame()
    
    pairs_df = pairs_df.sort_values('correlation', ascending = False)  # pairs_df.sort_values('correlation', ascending=False)
    
    # TODO 4.7: Take top N pairs
    pairs_df = pairs_df.head(max_pairs)  # pairs_df.head(max_pairs)
    
    return pairs_df


# ═══════════════════════════════════════════════════════════════
# STEP 5: Visualize Top Pairs
# ═══════════════════════════════════════════════════════════════

def plot_top_pairs(pairs_df, save_fig=True):
    """
    Bar chart showing top correlated pairs
    """
    if pairs_df is None or len(pairs_df) == 0:
        print("No pairs to plot")
        return
    
    # TODO 5.1: Create pair labels (e.g., "BTC-ETH")
    # Hint: pairs_df['asset_a'] + '-' + pairs_df['asset_b']
    # But cleaner: remove 'USDT' suffix first
    pairs_df = pairs_df.copy()
      # Create labels like "BTC-ETH"
    
    # Hint to create clean labels:
    # pairs_df['pair_label'] = (pairs_df['asset_a'].str.replace('USDT', '') + 
    #                          '-' + 
    #                          pairs_df['asset_b'].str.replace('USDT', ''))
    pairs_df['pair_label'] = (pairs_df['asset_a'].str.replace('USDT', '') +
                              '-' +
                              pairs_df['asset_b'].str.replace('USDT', ''))

    # TODO 5.2: Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # TODO 5.3: Create horizontal bar chart (easier to read labels)
    # Hint: ax.barh(pairs_df['pair_label'], pairs_df['correlation'], color='steelblue')
    
    ax.barh(pairs_df['pair_label'], pairs_df['correlation'], color = 'steelblue')
    
    # TODO 5.4: Add labels and title
    # ax.set_xlabel('Correlation', fontsize=12)
    # ax.set_ylabel('Pair', fontsize=12)
    # ax.set_title('Top Correlated Cryptocurrency Pairs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Correlation', fontsize = 12)
    ax.set_ylabel('Pair', fontsize = 12)
    ax.set_title('Top Correlated Cryptocurrency Pairs', fontsize = 14, fontweight = 'bold')
    
    # TODO 5.5: Add vertical line at correlation threshold
    # ax.axvline(x=0.7, color='red', linestyle='--', label='Threshold (0.7)')
    ax.axvline(x=0.7, color = 'red', linestyle = '--', label = 'Threshold (0.7)')
    
    # TODO 5.6: Add grid and legend
    # ax.grid(True, alpha=0.3, axis='x')
    # ax.legend()
    ax.grid(True, alpha = 0.3, axis = 'x' )
    ax.legend()
    
    
    plt.tight_layout()
    
    if save_fig:
        figures_dir = PROJECT_ROOT / 'results' / 'figures'
        os.makedirs(figures_dir, exist_ok=True)
        filepath = figures_dir / 'top_correlated_pairs.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════
# STEP 6: Plot Pair Price Comparison
# ═══════════════════════════════════════════════════════════════

def plot_pair_comparison(symbol_a, symbol_b, save_fig=True):
    """
    Plot two assets on the same chart to visually confirm correlation
    Uses normalized prices (start at 100) for comparison
    """
    # TODO 6.1: Load both datasets
    df_a = load_price_data(symbol_a)
    df_b = load_price_data(symbol_b)
    
    if df_a is None or df_b is None:
        print(f"Could not load data for {symbol_a} or {symbol_b}")
        return
    
    # TODO 6.2: Normalize prices (start at 100)
    # This lets us compare assets with different price scales
    # Formula: normalized = (price / first_price) * 100
    norm_a = df_a['close'] / df_a['close'].iloc[0] * 100  # (df_a['close'] / df_a['close'].iloc[0]) * 100
    norm_b = df_b['close'] / df_b['close'].iloc[0] * 100  # (df_b['close'] / df_b['close'].iloc[0]) * 100
    
    # TODO 6.3: Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # TODO 6.4: Plot both normalized prices
    # ax.plot(df_a.index, norm_a, label=symbol_a.replace('USDT', ''), linewidth=2)
    # ax.plot(df_b.index, norm_b, label=symbol_b.replace('USDT', ''), linewidth=2)
    ax.plot(df_a.index, norm_a, label = symbol_a.replace('USDT', ''), linewidth = 2)
    ax.plot(df_b.index, norm_b, label = symbol_b.replace('USDT', ''), linewidth = 2)
    
    # TODO 6.5: Add labels and title
    # ax.set_xlabel('Date', fontsize=12)
    # ax.set_ylabel('Normalized Price (Start=100)', fontsize=12)
    # ax.set_title(f'Price Comparison: {symbol_a} vs {symbol_b}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize = 12)
    ax.set_ylabel('Normalized Price (Start = 100)', fontsize = 12)
    ax.set_title(f'Price Comparison: {symbol_a} vs. {symbol_b}', fontsize = 14, fontweight = 'bold')
    
    # TODO 6.6: Add legend and grid
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    ax.legend()
    ax.grid(True, alpha = 0.3 )
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_fig:
        figures_dir = PROJECT_ROOT / 'results' / 'figures'
        os.makedirs(figures_dir, exist_ok=True)
        # Clean filename
        name_a = symbol_a.replace('USDT', '').lower()
        name_b = symbol_b.replace('USDT', '').lower()
        filepath = figures_dir / f'pair_comparison_{name_a}_{name_b}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════
# STEP 7: Save Results
# ═══════════════════════════════════════════════════════════════

def save_correlation_results(corr_matrix, pairs_df):
    """
    Save correlation matrix and top pairs to CSV
    """
    tables_dir = PROJECT_ROOT / 'results' / 'tables'
    os.makedirs(tables_dir, exist_ok=True)
    
    # Save correlation matrix
    corr_path = tables_dir / 'correlation_matrix.csv'
    corr_matrix.to_csv(corr_path)
    print(f"Saved: {corr_path}")
    
    # Save top pairs
    if pairs_df is not None and len(pairs_df) > 0:
        pairs_path = tables_dir / 'top_correlated_pairs.csv'
        pairs_df.to_csv(pairs_path, index=False)
        print(f"Saved: {pairs_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("DAY 5: CORRELATION ANALYSIS")
    print("="*60)
    
    # Load symbol list from your Day 3 analysis
    summary_path = PROJECT_ROOT / 'results' / 'tables' / 'crypto_analysis_summary.csv'
    summary_df = pd.read_csv(summary_path)
    symbols = summary_df['symbol'].tolist()
    
    print(f"\nAnalyzing {len(symbols)} assets...")
    
    # Step 1: Build returns matrix
    print("\n1. Building returns matrix...")
    # returns_matrix = build_returns_matrix(symbols)
    
    returns_matrix = build_returns_matrix(symbols)

    # Step 2: Calculate correlations
    print("\n2. Calculating correlation matrix...")
    # corr_matrix = calculate_correlation_matrix(returns_matrix)
    
    corr_matrix = calculate_correlation_matrix(returns_matrix)
    # Step 3: Plot heatmap
    print("\n3. Creating correlation heatmap...")
    # plot_correlation_heatmap(corr_matrix)
    
    plot_correlation_heatmap(corr_matrix)

    # Step 4: Find top pairs
    print("\n4. Finding top correlated pairs...")
    # pairs_df = find_top_correlated_pairs(corr_matrix, min_correlation=0.7)
    # print(f"\nTop 10 correlated pairs:")
    # print(pairs_df.head(10))

    pairs_df = find_top_correlated_pairs(corr_matrix, min_correlation = 0.7)
    print(f'\n Top 10 correlated pairs:')
    print(pairs_df.head(10))
    
    # Step 5: Plot top pairs
    print("\n5. Plotting top pairs...")
    # plot_top_pairs(pairs_df)
    
    plot_top_pairs(pairs_df)

    # Step 6: Plot comparison for top pair
    print("\n6. Plotting pair comparison...")
    # if len(pairs_df) > 0:
    #     top_pair = pairs_df.iloc[0]
    #     plot_pair_comparison(top_pair['asset_a'], top_pair['asset_b'])

    if len(pairs_df) > 0:
        top_pair = pairs_df.iloc[0]
        plot_pair_comparison(top_pair['asset_a'], top_pair['asset_b'])

    
    # Step 7: Save results
    print("\n7. Saving results...")
    # save_correlation_results(corr_matrix, pairs_df)

    save_correlation_results(corr_matrix, pairs_df)
    
    print("\n" + "="*60)
    print("✅ Day 5 Complete!")
    print("="*60)
    print("\nKey outputs:")
    print("- results/figures/correlation_heatmap.png")
    print("- results/figures/top_correlated_pairs.png")
    print("- results/tables/correlation_matrix.csv")
    print("- results/tables/top_correlated_pairs.csv")


# ═══════════════════════════════════════════════════════════════
# WHAT TO SUBMIT:
# ═══════════════════════════════════════════════════════════════
# 1. Your completed code
# 2. Screenshots:
#    - Correlation heatmap
#    - Top correlated pairs bar chart
#    - At least one pair comparison plot
# 3. List of your top 5 most correlated pairs
#
# QUESTIONS TO THINK ABOUT:
# - Which assets are most correlated? Does this make sense?
# - Are there any surprising correlations (or lack thereof)?
# - Why do you think some pairs are highly correlated?
# ═══════════════════════════════════════════════════════════════