"""
DAY 7: COINTEGRATION REFINEMENT (Concept-Driven)
Goal: Understand the stability and robustness of cointegration
Time: 2-3 hours

CONCEPTS:
- Parameter sensitivity (does window size matter?)
- Rolling cointegration (is it stable over time?)
- Statistical robustness (are results reliable?)
- Pair quality metrics (what makes a "good" pair?)

LEARNING OBJECTIVES:
- Understand how lookback window affects results
- Test if cointegration is stable or breaks down
- Build intuition for parameter selection
- Create a systematic pair ranking system
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import os

from src.features.basic_analysis import load_price_data
from src.features.cointegration import (
    calculate_hedge_ratio, 
    calculate_spread, 
    adf_test, 
    calculate_half_life
)


# ═══════════════════════════════════════════════════════════════
# CONCEPT 1: PARAMETER SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════
"""
QUESTION: Does cointegration depend on the lookback window?

INTUITION:
- Short window (500h): Captures recent behavior, but noisy
- Medium window (1000h): Balance of recency and stability
- Long window (2000h): More stable, but includes old regimes

YOUR TASK:
Write a function that tests a pair across multiple lookback windows
and sees if cointegration is consistent.

INPUTS:
- symbol_a, symbol_b: The pair to test
- windows: List of lookback periods to test (e.g., [500, 1000, 1500, 2000])

OUTPUTS:
- DataFrame with columns: window, p_value, half_life, is_cointegrated

ALGORITHM:
1. Load full price data for both assets
2. For each window size:
   a. Take last N hours of data
   b. Calculate hedge ratio on that window
   c. Calculate spread
   d. Run ADF test
   e. Calculate half-life
   f. Store results
3. Return DataFrame showing how results change with window

PANDAS HINT: To get last N rows: df.tail(N) or df.iloc[-N:]
"""

def test_parameter_sensitivity(symbol_a, symbol_b, windows=[500, 1000, 1500, 2000]):
    """
    Test how cointegration results change with different lookback windows
    
    Parameters:
        symbol_a (str): First asset
        symbol_b (str): Second asset
        windows (list): List of lookback periods to test
    
    Returns:
        pd.DataFrame: Results for each window size
    """
    # TODO: YOUR CODE HERE
    # Load full data
    df_a = load_price_data(symbol_a)
    df_b = load_price_data(symbol_b)

    if df_a is None or df_b is None:
        return None
    
    prices_a = df_a['close']
    prices_b = df_b['close']

    results = []

    for window in windows:
        print(f"Testing window size: {window} hours")
        
        prices_a_window = prices_a.tail(window)
        prices_b_window = prices_b.tail(window)

        hedge_ratio = calculate_hedge_ratio(prices_a_window, prices_b_window)
        spread = calculate_spread(prices_a_window, prices_b_window, hedge_ratio)
        adf_results = adf_test(spread)
        half_life = calculate_half_life(spread)


        results.append({
            'window': window,
            'p_value': adf_results['p_value'],
            'half_life': half_life,
            'is_cointegrated': adf_results['is_stationary'] and half_life < 100
        })

    # Loop through windows
    # For each window: calculate stats
    # Return DataFrame with results
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# CONCEPT 2: ROLLING COINTEGRATION TEST
# ═══════════════════════════════════════════════════════════════
"""
QUESTION: Is cointegration stable over time, or does it break down?

INTUITION:
If cointegration is real, it should be consistent across different time periods.
If it's spurious (random), p-values will jump around.

EXAMPLE:
Month 1: p=0.02 ✅ Cointegrated
Month 2: p=0.01 ✅ Still cointegrated
Month 3: p=0.45 ❌ Broke down (structural break!)

YOUR TASK:
Write a function that slides a rolling window through time
and calculates p-value for each window.

INPUTS:
- symbol_a, symbol_b: The pair
- window_size: Size of each window (e.g., 500 hours)
- step_size: How much to move window each time (e.g., 100 hours)

OUTPUTS:
- DataFrame with columns: window_start, window_end, p_value, half_life

ALGORITHM:
1. Load full price data
2. Start at beginning
3. While window fits in data:
   a. Extract data for current window
   b. Calculate hedge ratio, spread, ADF test
   c. Store results with window dates
   d. Move window forward by step_size
4. Return DataFrame

VISUALIZATION IDEA: Plot p-value over time to see stability

PANDAS HINT: Use df.iloc[start:end] to slice windows
"""

def rolling_cointegration_test(symbol_a, symbol_b, window_size=500, step_size=100):
    """
    Test cointegration on rolling windows through time
    
    Parameters:
        symbol_a (str): First asset
        symbol_b (str): Second asset
        window_size (int): Size of each window in hours
        step_size (int): Step size to move window
    
    Returns:
        pd.DataFrame: Results for each time window
    """
    # TODO: YOUR CODE HERE
    # Load full data
    # Initialize results list
    # Loop through windows (start from 0, step by step_size)
    # For each window: calculate cointegration stats
    # Return DataFrame with time series of results
    df_a = load_price_data(symbol_a)
    df_b = load_price_data(symbol_b)

    if df_a is None or df_b is None:
        return None
    
    prices_a = df_a['close']
    prices_b = df_b['close']

    total_length = len(prices_a)
    results = []

    start = 0

    while start + window_size <= total_length:

        prices_a_window = prices_a.iloc[start : start + window_size]
        prices_b_window = prices_b.iloc[start : start + window_size]

        window_start = df_a.index[start]
        window_end = df_a.index[start + window_size - 1]

        print(f"Testing window: {window_start} to {window_end}")

        hedge_ratio = calculate_hedge_ratio(prices_a_window, prices_b_window)
        spread = calculate_spread(prices_a_window, prices_b_window, hedge_ratio)
        adf_results = adf_test(spread)
        half_life = calculate_half_life(spread)

        results.append({
            'window_start': window_start,
            'window_end': window_end,
            'p_value': adf_results['p_value'],
            'half_life': half_life,
            'is_cointegrated': adf_results['is_stationary'] and half_life < 100
        })

        start += step_size

    return pd.DataFrame(results)
# ═══════════════════════════════════════════════════════════════
# CONCEPT 3: VISUALIZE PARAMETER SENSITIVITY
# ═══════════════════════════════════════════════════════════════
"""
GOAL: Create a plot showing how p-value and half-life change with window size

INPUTS: Results DataFrame from test_parameter_sensitivity()

OUTPUTS: 2-panel plot
- Top panel: P-value vs window size (with 0.05 threshold line)
- Bottom panel: Half-life vs window size (with 100h threshold line)

INSIGHT: If lines are flat → robust cointegration
         If lines jump around → unstable, be careful
"""

def plot_parameter_sensitivity(results_df, pair_name, save_fig=True):
    """
    Visualize how cointegration metrics change with window size
    
    Parameters:
        results_df (pd.DataFrame): Output from test_parameter_sensitivity()
        pair_name (str): Name of pair for title (e.g., "OP-PEPE")
        save_fig (bool): Whether to save figure
    """
    # TODO: YOUR CODE HERE
    # Create 2-panel figure
    # Top: Plot p_value vs window
    # Bottom: Plot half_life vs window
    # Add threshold lines (0.05 for p-value, 100 for half-life)
    # Save figure
    fig, axes = plt.subplots(2, 1, figsize=(12,8))
    axes[0].plot(results_df['window'], results_df['p_value'], marker='o', linewidth = 2)
    axes[0].axhline(y=0.05, color='red', linestyle='--', label='Threshold')
    axes[0].set_xlabel('Window Size (hours)')
    axes[0].set_ylabel('P_value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results_df['window'], results_df['half_life'], marker='s', linewidth=2)
    axes[1].axhline(y=100, color='red', linestyle='--', label='Max tradeable')
    axes[1].set_xlabel('Window Size (hours)')
    axes[1].set_ylabel('Half-Life')
    axes[1].legend()
    fig.suptitle(f'Parameter Sensititivy: {pair_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()


    if save_fig:
        figures_dir = PROJECT_ROOT / 'results' / 'figures'
        os.makedirs(figures_dir, exist_ok=True)
        filepath = figures_dir / f'sensitivity_{pair_name}.png'
        plt.savefig(filepath, dpi=300)

        plt.show()
        plt.close()


# ═══════════════════════════════════════════════════════════════
# CONCEPT 4: VISUALIZE ROLLING COINTEGRATION
# ═══════════════════════════════════════════════════════════════
"""
GOAL: Plot how cointegration changes over time

INPUTS: Results DataFrame from rolling_cointegration_test()

OUTPUTS: 2-panel plot showing time series of:
- Top: P-value over time (shaded green when < 0.05)
- Bottom: Half-life over time

INSIGHT: If cointegration is stable, p-value stays below 0.05
         If it breaks, you'll see spikes above 0.05
"""

def plot_rolling_cointegration(results_df, pair_name, save_fig=True):
    """
    Visualize cointegration stability over time
    
    Parameters:
        results_df (pd.DataFrame): Output from rolling_cointegration_test()
        pair_name (str): Name of pair for title
        save_fig (bool): Whether to save
    """
    # TODO: YOUR CODE HERE
    # Create 2-panel time series plot
    # Top: P-value over time with 0.05 threshold
    # Bottom: Half-life over time
    # Shade periods where cointegrated (p < 0.05)
    fig, axes = plt.subplots(2, 1, figsize=(14,8))
    axes[0].plot(results_df['window_end'],
                 results_df['p_value'],
                 linewidth=2,
                 color='purple',
                 label='P_value')
    
    axes[0].axhline(y=0.05,
                    color='red',
                    linestyle='--',
                    label='Threshold (0.05)')
    
    axes[0].set_ylabel('P_value', fontsize = 12)
    axes[0].set_title('P_value Over Time', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)


    axes[1].plot(results_df['window_end'],
                 results_df['half_life'],
                 linewidth=2,
                 color='orange',
                 label='Half-Life')
    
    axes[1].axhline(y=100,
                    color='red',
                    linestyle='--',
                    label='Max tradeable (100h)')
    
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Half-life (hours)', fontsize=12)
    axes[1].set_title('Half-life over Time', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.xticks(rotation=45)

    fig.suptitle(f'Rolling Cointegration {pair_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_fig:
        figures_dir = PROJECT_ROOT / 'results' / 'figures'
        os.makedirs(figures_dir, exist_ok=True)
        filepath = figures_dir / f'rolling_{pair_name.lower().replace("-", "_")}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.show()
    plt.close()

    is_coint = results_df['p_value'] < 0.05
    axes[0].fill_between(results_df['window_end'],
                         0, 0.05,
                         where=is_coint,
                         alpha=0.3,
                         color='green',
                         label='Cointegrated')
    
    is_tradeable = results_df['half_life'] < 100
    axes[1].fill_between(results_df['window_end'], 0, 100, 
                         where=is_tradeable,
                         alpha=0.3,
                         color='green')


# ═══════════════════════════════════════════════════════════════
# CONCEPT 5: PAIR QUALITY SCORE
# ═══════════════════════════════════════════════════════════════
"""
GOAL: Create a single "quality score" for each pair

FACTORS TO CONSIDER:
1. P-value (lower is better)
2. Half-life (faster is better, ~10-50h ideal)
3. Consistency across windows
4. Stability over time

SCORING SYSTEM (example):
- P-value score: (0.05 - p_value) / 0.05 * 40 points (max 40)
- Half-life score: Based on ideal range (max 30)
  - 10-50h: 30 points
  - <10h: penalize (too fast, might be noise)
  - >50h: penalize (too slow)
- Consistency: Std dev of p-values across windows (max 30)

YOUR TASK:
Design and implement a scoring function that takes cointegration stats
and outputs a quality score (0-100).
"""

def calculate_pair_quality_score(p_value, half_life, p_value_std=None):
    """
    Calculate quality score for a cointegrated pair
    
    Parameters:
        p_value (float): ADF test p-value
        half_life (float): Mean reversion half-life
        p_value_std (float, optional): Std dev of p-values across windows
    
    Returns:
        float: Quality score (0-100)
    """
    # TODO: YOUR CODE HERE
    # Design your scoring system
    # Consider: p-value, half-life, consistency
    # Return score 0-100
    score = 0

    if p_value < 0.05:
        p_score = (0.05 - p_value) / 0.05 * 40
    else:
        p_score = 0

    if 10 <= half_life <= 50:
        hl_score = 30
    elif half_life < 10:
        hl_score = 20
    else:
        hl_score = max(0, 30 - (half_life - 50) / 5)

    if p_value_std is not None:
        consistency_score = 30 * (1 - min(p_value_std, 1))
    else:
        consistency_score = 30

    score = p_score + hl_score + consistency_score

    return score



# ═══════════════════════════════════════════════════════════════
# CONCEPT 6: COMPARE MULTIPLE PAIRS
# ═══════════════════════════════════════════════════════════════
"""
GOAL: Rank all your cointegrated pairs by quality

INPUTS: List of cointegrated pairs from Day 6

OUTPUTS: DataFrame with quality scores, ranked

ALGORITHM:
1. Load cointegrated pairs CSV
2. For each pair:
   a. Run parameter sensitivity test
   b. Calculate average p-value, half-life
   c. Calculate consistency (std dev of p-values)
   d. Calculate quality score
3. Rank by quality score
4. Return top N pairs
"""

def rank_pairs_by_quality():
    """
    Rank all cointegrated pairs by quality score
    
    Returns:
        pd.DataFrame: Ranked pairs with quality scores
    """
    # TODO: YOUR CODE HERE
    # Load cointegrated_pairs.csv
    # For each pair, run tests and calculate score
    # Sort by score
    # Return ranked DataFrame
    tables_dir = PROJECT_ROOT / 'results' / 'tables'
    coint_pairs = pd.read_csv(tables_dir / 'cointegrated_pairs.csv')

    ranked_pairs = []

    for idx, row in coint_pairs.iterrows():
        symbol_a = row['asset_a']
        symbol_b = row['asset_b']

        sensitivity_results = test_parameter_sensitivity(symbol_a, symbol_b)

    avg_p_value = sensitivity_results['p_value'].mean()
    p_value_std = sensitivity_results['p_value'].std()

    avg_half_life = sensitivity_results['half_life'].mean()

    quality = calculate_pair_quality_score(avg_p_value, avg_half_life, p_value_std)

    ranked_pairs.append({
        'asset_a': symbol_a,
        'asset b': symbol_b,
        'avg_p_value': avg_p_value,
        'p_value_std': p_value_std,
        'avg_half_life': avg_half_life,
        'quality_score': quality

    })

    results_df = pd.DataFrame(ranked_pairs)
    
    results_df = results_df.sort_values('quality_score', ascending=False)

    return results_df

# ═══════════════════════════════════════════════════════════════
# CONCEPT 7: OPTIMAL LOOKBACK WINDOW
# ═══════════════════════════════════════════════════════════════
"""
QUESTION: What's the optimal lookback window for cointegration testing?

APPROACH:
Test multiple window sizes and see which gives most stable results.

METRIC: "Stability score" = 
  - High if cointegration is consistent across sub-windows
  - Low if results are noisy/inconsistent

YOUR TASK:
Write a function that recommends optimal window size for a pair.
"""

def find_optimal_window(symbol_a, symbol_b, test_windows=[500, 750, 1000, 1500, 2000]):
    """
    Find the optimal lookback window for a pair
    
    Parameters:
        symbol_a, symbol_b: The pair
        test_windows: Windows to test
    
    Returns:
        int: Recommended window size
    """
    # TODO: YOUR CODE HERE
    # Test each window
    # Calculate consistency/stability metric
    # Return window with best stability
    results = test_parameter_sensitivity(symbol_a, symbol_b, test_windows)
    p_value_var = results['p_value'].var()
    half_life_var = results['half_life'].var()
    stability = 1 / (1 + p_value_var + half_life_var * 0.01)
    best_window = results.loc[results['p_value'].idxmin(), 'window']
    return {
               'optimal_window': best_window,
              'stability_score': stability,
               'p_value_variance': p_value_var,
               'half_life_variance': half_life_var,
               'details': results
           }


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION & TESTING
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("DAY 7: COINTEGRATION REFINEMENT")
    print("="*60)
    
    # Load your cointegrated pairs from Day 6
    tables_dir = PROJECT_ROOT / 'results' / 'tables'
    coint_pairs = pd.read_csv(tables_dir / 'cointegrated_pairs.csv')
    
    print(f"\nYou have {len(coint_pairs)} cointegrated pair(s):")
    for _, row in coint_pairs.iterrows():
        print(f"  - {row['asset_a']} - {row['asset_b']}")
    
    # TODO: Test your functions
    # Example workflow:
    symbol_a = coint_pairs.iloc[0]['asset_a']
    symbol_b = coint_pairs.iloc[0]['asset_b']
    pair_name = f"{symbol_a.replace('USDT', '')} - {symbol_b.replace('USDT', '')}"

    print(f"\nAnalyzing: {pair_name}")
    print("=" * 60)

    print("\n1. Testing parameter sensitivity...")
    sensitivity_results = test_parameter_sensitivity(symbol_a, symbol_b)
    print(sensitivity_results)

    print("\n2. Plotting parameter sensitivity...")
    plot_parameter_sensitivity(sensitivity_results, pair_name)

    print("\n3. Testing roling cointegreation (this may take a minute)...")
    rolling_results = rolling_cointegration_test(symbol_a, symbol_b, window_size = 500, step_size = 100)
    print(f"Tested {len(rolling_results)} rolling windows")
    print(rolling_results.head())

    print("\n4. Plotting rolling cointegration...")
    plot_rolling_cointegration(rolling_results, pair_name)

    print("\n5. Calculating quality score...")
    p_val = coint_pairs.iloc[0]['p_value']
    hl = coint_pairs.iloc[0]['half_life']
    quality = calculate_pair_quality_score(p_val, hl)
    print(f"Quality Score: {quality:.1f}/100")


    print("\n6. Finding optimal window...")
    optimal = find_optimal_window(symbol_a, symbol_b)
    print(f"Optimal window: {optimal['optimal_window']} hours")
    print(f"Stability score: {optimal['stability_score']:.4f}")


    # 1. Test parameter sensitivity for your best pair
    # symbol_a = coint_pairs.iloc[0]['asset_a']
    # symbol_b = coint_pairs.iloc[0]['asset_b']
    # sensitivity_results = test_parameter_sensitivity(symbol_a, symbol_b)
    # print("\nParameter Sensitivity Results:")
    # print(sensitivity_results)
    
    # 2. Plot parameter sensitivity
    # plot_parameter_sensitivity(sensitivity_results, f"{symbol_a}-{symbol_b}")
    
    # 3. Test rolling cointegration
    # rolling_results = rolling_cointegration_test(symbol_a, symbol_b)
    # print("\nRolling Cointegration Results:")
    # print(rolling_results.head())
    
    # 4. Plot rolling cointegration
    # plot_rolling_cointegration(rolling_results, f"{symbol_a}-{symbol_b}")
    
    # 5. Calculate quality scores
    # quality_score = calculate_pair_quality_score(
    #     p_value=coint_pairs.iloc[0]['p_value'],
    #     half_life=coint_pairs.iloc[0]['half_life']
    # )
    # print(f"\nQuality Score: {quality_score:.1f}/100")
    
   


# ═══════════════════════════════════════════════════════════════
# EXPECTED LEARNING OUTCOMES
# ═══════════════════════════════════════════════════════════════
"""
After completing Day 7, you should understand:

1. PARAMETER SENSITIVITY:
   - How window size affects cointegration results
   - Trade-off between recency and stability
   - Why 2000h might not be optimal for all pairs

2. TEMPORAL STABILITY:
   - Whether cointegration is consistent over time
   - How to detect structural breaks
   - When to stop trading a pair (p-value spikes)

3. PAIR QUALITY:
   - What makes a "good" pairs trade
   - How to rank multiple opportunities
   - Trade-offs between p-value and half-life

4. ROBUSTNESS:
   - Statistical rigor in backtesting
   - Avoiding overfitting to specific windows
   - Building confidence in results

INTERVIEW READY:
You'll be able to explain:
- "How do you validate cointegration?"
- "What if the relationship breaks down?"
- "How do you choose lookback periods?"
"""