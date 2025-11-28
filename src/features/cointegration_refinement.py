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
    # Loop through windows
    # For each window: calculate stats
    # Return DataFrame with results
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    
    print("\n✅ Day 7 skeleton ready - implement the concepts!")


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