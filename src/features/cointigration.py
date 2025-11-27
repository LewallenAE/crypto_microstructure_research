"""
DAY 6: COINTEGRATION TESTING
Goal: Find which correlated pairs are actually cointegrated (mean-reverting)
Time: 1-2 hours

CONCEPTS:
- Correlation vs Cointegration
- Stationarity and mean reversion
- Augmented Dickey-Fuller (ADF) test
- Hedge ratio calculation
- Half-life of mean reversion

LEARNING OBJECTIVES:
- Understand why correlation ≠ cointegration
- Learn to test for stationarity using ADF
- Calculate optimal hedge ratios
- Measure mean reversion speed (half-life)
- Identify tradeable pairs

INSTALL:
pip install statsmodels
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


# ═══════════════════════════════════════════════════════════════
# CONCEPT CHECK: What's the difference?
# ═══════════════════════════════════════════════════════════════
# CORRELATION: Two assets move together in the SHORT TERM
#   - BTC goes up 5%, ETH goes up 4% → Correlated
#   - But tomorrow BTC might be at $50k, ETH at $3k
#   - Next year BTC might be $100k, ETH at $2k
#   - They moved together but DIVERGED permanently
#
# COINTEGRATION: Two assets have LONG-TERM equilibrium
#   - BTC and ETH maintain a stable ratio over time
#   - When they diverge, they REVERT back to ratio
#   - Like a rubber band - stretches but snaps back
#   - THIS is what we can trade!
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# STEP 1: Calculate Hedge Ratio (Beta)
# ═══════════════════════════════════════════════════════════════

def calculate_hedge_ratio(prices_a, prices_b):
    """
    Calculate the hedge ratio between two price series using linear regression
    
    CONCEPT: The hedge ratio tells us:
    "For every $1 of asset A I buy, how much of asset B should I short?"
    
    We're finding: price_a = beta * price_b + alpha
    Beta is the hedge ratio (slope of the line)
    
    Example: If beta = 2.5, then:
    - Buy $100 of BTC → Short $250 of ETH
    - This makes our position "market neutral"
    
    Parameters:
        prices_a (pd.Series): Price series for asset A
        prices_b (pd.Series): Price series for asset B
    
    Returns:
        float: Hedge ratio (beta)
    """
    # TODO 1.1: Align the two series using pd.concat
    # Some timestamps might be missing in one series
    # Hint: df = pd.concat([prices_a, prices_b], axis=1).dropna()
    df = pd.concat([prices_a, prices_b], axis = 1).dropna()
    
    # TODO 1.2: Extract y and X for regression
    # y = Asset A prices (what we're predicting)
    # X = Asset B prices (what we're using to predict)
    y = df.iloc[:, 0]
    X = df.iloc[:,1]
    
    # TODO 1.3: Run OLS regression
    # OLS = Ordinary Least Squares (linear regression)
    # This finds the best-fit line: y = beta * X + alpha
    # Hint: model = OLS(y, X).fit()
    model = OLS(y, X).fit()
    
    # TODO 1.4: Extract beta (the coefficient/slope)
    # model.params[0] gives you the beta coefficient
    beta = model.params[0]
    
    return beta


# ═══════════════════════════════════════════════════════════════
# STEP 2: Calculate the Spread
# ═══════════════════════════════════════════════════════════════

def calculate_spread(prices_a, prices_b, hedge_ratio):
    """
    Calculate the spread between two assets
    
    CONCEPT: The spread is the "distance" between the two assets
    Formula: spread = price_a - hedge_ratio * price_b
    
    If cointegrated, this spread should:
    - Oscillate around a mean (doesn't trend up/down forever)
    - Revert back to mean when it deviates
    - Stay within predictable bounds
    
    Example:
    - BTC = $50,000, ETH = $3,000, hedge_ratio = 15
    - Spread = 50000 - 15*3000 = 50000 - 45000 = 5000
    - If spread goes to 8000, we expect it to revert to ~5000
    
    Parameters:
        prices_a (pd.Series): Price series for asset A
        prices_b (pd.Series): Price series for asset B
        hedge_ratio (float): The hedge ratio from Step 1
    
    Returns:
        pd.Series: The spread (should be mean-reverting if cointegrated)
    """
    # TODO 2.1: Align both price series
    # Hint: Same as Step 1.1
    df = pd.concat([prices_a, prices_b], axis = 1).dropna()
    
    # TODO 2.2: Calculate spread using the formula above
    # spread = asset_a_prices - hedge_ratio * asset_b_prices
    # Hint: df.iloc[:, 0] is asset A, df.iloc[:, 1] is asset B
    spread = df.iloc[:, 0] - (hedge_ratio * df.iloc[:, 1])
    
    return spread


# ═══════════════════════════════════════════════════════════════
# STEP 3: Test for Stationarity (ADF Test)
# ═══════════════════════════════════════════════════════════════

def adf_test(series):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    
    CONCEPT: Stationarity means the series:
    - Has constant mean over time (doesn't trend)
    - Has constant variance over time
    - Reverts to its mean (mean-reverting)
    
    WHY THIS MATTERS:
    - If spread is stationary → We can trade it!
    - If spread trends → We can't predict it → Don't trade
    
    THE TEST:
    - Null Hypothesis (H0): Series is NOT stationary (has unit root)
    - Alternative (H1): Series IS stationary
    - If p-value < 0.05: Reject H0 → Series IS stationary ✅
    - If p-value > 0.05: Cannot reject H0 → Series NOT stationary ❌
    
    Parameters:
        series (pd.Series): The time series to test (our spread)
    
    Returns:
        dict: Test results including p-value and conclusion
    """
    # TODO 3.1: Run ADF test
    # adfuller() returns a tuple of results
    # autolag='AIC' automatically picks the best lag order
    # Hint: result = adfuller(series, autolag='AIC')
    result = adfuller(series, autolag='AIC')
    
    # TODO 3.2: Extract the key statistics from result tuple
    # result[0] = ADF test statistic (more negative = more stationary)
    # result[1] = p-value (< 0.05 means stationary)
    # result[4] = dictionary of critical values
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    # TODO 3.3: Determine if series is stationary
    # Check if p_value < 0.05
    is_stationary = p_value < 0.05 # YOUR CODE HERE (hint: p_value < 0.05)
    
    # Return all the important info
    return {
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_stationary': is_stationary
    }


# ═══════════════════════════════════════════════════════════════
# STEP 4: Calculate Half-Life of Mean Reversion
# ═══════════════════════════════════════════════════════════════

def calculate_half_life(spread):
    """
    Calculate the half-life of mean reversion
    
    CONCEPT: Half-life tells us:
    "If spread deviates by X, how long until it reverts halfway back to mean?"
    
    Example:
    - Spread mean = 100
    - Spread goes to 120 (deviation = 20)
    - Half-life = 10 hours
    - In 10 hours, spread should be at ~110 (halfway back)
    
    WHY THIS MATTERS:
    - Short half-life (5-20 hours) = Fast mean reversion = Good for trading
    - Long half-life (100+ hours) = Slow reversion = Bad for trading
    - Infinity = Not mean-reverting at all = Don't trade!
    
    THE MATH:
    We fit an AR(1) model: spread[t] - spread[t-1] = α + β*spread[t-1]
    Half-life = -log(2) / log(1 + β)
    
    Parameters:
        spread (pd.Series): The spread series
    
    Returns:
        float: Half-life in time periods (hours for us)
    """
    # TODO 4.1: Create lagged spread (spread shifted by 1)
    # We need spread[t-1] to predict spread[t]
    # Hint: spread_lag = spread.shift(1)
    spread_lag = spread.shift(1)
    
    # TODO 4.2: Calculate spread change (spread[t] - spread[t-1])
    # Hint: spread_ret = spread - spread_lag
    spread_ret = spread - spread_lag
    
    # TODO 4.3: Combine and drop NaN
    # First row will be NaN because of shift
    # Hint: df = pd.concat([spread_ret, spread_lag], axis=1).dropna()
    df = pd.concat([spread_ret, spread_lag], axis = 1).dropna()
    
    # TODO 4.4: Run regression: spread_change ~ spread_lag
    # We're regressing spread_ret on spread_lag
    # Hint: model = OLS(df.iloc[:, 0], df.iloc[:, 1]).fit()
    model = OLS(df.iloc[:, 0], df.iloc[:, 1]).fit() # YOUR CODE HERE
    
    # TODO 4.5: Extract beta coefficient
    beta = model.params[0] # YOUR CODE HERE
    
    # TODO 4.6: Calculate half-life using the formula
    # If beta >= 0, spread is NOT mean-reverting (return infinity)
    # Otherwise: half_life = -log(2) / log(1 + beta)
    # Hint: Use np.log() for natural log, np.inf for infinity
    if beta >= 0:
        return np.inf # YOUR CODE HERE (hint: np.inf)
    
    half_life = -np.log(2) / np.log(1 + beta)# YOUR CODE HERE (use the formula above)
    
    return half_life


# ═══════════════════════════════════════════════════════════════
# STEP 5: Test a Pair for Cointegration (Put it all together!)
# ═══════════════════════════════════════════════════════════════

def test_cointegration(symbol_a, symbol_b):
    """
    Complete cointegration test for a pair
    
    This combines all the steps above:
    1. Load price data
    2. Calculate hedge ratio
    3. Calculate spread
    4. Test if spread is stationary (ADF)
    5. Calculate half-life
    6. Determine if pair is tradeable
    
    Parameters:
        symbol_a (str): First trading pair
        symbol_b (str): Second trading pair
    
    Returns:
        dict: Complete cointegration results or None if failed
    """
    # TODO 5.1: Load price data for both assets
    # Hint: Use load_price_data() from Day 3
    df_a = load_price_data(symbol_a)
    df_b = load_price_data(symbol_b)
    
    # Check if loading succeeded
    if df_a is None or df_b is None:
        return None
    
    # TODO 5.2: Extract close prices
    prices_a = df_a['close'] # YOUR CODE HERE (hint: df_a['close'])
    prices_b = df_b['close'] # YOUR CODE HERE
    
    # TODO 5.3: Calculate hedge ratio
    # Hint: Use calculate_hedge_ratio() from Step 1
    hedge_ratio = calculate_hedge_ratio(prices_a, prices_b)# YOUR CODE HERE
    
    # TODO 5.4: Calculate spread
    # Hint: Use calculate_spread() from Step 2
    spread = calculate_spread(prices_a, prices_b, hedge_ratio)# YOUR CODE HERE
    
    # TODO 5.5: Test spread for stationarity
    # Hint: Use adf_test() from Step 3
    adf_results = adf_test(spread)# YOUR CODE HERE
    
    # TODO 5.6: Calculate half-life
    # Hint: Use calculate_half_life() from Step 4
    half_life = calculate_half_life(spread)# YOUR CODE HERE
    
    # TODO 5.7: Determine if pair is tradeable
    # A pair is "cointegrated" (tradeable) if:
    # - Spread is stationary (p-value < 0.05)
    # - Half-life is reasonable (< 100 hours)
    # Hint: is_cointegrated = adf_results['is_stationary'] and half_life < 100
    is_cointegrated = adf_results['is_stationary'] and half_life < 100# YOUR CODE HERE
    
    # TODO 5.8: Compile all results into a dictionary
    results = {
        'asset_a': symbol_a,
        'asset_b': symbol_b,
        'hedge_ratio': hedge_ratio,
        'adf_statistic': adf_results['adf_statistic'],
        'p_value': adf_results['p_value'],
        'is_stationary': adf_results['is_stationary'],
        'half_life': half_life,
        'is_cointegrated': is_cointegrated
    }
    
    return results


# ═══════════════════════════════════════════════════════════════
# STEP 6: Test All Correlated Pairs from Day 5
# ═══════════════════════════════════════════════════════════════

def test_all_pairs(pairs_df):
    """
    Test all pairs from Day 5 correlation analysis for cointegration
    
    Parameters:
        pairs_df (pd.DataFrame): DataFrame with 'asset_a' and 'asset_b' columns
    
    Returns:
        pd.DataFrame: Results with cointegration statistics
    """
    results = []
    
    print(f"Testing {len(pairs_df)} pairs for cointegration...")
    print("This might take a few minutes...\n")
    
    # TODO 6.1: Loop through each pair in pairs_df
    # Hint: for idx, row in pairs_df.iterrows():
    for idx, row in pairs_df.iterrows():
        symbol_a = row['asset_a']
        symbol_b = row['asset_b']
        
        print(f"Testing {symbol_a} - {symbol_b}...", end=" ")
        
        # TODO 6.2: Test this pair using test_cointegration()
        result = test_cointegration(symbol_a, symbol_b)# YOUR CODE HERE
        
        # Check if test failed
        if result is None:
            print("❌ Failed to load data")
            continue
        
        # TODO 6.3: Add the original correlation to results
        # Hint: result['correlation'] = row['correlation']
        result['correlation'] = row['correlation'] # YOUR CODE HERE
        
        # TODO 6.4: Append to results list
        # YOUR CODE HERE
        results.append(result)

        # Print quick status
        status = "✅ COINTEGRATED" if result['is_cointegrated'] else "❌ Not cointegrated"
        print(f"{status} (p={result['p_value']:.4f}, HL={result['half_life']:.1f})")
    
    # TODO 6.5: Convert results list to DataFrame
    results_df = pd.DataFrame(results)# YOUR CODE HERE (hint: pd.DataFrame(results))
    
    # TODO 6.6: Sort by p-value (lowest p-value = most cointegrated)
    # Hint: results_df.sort_values('p_value')
    results_df = results_df.sort_values('p_value') # YOUR CODE HERE
    
    return results_df


# ═══════════════════════════════════════════════════════════════
# STEP 7: Visualize Spread for a Cointegrated Pair
# ═══════════════════════════════════════════════════════════════

def plot_spread(symbol_a, symbol_b, save_fig=True):
    """
    Visualize the spread between two assets
    
    This creates a 2-panel plot:
    - Top: Normalized prices (both start at 100)
    - Bottom: The spread with mean and standard deviation bands
    """
    # TODO 7.1: Load data for both assets
    df_a = load_price_data(symbol_a) # YOUR CODE HERE
    df_b = load_price_data(symbol_b)# YOUR CODE HERE
    
    if df_a is None or df_b is None:
        print("Failed to load data")
        return
    
    prices_a = df_a['close']
    prices_b = df_b['close']
    
    # TODO 7.2: Calculate hedge ratio and spread
    hedge_ratio = calculate_hedge_ratio(prices_a, prices_b)# YOUR CODE HERE
    spread = calculate_spread(prices_a, prices_b, hedge_ratio)# YOUR CODE HERE
    
    # TODO 7.3: Calculate spread statistics
    # We need mean and standard deviation
    spread_mean = spread.mean()# YOUR CODE HERE (hint: spread.mean())
    spread_std = spread.std()# YOUR CODE HERE (hint: spread.std())
    
    # TODO 7.4: Run ADF test and calculate half-life
    adf_results = adf_test(spread) # YOUR CODE HERE
    half_life = calculate_half_life(spread)# YOUR CODE HERE
    
    # TODO 7.5: Create figure with 2 subplots (2 rows, 1 column)
    # Hint: fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig, axes = plt.subplots(2,1, figsize=(14, 10))# YOUR CODE HERE
    
    # ---- SUBPLOT 1: Normalized Prices ----
    # TODO 7.6: Normalize both price series to start at 100
    # Formula: normalized = (price / first_price) * 100
    norm_a =  (prices_a / prices_a.iloc[0] * 100)# YOUR CODE HERE
    norm_b = (prices_b / prices_b.iloc[0] * 100)# YOUR CODE HERE
    
    # TODO 7.7: Plot both normalized prices on axes[0]
    # Hint: axes[0].plot(prices_a.index, norm_a, label=..., linewidth=2)
    axes[0].plot(prices_a.index, norm_a, label = symbol_a.replace('USDT', ''), linewidth=2)# YOUR CODE HERE (plot norm_a)
    axes[0].plot(prices_b.index, norm_b, label = symbol_b.replace('USDT', ''), linewidth=2)# YOUR CODE HERE (plot norm_b)



    
    # TODO 7.8: Add labels and formatting to top subplot
    axes[0].set_title('Normalized Prices (Start = 100)', fontweight='bold')
    axes[0].set_ylabel('Normalized Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ---- SUBPLOT 2: Spread with Bands ----
    # TODO 7.9: Plot the spread on axes[1]
    axes[1].plot(spread.index, spread, linewidth=2, color='purple', label='Spread')# YOUR CODE HERE (plot spread)
    
    # TODO 7.10: Add horizontal lines for mean and standard deviations
    # - Black dashed line at mean
    # - Red dashed lines at ±1 std
    # - Orange dotted lines at ±2 std
    # Hint: axes[1].axhline(y=spread_mean, color='black', linestyle='--', label='Mean')
    # YOUR CODE HERE (add mean line)
    # YOUR CODE HERE (add +1 std line: spread_mean + spread_std)
    # YOUR CODE HERE (add -1 std line: spread_mean - spread_std)
    # YOUR CODE HERE (add +2 std line)
    # YOUR CODE HERE (add -2 std line)

    axes[1].axhline(y=spread_mean, color='black', linestyle='--', label='Mean')
    axes[1].axhline(y=spread_mean + spread_std, color = 'grey', linestyle = ':', alpha = 0.7, label = '+1 std')
    axes[1].axhline(y=spread_mean - spread_std, color = 'grey', linestyle = ':', alpha = 0.7, label = '-1 std')
    axes[1].axhline(y=spread_mean + 2 * spread_std, color = 'red', linestyle = ':', alpha = 0.7, label = '+2 std')
    axes[1].axhline(y=spread_mean - 2 * spread_std, color = 'red', linestyle = ':', alpha = 0.7, label = '-2 std')
    axes[1].legend()
    
    
    # TODO 7.11: Add labels and formatting to bottom subplot
    axes[1].set_title(f'Spread (Hedge Ratio: {hedge_ratio:.4f})', fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Spread Value')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # TODO 7.12: Add statistics text box
    # Show ADF p-value, half-life, and whether stationary
    stats_text = f'ADF p-value: {adf_results["p_value"]:.4f}\n'
    stats_text += f'Half-life: {half_life:.1f} hours\n'
    stats_text += f'Stationary: {"YES" if adf_results["is_stationary"] else "NO"}'
    
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', 
                 facecolor='wheat', alpha=0.5), fontsize=10)
    
    # Overall title
    fig.suptitle(f'Cointegration Analysis: {symbol_a} vs {symbol_b}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_fig:
        figures_dir = PROJECT_ROOT / 'results' / 'figures'
        os.makedirs(figures_dir, exist_ok=True)
        name_a = symbol_a.replace('USDT', '').lower()
        name_b = symbol_b.replace('USDT', '').lower()
        filepath = figures_dir / f'spread_{name_a}_{name_b}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════
# STEP 8: Save Results
# ═══════════════════════════════════════════════════════════════

def save_cointegration_results(results_df):
    """
    Save cointegration test results to CSV files
    """
    tables_dir = PROJECT_ROOT / 'results' / 'tables'
    os.makedirs(tables_dir, exist_ok=True)
    
    # TODO 8.1: Save all results to CSV
    filepath = tables_dir / 'cointegration_results.csv'
    results_df.to_csv(filepath, index=False)# YOUR CODE HERE (hint: results_df.to_csv(...))
    print(f"\nSaved all results: {filepath}")
    
    # TODO 8.2: Filter for only cointegrated pairs
    # Hint: cointegrated = results_df[results_df['is_cointegrated']]
    cointegrated = results_df[results_df['is_cointegrated']] # YOUR CODE HERE
    
    # TODO 8.3: If we found any, save them separately
    if len(cointegrated) > 0:
        filepath_coint = tables_dir / 'cointegrated_pairs.csv'
        cointegrated.to_csv(filepath_coint, index=False) # YOUR CODE HERE (save cointegrated DataFrame)
        print(f"Saved cointegrated pairs: {filepath_coint}")
        print(f"\n✅ Found {len(cointegrated)} cointegrated pairs!")
    else:
        print("\n⚠️  No cointegrated pairs found with current thresholds")
        print("Try: Lower correlation threshold in Day 5, or fetch more historical data")


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("DAY 6: COINTEGRATION TESTING")
    print("="*60)
    
    # TODO: Load top correlated pairs from Day 5
    pairs_path = PROJECT_ROOT / 'results' / 'tables' / 'top_correlated_pairs.csv'
    pairs_df = pd.read_csv(pairs_path)
    
    print(f"\nLoaded {len(pairs_df)} correlated pairs from Day 5")
    
    # TODO: Test all pairs for cointegration
    print("\n" + "="*60)
    print("TESTING FOR COINTEGRATION")
    print("="*60)
    
    results_df = test_all_pairs(pairs_df)
    
    # TODO: Display summary statistics
    print("\n" + "="*60)
    print("COINTEGRATION SUMMARY")
    print("="*60)
    
    print(f"\nTotal pairs tested: {len(results_df)}")
    print(f"Cointegrated pairs: {results_df['is_cointegrated'].sum()}")
    print(f"Success rate: {results_df['is_cointegrated'].mean()*100:.1f}%")
    
    # Show top 10 most cointegrated (lowest p-values)
    print("\n📊 TOP 10 MOST COINTEGRATED PAIRS:")
    print(results_df[['asset_a', 'asset_b', 'p_value', 'half_life', 'is_cointegrated']].head(10))
    
    # TODO: Save results
    save_cointegration_results(results_df)
    
    # TODO: Plot the best cointegrated pairs
    cointegrated = results_df[results_df['is_cointegrated']]
    
    if len(cointegrated) > 0:
        print("\n" + "="*60)
        print("VISUALIZING BEST PAIRS")
        print("="*60)
        
        # Plot top 3 cointegrated pairs
        num_to_plot = min(3, len(cointegrated))
        print(f"\nPlotting top {num_to_plot} cointegrated pairs...")
        
        for i in range(num_to_plot):
            pair = cointegrated.iloc[i]
            print(f"\n{i+1}. {pair['asset_a']} - {pair['asset_b']}")
            print(f"   P-value: {pair['p_value']:.4f}")
            print(f"   Half-life: {pair['half_life']:.1f} hours")
            plot_spread(pair['asset_a'], pair['asset_b'])
    
    print("\n" + "="*60)
    print("✅ DAY 6 COMPLETE!")
    print("="*60)
    print("\nKey outputs:")
    print("- results/tables/cointegration_results.csv")
    print("- results/tables/cointegrated_pairs.csv")
    print("- results/figures/spread_*.png")
    print("\nNext: Day 7 - Refine cointegration analysis and parameter selection")


# ═══════════════════════════════════════════════════════════════
# WHAT TO SUBMIT:
# ═══════════════════════════════════════════════════════════════
# 1. Your completed code
# 2. Screenshots:
#    - Terminal output showing cointegration test results
#    - At least 2-3 spread plots
#    - The cointegrated_pairs.csv file
# 3. Answer these questions:
#    - How many cointegrated pairs did you find?
#    - What's your best pair (lowest p-value)?
#    - What's the half-life of your best pair?
#    - Explain in your own words: Why does high correlation NOT
#      guarantee cointegration?
#
# EXPECTED RESULTS:
# - You should find 5-15 cointegrated pairs (varies by data)
# - P-values should be < 0.05 for cointegrated pairs
# - Half-lives should be 10-50 hours for good pairs
# - Spreads should oscillate around mean in your plots
#
# DEBUGGING TIPS:
# - If no pairs found: Check your Day 5 correlation threshold
# - If code crashes: Verify CSV files from Days 2-3 exist
# - If results look weird: Print intermediate values to debug
# ═══════════════════════════════════════════════════════════════