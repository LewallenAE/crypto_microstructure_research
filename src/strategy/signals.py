"""
WEEK 2 DAY 8: Z-SCORE SIGNAL GENERATION
Goal: Build entry/exit signals for pairs trading
Time: 2-3 hours

CONCEPTS:
- Z-score normalization (standardizing the spread)
- Entry signals (when spread deviates significantly)
- Exit signals (when spread reverts to mean)
- Position direction (long vs short the spread)

LEARNING OBJECTIVES:
- Understand why z-scores matter (scale-invariant signals)
- Learn threshold-based trading rules
- Generate actionable buy/sell signals
- Visualize trading opportunities

READING:
- Tsay Ch. 3 (ARMA models, mean reversion)
- Harris Ch. 5-6 (bid-ask spread, execution)
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from src.features.basic_analysis import load_price_data
from src.features.cointegration import (
    calculate_hedge_ratio,
    calculate_spread
)


# ═══════════════════════════════════════════════════════════════
# CONCEPT 1: Z-SCORE CALCULATION
# ═══════════════════════════════════════════════════════════════
"""
WHAT IS A Z-SCORE?

Z-score = (value - mean) / std_dev

It tells you "how many standard deviations away from the mean is this value?"

EXAMPLES:
- Spread = 100, Mean = 100, Std = 10 → Z-score = 0 (at mean)
- Spread = 120, Mean = 100, Std = 10 → Z-score = +2 (2σ above mean)
- Spread = 80,  Mean = 100, Std = 10 → Z-score = -2 (2σ below mean)

WHY Z-SCORES?
- Scale-invariant (works for any price level)
- Standard thresholds (±2σ is common for entry)
- Easy to interpret (distance from mean in std devs)

TRADING LOGIC:
- Z-score > +2: Spread too high → SHORT the spread (sell asset A, buy asset B)
- Z-score < -2: Spread too low → LONG the spread (buy asset A, sell asset B)
- Z-score near 0: Close position (spread returned to mean)
"""

def calculate_z_score(spread, window=None):
    """
    Calculate rolling z-score of the spread
    
    Z-score = (spread - rolling_mean) / rolling_std
    
    CONCEPT: We use a rolling window because the mean/std might change over time.
    If window=None, we use expanding window (all data up to that point).
    
    Parameters:
        spread (pd.Series): The spread time series
        window (int, optional): Rolling window size (e.g., 60 for 60-period)
                                If None, uses expanding window
    
    Returns:
        pd.Series: Z-score time series
    """
    # TODO: Calculate rolling (or expanding) mean
    # Hint: If window is None: rolling_mean = spread.expanding().mean()
    #       Otherwise: rolling_mean = spread.rolling(window=window).mean()
    
    # TODO: Calculate rolling (or expanding) std
    # Hint: Similar to mean, but use .std()
    
    # TODO: Calculate z-score
    # Hint: z_score = (spread - rolling_mean) / rolling_std
    
    # TODO: Return z-score
    pass


# ═══════════════════════════════════════════════════════════════
# CONCEPT 2: SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════
"""
TRADING SIGNALS:

1. ENTRY SIGNALS:
   - When z-score crosses above +entry_threshold → SHORT signal (-1)
   - When z-score crosses below -entry_threshold → LONG signal (+1)
   
2. EXIT SIGNALS:
   - When z-score returns to within ±exit_threshold → CLOSE position (0)
   
EXAMPLE with entry=2.0, exit=0.5:
   Time 1: z = -2.5 → LONG (+1) - Buy spread
   Time 2: z = -1.0 → HOLD (+1) - Still in position
   Time 3: z = 0.3  → EXIT (0) - Close position
   Time 4: z = 2.1  → SHORT (-1) - Sell spread
   Time 5: z = 1.0  → HOLD (-1) - Still in position
   Time 6: z = -0.4 → EXIT (0) - Close position

POSITION MEANINGS:
   +1 = LONG the spread (buy asset A, sell asset B)
   -1 = SHORT the spread (sell asset A, buy asset B)
    0 = No position (flat)
"""

def generate_signals(z_score, entry_threshold=2.0, exit_threshold=0.5):
    """
    Generate trading signals based on z-score thresholds
    
    Parameters:
        z_score (pd.Series): Z-score time series
        entry_threshold (float): Z-score level to enter position (default 2.0)
        exit_threshold (float): Z-score level to exit position (default 0.5)
    
    Returns:
        pd.Series: Signal series with values: +1 (long), -1 (short), 0 (no position)
    """
    # TODO: Initialize signals as Series of zeros (same index as z_score)
    # Hint: signals = pd.Series(0, index=z_score.index)
    
    # TODO: Create position tracker (what position are we currently in?)
    # Hint: position = 0  # Start with no position
    
    # TODO: Loop through z_score values
    # Hint: for i in range(len(z_score)):
    #           current_z = z_score.iloc[i]
    
        # TODO: ENTRY LOGIC - If no position currently
        # Hint: if position == 0:
        #           if current_z > entry_threshold:
        #               position = -1  # Enter SHORT
        #           elif current_z < -entry_threshold:
        #               position = 1   # Enter LONG
        
        # TODO: EXIT LOGIC - If in a position
        # Hint: elif position != 0:
        #           if abs(current_z) < exit_threshold:
        #               position = 0  # Exit position
        
        # TODO: Store current position in signals
        # Hint: signals.iloc[i] = position
    
    # TODO: Return signals
    return signals


# ═══════════════════════════════════════════════════════════════
# CONCEPT 3: CALCULATE RETURNS
# ═══════════════════════════════════════════════════════════════
"""
STRATEGY RETURNS:

When LONG spread (+1):
  - Profit if spread increases (reverts up to mean)
  - Return = spread_change * position

When SHORT spread (-1):
  - Profit if spread decreases (reverts down to mean)
  - Return = -spread_change * position = spread_change * (-1)

NO POSITION (0):
  - Return = 0

FORMULA: strategy_return = position[t-1] * spread_return[t]
(We use position from previous period because we decide position yesterday
 and earn returns today based on that decision)
"""

def calculate_strategy_returns(spread, signals):
    """
    Calculate returns from the trading strategy
    
    Parameters:
        spread (pd.Series): The spread time series
        signals (pd.Series): Position signals (+1, -1, 0)
    
    Returns:
        pd.Series: Strategy returns
    """
    # TODO: Calculate spread returns (percent change)
    # Hint: spread_returns = spread.pct_change()
    
    # TODO: Shift signals by 1 (use yesterday's position for today's return)
    # Hint: lagged_signals = signals.shift(1)
    
    # TODO: Calculate strategy returns
    # Hint: strategy_returns = lagged_signals * spread_returns
    
    # TODO: Fill NaN with 0
    # Hint: strategy_returns = strategy_returns.fillna(0)
    
    # TODO: Return strategy returns
    return strategy_returns


# ═══════════════════════════════════════════════════════════════
# CONCEPT 4: PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════
"""
KEY METRICS:

1. Total Return: Cumulative return over entire period
2. Number of Trades: How many times did we enter a position?
3. Win Rate: % of trades that were profitable
4. Sharpe Ratio: Risk-adjusted return (return / volatility)
"""

def calculate_performance_metrics(returns, signals):
    """
    Calculate key performance metrics
    
    Parameters:
        returns (pd.Series): Strategy returns
        signals (pd.Series): Position signals
    
    Returns:
        dict: Performance metrics
    """
    # TODO: Calculate total return
    # Hint: total_return = (1 + returns).prod() - 1
    
    # TODO: Count number of trades (position changes)
    # Hint: trades = (signals.diff() != 0).sum()
    
    # TODO: Calculate Sharpe ratio (annualized)
    # Assuming hourly data, there are ~8760 hours/year
    # Hint: sharpe = returns.mean() / returns.std() * np.sqrt(8760)
    
    # TODO: Calculate max drawdown
    # Hint: cumulative = (1 + returns).cumprod()
    #       running_max = cumulative.expanding().max()
    #       drawdown = (cumulative - running_max) / running_max
    #       max_dd = drawdown.min()
    
    # TODO: Return metrics as dictionary
    return {
        'total_return': total_return,
        'num_trades': trades,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }


# ═══════════════════════════════════════════════════════════════
# CONCEPT 5: VISUALIZATION
# ═══════════════════════════════════════════════════════════════
"""
PLOT: 3-panel visualization
1. Top: Spread with entry/exit points marked
2. Middle: Z-score with threshold lines
3. Bottom: Cumulative returns
"""

def plot_trading_signals(spread, z_score, signals, returns, pair_name, save_fig=True):
    """
    Visualize trading strategy with signals and returns
    """
    # TODO: Create 3-panel figure
    # Hint: fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # ============ PANEL 1: SPREAD WITH SIGNALS ============
    # TODO: Plot spread
    # Hint: axes[0].plot(spread.index, spread, linewidth=1, color='blue', alpha=0.7)
    
    # TODO: Mark LONG entries (signal changes to +1)
    # Hint: long_entries = signals[(signals == 1) & (signals.shift(1) != 1)]
    #       axes[0].scatter(long_entries.index, spread[long_entries.index], 
    #                       color='green', marker='^', s=100, label='Long Entry')
    
    # TODO: Mark SHORT entries (signal changes to -1)
    # TODO: Mark EXITS (signal changes to 0)
    
    # TODO: Add labels, title, legend, grid
    
    # ============ PANEL 2: Z-SCORE WITH THRESHOLDS ============
    # TODO: Plot z-score
    # TODO: Add horizontal lines at ±2 (entry) and ±0.5 (exit)
    # TODO: Add labels, title, legend, grid
    
    # ============ PANEL 3: CUMULATIVE RETURNS ============
    # TODO: Calculate cumulative returns
    # Hint: cumulative_returns = (1 + returns).cumprod()
    # TODO: Plot cumulative returns
    # TODO: Add labels, title, grid
    
    # TODO: Add overall title
    # TODO: plt.tight_layout()
    # TODO: Save and show
    pass


# ═══════════════════════════════════════════════════════════════
# CONCEPT 6: COMPLETE BACKTEST
# ═══════════════════════════════════════════════════════════════

def backtest_pair(symbol_a, symbol_b, window=1500, 
                  entry_threshold=2.0, exit_threshold=0.5):
    """
    Complete backtest pipeline for a pair
    
    Parameters:
        symbol_a, symbol_b: Trading pair symbols
        window: Lookback window for cointegration (use optimal from Week 1)
        entry_threshold: Z-score for entry (default 2.0)
        exit_threshold: Z-score for exit (default 0.5)
    
    Returns:
        dict: Complete backtest results
    """
    print(f"\n{'='*60}")
    print(f"BACKTESTING: {symbol_a} - {symbol_b}")
    print(f"{'='*60}\n")
    
    # TODO 1: Load data
    # Hint: df_a = load_price_data(symbol_a)
    #       df_b = load_price_data(symbol_b)
    
    # TODO 2: Use last 'window' hours
    # Hint: prices_a = df_a['close'].tail(window)
    #       prices_b = df_b['close'].tail(window)
    
    # TODO 3: Calculate hedge ratio and spread
    # Hint: hedge_ratio = calculate_hedge_ratio(prices_a, prices_b)
    #       spread = calculate_spread(prices_a, prices_b, hedge_ratio)
    
    # TODO 4: Calculate z-score
    # Hint: z_score = calculate_z_score(spread, window=60)
    
    # TODO 5: Generate signals
    # Hint: signals = generate_signals(z_score, entry_threshold, exit_threshold)
    
    # TODO 6: Calculate returns
    # Hint: returns = calculate_strategy_returns(spread, signals)
    
    # TODO 7: Calculate metrics
    # Hint: metrics = calculate_performance_metrics(returns, signals)
    
    # TODO 8: Print results
    print("PERFORMANCE METRICS:")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    
    # TODO 9: Plot results
    pair_name = f"{symbol_a.replace('USDT','')}-{symbol_b.replace('USDT','')}"
    plot_trading_signals(spread, z_score, signals, returns, pair_name)
    
    # TODO 10: Return everything
    return {
        'spread': spread,
        'z_score': z_score,
        'signals': signals,
        'returns': returns,
        'metrics': metrics
    }


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("WEEK 2 DAY 8: Z-SCORE SIGNAL GENERATION")
    print("="*60)
    
    # Test on LINK-ADA (your best pair from Week 1)
    results = backtest_pair('LINKUSDT', 'ADAUSDT', 
                           window=1500,  # Optimal window from Week 1
                           entry_threshold=2.0,
                           exit_threshold=0.5)
    
    print("\n" + "="*60)
    print("✅ DAY 8 COMPLETE!")
    print("="*60)
    print("\nYou've built your first pairs trading strategy!")
    print("Next: Day 9 - Position sizing and risk management")


# ═══════════════════════════════════════════════════════════════
# EXPECTED OUTCOMES
# ═══════════════════════════════════════════════════════════════
"""
After completing Day 8, you should have:

1. FUNCTIONS IMPLEMENTED:
   - calculate_z_score()
   - generate_signals()
   - calculate_strategy_returns()
   - calculate_performance_metrics()
   - plot_trading_signals()
   - backtest_pair()

2. PERFORMANCE METRICS:
   - Total return (hopefully positive!)
   - Number of trades (probably 5-15 for LINK-ADA)
   - Sharpe ratio (> 1.0 is good)
   - Max drawdown (< 20% is good)

3. VISUALIZATIONS:
   - 3-panel plot showing spread, z-score, and returns
   - Entry/exit points marked clearly

4. UNDERSTANDING:
   - How z-scores normalize the spread
   - Why threshold-based rules work for mean reversion
   - How to calculate strategy returns
   - Basic performance metrics

INTERVIEW READY:
"I built a threshold-based pairs trading strategy using z-score signals.
The strategy enters when the spread deviates 2 standard deviations from
the mean and exits when it reverts within 0.5 standard deviations. On
LINK-ADA over 1500 hours, this generated X trades with a Sharpe ratio of Y."
"""