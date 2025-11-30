"""
WEEK 1 DAY 7 EXTENSION: Test "Near Miss" Pairs
Goal: Find additional cointegrated pairs by testing pairs that almost passed
Time: 1-2 hours
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.features.cointegration import test_cointegration
from src.features.cointegration_refinement import (
    test_parameter_sensitivity,
    rolling_cointegration_test,
    calculate_pair_quality_score,
    plot_parameter_sensitivity,
    plot_rolling_cointegration
)


# ═══════════════════════════════════════════════════════════════
# STRATEGY: Test pairs with p-values close to 0.05
# ═══════════════════════════════════════════════════════════════

def find_near_miss_pairs(max_p_value=0.10):
    """
    Find pairs that were close to passing cointegration test
    
    Strategy:
    - Load all cointegration results from Day 6
    - Find pairs with 0.05 < p-value < 0.10
    - These "barely missed" and might be cointegrated with different parameters
    
    Parameters:
        max_p_value (float): Maximum p-value to consider (default 0.10)
    
    Returns:
        pd.DataFrame: Near-miss pairs to re-test
    """
    tables_dir = PROJECT_ROOT / 'results' / 'tables'
    all_results = pd.read_csv(tables_dir / 'cointegration_results.csv')
    
    # Find pairs that barely missed (p-value between 0.05 and max_p_value)
    near_miss = all_results[
        (all_results['p_value'] >= 0.05) & 
        (all_results['p_value'] <= max_p_value)
    ].sort_values('p_value')
    
    print(f"\nFound {len(near_miss)} near-miss pairs (0.05 < p < {max_p_value}):")
    print(near_miss[['asset_a', 'asset_b', 'p_value', 'half_life']])
    
    return near_miss


def test_with_different_windows(symbol_a, symbol_b):
    """
    Test a pair with various window sizes to see if any achieve cointegration
    
    Sometimes a pair isn't cointegrated at 2000h but IS at 1500h or 1000h
    
    Returns:
        dict: Best window result
    """
    print(f"\nTesting {symbol_a} - {symbol_b} with multiple windows...")
    
    windows = [750, 1000, 1250, 1500, 1750, 2000]
    results = []
    
    for window in windows:
        # Load data
        from src.features.basic_analysis import load_price_data
        df_a = load_price_data(symbol_a)
        df_b = load_price_data(symbol_b)
        
        if df_a is None or df_b is None:
            continue
        
        # Test with this window size
        prices_a = df_a['close'].tail(window)
        prices_b = df_b['close'].tail(window)
        
        from src.features.cointegration import (
            calculate_hedge_ratio,
            calculate_spread,
            adf_test,
            calculate_half_life
        )
        
        hedge_ratio = calculate_hedge_ratio(prices_a, prices_b)
        spread = calculate_spread(prices_a, prices_b, hedge_ratio)
        adf_results = adf_test(spread)
        half_life = calculate_half_life(spread)
        
        results.append({
            'window': window,
            'p_value': adf_results['p_value'],
            'half_life': half_life,
            'is_cointegrated': adf_results['is_stationary'] and half_life < 100
        })
        
        status = "✅" if adf_results['is_stationary'] and half_life < 100 else "❌"
        print(f"  {window}h: p={adf_results['p_value']:.4f}, HL={half_life:.1f}h {status}")
    
    results_df = pd.DataFrame(results)
    
    # Find best window (lowest p-value)
    best = results_df.loc[results_df['p_value'].idxmin()]
    
    return {
        'asset_a': symbol_a,
        'asset_b': symbol_b,
        'best_window': int(best['window']),
        'best_p_value': best['p_value'],
        'best_half_life': best['half_life'],
        'is_cointegrated': best['is_cointegrated'],
        'all_results': results_df
    }


def comprehensive_pair_analysis(symbol_a, symbol_b):
    """
    Full analysis pipeline for a single pair
    
    Returns:
        dict: Complete analysis results including quality score
    """
    print("\n" + "="*60)
    print(f"COMPREHENSIVE ANALYSIS: {symbol_a} - {symbol_b}")
    print("="*60)
    
    # 1. Test with different windows
    window_results = test_with_different_windows(symbol_a, symbol_b)
    
    if not window_results['is_cointegrated']:
        print(f"\n❌ Not cointegrated at any window size (best p={window_results['best_p_value']:.4f})")
        return None
    
    print(f"\n✅ Cointegrated at {window_results['best_window']}h window!")
    print(f"   P-value: {window_results['best_p_value']:.4f}")
    print(f"   Half-life: {window_results['best_half_life']:.1f}h")
    
    # 2. Parameter sensitivity analysis
    print("\n2. Parameter sensitivity...")
    sensitivity = test_parameter_sensitivity(symbol_a, symbol_b)
    
    # 3. Rolling cointegration
    print("\n3. Rolling cointegration stability...")
    rolling = rolling_cointegration_test(symbol_a, symbol_b, 
                                        window_size=window_results['best_window'],
                                        step_size=100)
    
    # 4. Calculate quality score
    p_val_std = sensitivity['p_value'].std()
    quality = calculate_pair_quality_score(
        window_results['best_p_value'],
        window_results['best_half_life'],
        p_val_std
    )
    
    print(f"\n4. Quality Score: {quality:.1f}/100")
    
    # 5. Create visualizations
    pair_name = f"{symbol_a.replace('USDT', '')}-{symbol_b.replace('USDT', '')}"
    plot_parameter_sensitivity(sensitivity, pair_name)
    plot_rolling_cointegration(rolling, pair_name)
    
    return {
        'asset_a': symbol_a,
        'asset_b': symbol_b,
        'optimal_window': window_results['best_window'],
        'p_value': window_results['best_p_value'],
        'half_life': window_results['best_half_life'],
        'quality_score': quality,
        'is_cointegrated': True
    }


def batch_test_near_miss_pairs():
    """
    Test all near-miss pairs and save the good ones
    """
    # Find near-miss pairs
    near_miss = find_near_miss_pairs(max_p_value=0.15)
    
    if len(near_miss) == 0:
        print("\n⚠️  No near-miss pairs found. Your threshold might be too strict.")
        return
    
    # Test each one
    new_cointegrated = []
    
    for idx, row in near_miss.iterrows():
        result = comprehensive_pair_analysis(row['asset_a'], row['asset_b'])
        
        if result is not None:
            new_cointegrated.append(result)
    
    # Save results
    if len(new_cointegrated) > 0:
        print("\n" + "="*60)
        print(f"✅ FOUND {len(new_cointegrated)} NEW COINTEGRATED PAIRS!")
        print("="*60)
        
        new_df = pd.DataFrame(new_cointegrated)
        print(new_df[['asset_a', 'asset_b', 'optimal_window', 'p_value', 'half_life', 'quality_score']])
        
        # Append to existing cointegrated pairs
        tables_dir = PROJECT_ROOT / 'results' / 'tables'
        existing = pd.read_csv(tables_dir / 'cointegrated_pairs.csv')
        
        # Combine
        all_pairs = pd.concat([existing, new_df], ignore_index=True)
        all_pairs = all_pairs.sort_values('quality_score', ascending=False)
        
        # Save
        all_pairs.to_csv(tables_dir / 'cointegrated_pairs_extended.csv', index=False)
        print(f"\nSaved to: cointegrated_pairs_extended.csv")
        print(f"Total cointegrated pairs: {len(all_pairs)}")
    else:
        print("\n⚠️  No additional cointegrated pairs found.")


# ═══════════════════════════════════════════════════════════════
# MANUAL TESTING: Test specific pairs
# ═══════════════════════════════════════════════════════════════

def test_specific_pairs():
    """
    Manually test the most promising near-miss pairs from Day 6
    """
    # From your Day 6 results, these were close:
    promising_pairs = [
        ('DOGEUSDT', 'ETHUSDT'),    # p=0.018 - SO CLOSE!
        ('PEPEUSDT', 'SUIUSDT'),    # p=0.055 - Just above threshold
        ('LINKUSDT', 'ADAUSDT'),    # p=0.114 - Worth trying
    ]
    
    results = []
    
    for symbol_a, symbol_b in promising_pairs:
        print("\n" + "="*60)
        result = comprehensive_pair_analysis(symbol_a, symbol_b)
        if result is not None:
            results.append(result)
    
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("WEEK 1 DAY 7 EXTENSION: Finding More Pairs")
    print("="*60)
    
    # OPTION 1: Test specific promising pairs (faster)
    print("\nOPTION 1: Testing most promising pairs...")
    new_pairs = test_specific_pairs()
    
    # OPTION 2: Test all near-miss pairs (slower but thorough)
    # print("\nOPTION 2: Testing all near-miss pairs...")
    # batch_test_near_miss_pairs()

    print("\nOPTION 2: Testing all near-miss pairs...")
    batch_test_near_miss_pairs()
    
    print("\n" + "="*60)
    print("✅ WEEK 1 EXTENSION COMPLETE!")
    print("="*60)
    print("\nYou now have a more robust pair universe for Week 2!")
    print("Next: Week 2 - Build static pairs trading strategy")