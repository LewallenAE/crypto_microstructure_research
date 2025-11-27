# Research Notes

## Day 5 Observations (Correlation Analysis)

### Key Findings
- 30 pairs with correlation > 0.7 identified
- Strong clustering patterns visible in heatmap
- Layer 1 platforms (ETH, SOL, ADA, AVAX) form tight cluster
- BTC-ETH correlation: [check your data]
- Highest correlation: ADA-DOGE at 0.88

### Literature to Review
- Engle & Granger (1987) - Cointegration and Error Correction
- Gatev, Goetzmann, Rouwenhorst (2006) - Pairs Trading Performance
- [Add more as you find them]

### Questions for Investigation
1. Why is ADA-DOGE correlation so high? (Both retail-driven?)
2. Do correlations break down during high volatility?
3. Which of these 30 pairs are actually cointegrated?

### Hypotheses
- H1: Layer 1 platforms will show cointegration (substitute goods)
- H2: Correlation will be regime-dependent (breaks during stress)
- H3: Kalman filter will improve static hedge ratios

### Next Steps
- Week 6: Deep dive into cointegration theory
- Week 7: Implement Engle-Granger and Johansen tests
- Test all 30 pairs for cointegration