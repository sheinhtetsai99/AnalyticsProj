# Portfolio Optimization Summary

## Overview
This report summarizes the results of portfolio optimization across multiple models using ETF data over a 10-year period.

## Input Data
- Asset statistics are available in: `1_input_data/asset_statistics.csv`

## Portfolio Models
Three optimization models were tested:
1. Basic Mean-Variance: `2_portfolio_models/etf_basic_mv_weights.csv`
2. Asset-Class Constrained: `2_portfolio_models/etf_constrained_mv_weights.csv`
3. Simplified 6-ETF Portfolio: `2_portfolio_models/etf_integer_mv_weights.csv`

## Model Comparison
- Detailed performance comparison: `3_performance_analysis/model_comparison.csv`

## Backtest Results
- Backtest metrics: `4_backtest_results/backtest_metrics.csv`
- Monthly returns time series: `4_backtest_results/backtest_returns.csv`

## Market Analysis
- Asset performance across market regimes: `5_market_analysis/market_regimes_analysis.csv`
- Portfolio performance across market regimes: `5_market_analysis/portfolio_periods_performance.csv`

## Visualizations
All visualization charts are available in the `charts/` directory, organized by category.

## Key Findings
- The optimized portfolio achieved an annual return of 7.67% with 10.12% volatility
- The portfolio outperformed the 60/40 benchmark by 0.61 units of risk (Information Ratio)
- Maximum drawdown during the 10-year period was -18.21%

