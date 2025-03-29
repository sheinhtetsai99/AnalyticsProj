# Portfolio Optimization Results

This directory contains the results of portfolio optimization analysis, organized by workflow stage.

## Directory Structure

### 1_input_data/
Contains the base statistics for all assets used in the optimization.

### 2_portfolio_models/
Contains the portfolio weights from each optimization model:
- `etf_basic_mv_weights.csv`: Basic mean-variance optimization (typically 2 ETFs)
- `etf_constrained_mv_weights.csv`: Asset-class constrained portfolio (typically 7 ETFs)
- `etf_integer_mv_weights.csv`: Simplified portfolio with 6 ETFs

### 3_performance_analysis/
Contains comparative analysis of model performance:
- `model_comparison.csv`: Side-by-side metrics for all models

### 4_backtest_results/
Contains historical simulation results:
- `backtest_metrics.csv`: Performance summary (returns, volatility, etc.)
- `backtest_returns.csv`: Detailed monthly return time series

### 5_market_analysis/
Contains analysis across different market conditions:
- `market_regimes_analysis.csv`: Asset performance in different time periods
- `portfolio_periods_performance.csv`: Portfolio performance across different markets

### charts/
Contains all visualizations, organized by category.

### reports/
Contains summary reports and documentation.
