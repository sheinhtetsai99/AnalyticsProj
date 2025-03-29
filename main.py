import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import os
from portfolio_builder import PortfolioBuilder

# Set seed for reproducibility
np.random.seed(42)

def create_streamlined_structure(base_dir='portfolio_results'):
    """Create a clean, workflow-oriented directory structure"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    # Create directories following the analysis workflow
    folders = [
        '1_input_data',
        '2_portfolio_models',
        '3_performance_analysis',
        '4_backtest_results',
        '5_market_analysis',
        'charts',
        'reports'  # Added reports folder explicitly
    ]
    
    # Create chart subdirectories
    chart_subdirs = [
        'portfolio_weights',
        'efficient_frontier',
        'backtest',
        'comparison'
    ]
    
    # Create each directory
    dirs = {}
    # First add the base directory
    dirs['base_dir'] = base_dir
    
    # Create directories
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        dirs[folder] = folder_path
        
    # Create chart subdirectories
    for subdir in chart_subdirs:
        subdir_path = os.path.join(dirs['charts'], subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        dirs[f'charts/{subdir}'] = subdir_path
    
    # Create a README file
    readme_content = """# Portfolio Optimization Results

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
"""

    with open(os.path.join(base_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    print(f"Created streamlined directory structure in '{base_dir}'")
    return dirs

def fetch_etf_data(start_date=None, end_date=None):
    """Fetch data for a diverse set of ETFs across asset classes and regions"""
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=10*365)  # 10 years of data
    
    print(f"Fetching ETF data from {start_date.date()} to {end_date.date()} (10-year period)")
    
    # Define ETFs by asset class
    etfs = {
        # Equities by Region
        'SPY': 'US Equities',          # S&P 500
        'VGK': 'European Equities',    # Vanguard FTSE Europe
        'EEM': 'Emerging Markets',     # Emerging Markets
        'VPL': 'Asia-Pacific',         # Vanguard Pacific

        # Fixed Income
        'AGG': 'US Aggregate Bonds',   # US Aggregate Bond
        'TLT': 'Long-Term Treasury',   # 20+ Year Treasury
        'LQD': 'Corporate Bonds',      # Investment Grade Corporate Bonds
        'BNDX': 'International Bonds', # Vanguard Total International Bond

        # Alternative Assets
        'GLD': 'Gold',                 # Gold
        'SLV': 'Silver',               # Silver
        'VNQ': 'Real Estate',          # Vanguard Real Estate
        'GSG': 'Commodities'           # iShares S&P GSCI Commodity
    }
    
    # Download historical data
    tickers = list(etfs.keys())
    data = yf.download(tickers, start=start_date, end=end_date, interval='1mo')
    
    # Use Close prices instead of Adj Close if needed
    if 'Close' in data.columns and 'Adj Close' not in data.columns:
        prices = data['Close']
    elif isinstance(data.columns, pd.MultiIndex) and ('Adj Close', tickers[0]) in data.columns:
        prices = data['Adj Close']
    elif isinstance(data.columns, pd.MultiIndex) and ('Close', tickers[0]) in data.columns:
        prices = data['Close']
    else:
        if isinstance(data.columns, pd.MultiIndex):
            first_level = data.columns.levels[0][0]
            prices = data[first_level]
        else:
            prices = data
    
    # Calculate monthly returns
    returns = prices.pct_change().dropna()
    
    print(f"Downloaded data for {len(tickers)} ETFs from {start_date.date()} to {end_date.date()}")
    print(f"Time periods: {len(returns)}")
    
    # Save asset class mapping for later use
    asset_class_mapping = pd.Series(etfs)
    
    return returns, prices, asset_class_mapping

def generate_sample_data(n_assets=12, n_periods=120):  # 10 years of monthly data
    """Generate sample return data for testing if fetching real data fails"""
    print("Generating synthetic data as fallback (10 years of monthly data)...")
    
    # Create ETF names and asset classes
    etfs = {
        'SPY': 'US Equities', 'VGK': 'European Equities', 'EEM': 'Emerging Markets', 'VPL': 'Asia-Pacific',
        'AGG': 'US Aggregate Bonds', 'TLT': 'Long-Term Treasury', 'LQD': 'Corporate Bonds', 'BNDX': 'International Bonds',
        'GLD': 'Gold', 'SLV': 'Silver', 'VNQ': 'Real Estate', 'GSG': 'Commodities'
    }
    
    tickers = list(etfs.keys())[:n_assets]
    asset_class_mapping = pd.Series({ticker: etfs[ticker] for ticker in tickers})
    
    # Create random expected returns by asset class
    expected_monthly_returns = {
        'US Equities': 0.007, 'European Equities': 0.006, 'Emerging Markets': 0.008, 'Asia-Pacific': 0.006,
        'US Aggregate Bonds': 0.002, 'Long-Term Treasury': 0.0015, 'Corporate Bonds': 0.0025, 'International Bonds': 0.001,
        'Gold': 0.003, 'Silver': 0.004, 'Real Estate': 0.005, 'Commodities': 0.002
    }
    
    # Create random volatilities by asset class
    volatilities = {
        'US Equities': 0.04, 'European Equities': 0.05, 'Emerging Markets': 0.06, 'Asia-Pacific': 0.05,
        'US Aggregate Bonds': 0.01, 'Long-Term Treasury': 0.02, 'Corporate Bonds': 0.015, 'International Bonds': 0.012,
        'Gold': 0.05, 'Silver': 0.08, 'Real Estate': 0.05, 'Commodities': 0.07
    }
    
    # Generate synthetic data - details omitted for brevity
    # See full implementation in previous code
    
    # Placeholder implementation - simplified for brevity
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='M')
    returns_data = pd.DataFrame(np.random.normal(0.005, 0.03, (n_periods, n_assets)), 
                               index=dates, columns=tickers)
    prices_data = (1 + returns_data).cumprod() * 100
    
    print(f"Generated synthetic data for {n_assets} ETFs with {n_periods} periods (10 years)")
    
    return returns_data, prices_data, asset_class_mapping

def main():
    # Create the streamlined directory structure
    dirs = create_streamlined_structure('portfolio_results')
    
    try:
        # First try to fetch real ETF data (10 years)
        print("Fetching ETF data (10-year period)...")
        returns_df, prices_df, asset_class_mapping = fetch_etf_data()
    except Exception as e:
        # If that fails, use synthetic data
        print(f"Error fetching real data: {e}")
        returns_df, prices_df, asset_class_mapping = generate_sample_data(n_periods=120)  # 10 years of monthly data
    
    # Create a portfolio builder
    builder = PortfolioBuilder(returns_df)
    
    # Print asset statistics
    annual_returns = returns_df.mean() * 12  # Annualize monthly returns
    annual_volatility = returns_df.std() * np.sqrt(12)  # Annualize volatility
    
    print("\nAsset Statistics (10-Year Performance):")
    stats_df = pd.DataFrame({
        'Asset_Class': asset_class_mapping.values,
        'Annual_Return': annual_returns.values,
        'Annual_Volatility': annual_volatility.values,
        'Sharpe_Ratio': annual_returns.values / annual_volatility.values
    }, index=annual_returns.index)
    
    print(stats_df)
    
    # Save asset statistics to 1_input_data folder
    stats_file = os.path.join(dirs['1_input_data'], "asset_statistics.csv")
    stats_df.to_csv(stats_file)
    print(f"Asset statistics saved to {stats_file}")
    
    # Step 1: Basic Mean-Variance Optimization
    print("\nStep 1: Basic Mean-Variance Optimization")
    mv_model = builder.create_mean_variance_model(risk_aversion=2.0, model_name="etf_basic_mv")
    mv_model.build_model()
    mv_solution = mv_model.solve()
    
    print(f"Expected Return: {mv_solution['expected_return']:.4f}")
    print(f"Volatility: {mv_solution['portfolio_volatility']:.4f}")
    print(f"Sharpe Ratio: {mv_solution['sharpe_ratio']:.4f}")
    
    # Get weights and show them with asset classes
    weights = mv_model.get_weights()
    weights_with_class = pd.DataFrame({
        'Weight': weights,
        'Asset_Class': asset_class_mapping.reindex(weights.index).values
    })
    
    print("\nPortfolio Allocation:")
    print(weights_with_class[weights_with_class['Weight'] > 0.01].sort_values('Weight', ascending=False))
    
    # Save basic MV weights to 2_portfolio_models folder
    weights_file = os.path.join(dirs['2_portfolio_models'], "etf_basic_mv_weights.csv")
    weights_with_class.to_csv(weights_file)
    print(f"Basic MV weights saved to {weights_file}")
    
    # Plot the portfolio weights
    fig1 = builder.plot_portfolio_weights("etf_basic_mv")
    weights_plot_file = os.path.join(dirs['charts/portfolio_weights'], "etf_basic_mv_weights.png")
    plt.savefig(weights_plot_file)
    plt.close()
    print(f"Basic MV weights plot saved to {weights_plot_file}")
    
    # Step 2: Add Asset Class Constraints
    print("\nStep 2: Add Asset Class Constraints")
    
    # Define asset class groups
    asset_classes = {
        'Equities': [t for t, c in asset_class_mapping.items() if 'Equities' in c],
        'Fixed Income': [t for t, c in asset_class_mapping.items() if any(x in c for x in ['Bond', 'Treasury'])],
        'Alternatives': [t for t, c in asset_class_mapping.items() if any(x in c for x in ['Gold', 'Silver', 'Real Estate', 'Commodities'])]
    }
    
    # Create constraints dictionary
    constraints = {
        'group_constraints': {
            'groups': asset_classes,
            'min_allocation': {
                'Equities': 0.30,       # At least 30% in equities
                'Fixed Income': 0.20,    # At least 20% in bonds
                'Alternatives': 0.10     # At least 10% in alternatives
            },
            'max_allocation': {
                'Equities': 0.60,        # At most 60% in equities
                'Fixed Income': 0.50,    # At most 50% in bonds
                'Alternatives': 0.30     # At most 30% in alternatives
            }
        },
        'asset_bounds': {
            'min_weights': {},
            'max_weights': {etf: 0.20 for etf in returns_df.columns}  # No single ETF > 20%
        }
    }
    
    # Build and solve constrained model
    constrained_model = builder.build_and_solve_model(
        model_type="constrained",
        risk_aversion=2.0,
        constraints=constraints,
        model_name="etf_constrained_mv"
    )
    
    # Print results
    constrained_solution = constrained_model.solution
    print(f"Expected Return: {constrained_solution['expected_return']:.4f}")
    print(f"Volatility: {constrained_solution['portfolio_volatility']:.4f}")
    print(f"Sharpe Ratio: {constrained_solution['sharpe_ratio']:.4f}")
    
    # Get weights and show them with asset classes
    constrained_weights = constrained_model.get_weights()
    constrained_weights_with_class = pd.DataFrame({
        'Weight': constrained_weights,
        'Asset_Class': asset_class_mapping.reindex(constrained_weights.index).values
    })
    
    print("\nConstrained Portfolio Allocation:")
    print(constrained_weights_with_class[constrained_weights_with_class['Weight'] > 0.01].sort_values('Weight', ascending=False))
    
    # Save constrained MV weights to 2_portfolio_models folder
    constrained_weights_file = os.path.join(dirs['2_portfolio_models'], "etf_constrained_mv_weights.csv")
    constrained_weights_with_class.to_csv(constrained_weights_file)
    print(f"Constrained MV weights saved to {constrained_weights_file}")
    
    # Plot the portfolio weights
    fig2 = builder.plot_portfolio_weights("etf_constrained_mv")
    constrained_weights_plot_file = os.path.join(dirs['charts/portfolio_weights'], "etf_constrained_mv_weights.png")
    plt.savefig(constrained_weights_plot_file)
    plt.close()
    print(f"Constrained MV weights plot saved to {constrained_weights_plot_file}")
    
    # Step 3: Add Integer Constraints for a Simplified Portfolio
    print("\nStep 3: Add Integer Constraints for a Simplified Portfolio")
    
    # Add integer constraints
    int_constraints = constraints.copy()
    int_constraints.update({
        'max_assets': 6,  # Limit to 6 ETFs for simplicity
        'min_position_size': 0.05  # Minimum 5% allocation if included
    })
    
    # Build and solve integer model
    integer_model = builder.build_and_solve_model(
        model_type="integer",
        risk_aversion=2.0,
        constraints=int_constraints,
        model_name="etf_integer_mv"
    )
    
    # Print results
    integer_solution = integer_model.solution
    print(f"Expected Return: {integer_solution['expected_return']:.4f}")
    print(f"Volatility: {integer_solution['portfolio_volatility']:.4f}")
    print(f"Sharpe Ratio: {integer_solution['sharpe_ratio']:.4f}")
    print(f"Number of assets: {integer_solution['num_assets_selected']}")
    print(f"Selected assets: {integer_solution['selected_asset_names']}")
    
    # Get weights and show them with asset classes
    integer_weights = integer_model.get_weights()
    integer_weights_with_class = pd.DataFrame({
        'Weight': integer_weights,
        'Asset_Class': asset_class_mapping.reindex(integer_weights.index).values
    })
    
    print("\nInteger-Constrained Portfolio Allocation (Simplified 6-ETF Portfolio):")
    print(integer_weights_with_class[integer_weights_with_class['Weight'] > 0.01].sort_values('Weight', ascending=False))
    
    # Save integer MV weights to 2_portfolio_models folder
    integer_weights_file = os.path.join(dirs['2_portfolio_models'], "etf_integer_mv_weights.csv")
    integer_weights_with_class.to_csv(integer_weights_file)
    print(f"Integer MV weights saved to {integer_weights_file}")
    
    # Plot the weights
    fig3 = builder.plot_portfolio_weights("etf_integer_mv")
    integer_weights_plot_file = os.path.join(dirs['charts/portfolio_weights'], "etf_integer_mv_weights.png")
    plt.savefig(integer_weights_plot_file)
    plt.close()
    print(f"Integer MV weights plot saved to {integer_weights_plot_file}")
    
    # Step 4: Compare Models
    print("\nStep 4: Compare Models")
    comparison = builder.compare_models()
    print(comparison)
    
    # Save model comparison to 3_performance_analysis folder
    comparison_file = os.path.join(dirs['3_performance_analysis'], "model_comparison.csv")
    comparison.to_csv(comparison_file)
    print(f"Model comparison saved to {comparison_file}")
    
    # Create model comparison chart
    plt.figure(figsize=(12, 8))
    x = np.arange(len(comparison))
    width = 0.3
    
    plt.bar(x - width, comparison['expected_return'], width, label='Expected Return')
    plt.bar(x, comparison['portfolio_volatility'], width, label='Volatility')
    plt.bar(x + width, comparison['sharpe_ratio'], width, label='Sharpe Ratio')
    
    plt.xlabel('Portfolio Model')
    plt.ylabel('Value')
    plt.title('Model Comparison')
    plt.xticks(x, comparison['model_name'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    model_comparison_chart = os.path.join(dirs['charts/comparison'], "model_comparison.png")
    plt.savefig(model_comparison_chart)
    plt.close()
    print(f"Model comparison chart saved to {model_comparison_chart}")
    
    # Step 5: Plot Efficient Frontier
    print("\nStep 5: Plot Efficient Frontier")
    fig4 = builder.plot_efficient_frontier()
    plt.title("Efficient Frontier (Based on 10 Years of Data)")
    ef_plot_file = os.path.join(dirs['charts/efficient_frontier'], "efficient_frontier.png")
    plt.savefig(ef_plot_file)
    plt.close()
    print(f"Efficient frontier plot saved to {ef_plot_file}")
    
    # Step 6: Analyze Market Regimes
    print("\nStep 6: Analyze Performance Across Market Regimes")
    
    # Split the data into two 5-year periods to analyze different market regimes
    if len(returns_df) >= 60:  # Make sure we have at least 5 years of data
        first_period = returns_df.iloc[:60]  # First 5 years
        second_period = returns_df.iloc[60:]  # Second 5 years
        
        print(f"First period: {first_period.index[0]} to {first_period.index[-1]} ({len(first_period)} months)")
        print(f"Second period: {second_period.index[0]} to {second_period.index[-1]} ({len(second_period)} months)")
        
        # Calculate annualized returns for each period
        first_period_returns = first_period.mean() * 12
        second_period_returns = second_period.mean() * 12
        
        # Compare asset performance across periods
        performance_comparison = pd.DataFrame({
            'Asset_Class': asset_class_mapping.values,
            'First_5yr_Return': first_period_returns.values,
            'Second_5yr_Return': second_period_returns.values if len(second_period) > 0 else np.nan,
            'Return_Difference': (second_period_returns - first_period_returns).values if len(second_period) > 0 else np.nan
        }, index=first_period_returns.index)
        
        print("\nPerformance Across Market Regimes:")
        print(performance_comparison)
        
        # Save market regime analysis to 5_market_analysis folder
        regimes_file = os.path.join(dirs['5_market_analysis'], "market_regimes_analysis.csv")
        performance_comparison.to_csv(regimes_file)
        print(f"Market regimes analysis saved to {regimes_file}")
        
        # Calculate portfolio performance across periods
        if len(second_period) > 0:
            print("\nPortfolio Performance Across Regimes:")
            periods_performance = []
            
            for model_name, weights in [
                ("Unconstrained", weights),
                ("Asset-Class Constrained", constrained_weights),
                ("Simplified 6-ETF", integer_weights)
            ]:
                first_period_return = (weights * first_period_returns).sum()
                second_period_return = (weights * second_period_returns).sum()
                
                print(f"{model_name} Portfolio:")
                print(f"  First 5-Year Period Return: {first_period_return:.4f} ({first_period_return*100:.2f}%)")
                print(f"  Second 5-Year Period Return: {second_period_return:.4f} ({second_period_return*100:.2f}%)")
                print(f"  Difference: {second_period_return-first_period_return:.4f} ({(second_period_return-first_period_return)*100:.2f}%)")
                
                periods_performance.append({
                    'Model': model_name,
                    'First_Period_Return': first_period_return,
                    'Second_Period_Return': second_period_return,
                    'Return_Difference': second_period_return - first_period_return
                })
                
            # Save periods performance comparison to 5_market_analysis folder
            periods_file = os.path.join(dirs['5_market_analysis'], "portfolio_periods_performance.csv")
            pd.DataFrame(periods_performance).to_csv(periods_file)
            print(f"Portfolio periods performance saved to {periods_file}")
                
    # Step 7: Backtest Portfolio over the 10-year period
    print("\nStep 7: 10-Year Portfolio Backtest")

    # Use all historical data for backtesting
    test_returns = returns_df
    
    # Add a benchmark (e.g., 60/40 portfolio)
    benchmark_weights = pd.Series({
        returns_df.columns[0]: 0.60,  # 60% in first ETF (proxy for equities)
        returns_df.columns[4]: 0.40   # 40% in fifth ETF (proxy for bonds)
    })
    test_returns['Benchmark_60_40'] = test_returns[benchmark_weights.index].dot(benchmark_weights)

    # Run backtest
    backtest_results = builder.backtest_portfolio(
        model_name="etf_integer_mv",
        test_data=test_returns,
        benchmark='Benchmark_60_40'
    )
    
    # Print key performance metrics
    print(f"10-Year Annual Return: {backtest_results['annual_return']:.4f} ({backtest_results['annual_return']*100:.2f}%)")
    print(f"10-Year Annual Volatility: {backtest_results['annual_volatility']:.4f} ({backtest_results['annual_volatility']*100:.2f}%)")
    print(f"10-Year Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown: {backtest_results['max_drawdown']:.4f} ({backtest_results['max_drawdown']*100:.2f}%)")
    print(f"Information Ratio vs. 60/40: {backtest_results['information_ratio']:.4f}")
    
    # Save backtest metrics to 4_backtest_results folder
    backtest_metrics = {
        'Annual_Return': backtest_results['annual_return'],
        'Annual_Volatility': backtest_results['annual_volatility'],
        'Sharpe_Ratio': backtest_results['sharpe_ratio'],
        'Max_Drawdown': backtest_results['max_drawdown'],
        'Information_Ratio': backtest_results['information_ratio'],
        'Benchmark_Return': backtest_results.get('benchmark_annual_return'),
        'Benchmark_Volatility': backtest_results.get('benchmark_annual_volatility'),
        'Benchmark_Sharpe': backtest_results.get('benchmark_sharpe_ratio')
    }
    
    backtest_metrics_file = os.path.join(dirs['4_backtest_results'], "backtest_metrics.csv")
    pd.DataFrame([backtest_metrics]).to_csv(backtest_metrics_file)
    print(f"Backtest metrics saved to {backtest_metrics_file}")
    
    # Save backtest returns to 4_backtest_results folder
    backtest_returns_file = os.path.join(dirs['4_backtest_results'], "backtest_returns.csv")
    pd.DataFrame({
        'Portfolio_Returns': backtest_results['portfolio_returns'],
        'Benchmark_Returns': backtest_results.get('benchmark_returns'),
        'Portfolio_Cumulative': backtest_results['cumulative_returns'],
        'Benchmark_Cumulative': backtest_results.get('benchmark_cumulative')
    }).to_csv(backtest_returns_file)
    print(f"Backtest returns saved to {backtest_returns_file}")
    
    # Plot backtest results
    fig5 = builder.plot_backtest_results(backtest_results)
    plt.title("10-Year Backtest Performance")
    backtest_plot_file = os.path.join(dirs['charts/backtest'], "backtest_performance.png")
    plt.savefig(backtest_plot_file)
    plt.close()
    print(f"Backtest results plot saved to {backtest_plot_file}")
    
    # Final report document with links to all resources
    report_content = """# Portfolio Optimization Summary

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
- The optimized portfolio achieved an annual return of {annual_return:.2%} with {annual_volatility:.2%} volatility
- The portfolio outperformed the 60/40 benchmark by {information_ratio:.2f} units of risk (Information Ratio)
- Maximum drawdown during the 10-year period was {max_drawdown:.2%}

""".format(
        annual_return=backtest_results['annual_return'],
        annual_volatility=backtest_results['annual_volatility'],
        information_ratio=backtest_results['information_ratio'],
        max_drawdown=backtest_results['max_drawdown']
    )
    
    # Save report to reports directory
    report_file = os.path.join(dirs['reports'], "portfolio_optimization_summary.md")
    with open(report_file, 'w') as f:
        f.write(report_content)
    print(f"Summary report saved to {report_file}")
    
    print("\nDone! All results and charts have been saved in a clean, organized structure.")

if __name__ == "__main__":
    main()