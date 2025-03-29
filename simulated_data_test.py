# example_usage.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_builder import PortfolioBuilder

# Set seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_assets=10, n_periods=252, annual_returns=None, volatilities=None):
    """Generate sample return data for testing"""
    # Default annual returns and volatilities if not provided
    if annual_returns is None:
        annual_returns = np.random.normal(0.08, 0.05, n_assets)
    if volatilities is None:
        volatilities = np.random.uniform(0.1, 0.3, n_assets)
    
    # Create properly formatted asset names
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
    
    # Generate random correlation matrix that is guaranteed to be positive-semidefinite
    # Use the Cholesky method to ensure positive-definiteness
    random_matrix = np.random.randn(n_assets, n_assets)
    correlation = np.dot(random_matrix, random_matrix.T)
    # Normalize to get correlation matrix
    d = np.sqrt(np.diag(correlation))
    correlation = correlation / np.outer(d, d)
    
    # Compute covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    
    # Add a small value to diagonal to ensure numerical stability
    cov_matrix += np.eye(n_assets) * 1e-6
    
    # Generate returns using a more stable approach
    daily_returns = np.random.multivariate_normal(
        annual_returns / 252,  # Daily expected returns
        cov_matrix / 252,      # Daily covariance
        n_periods,
        check_valid='warn'     # Only warn, don't raise error
    )
    
    # Create DataFrame
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='B')
    returns_df = pd.DataFrame(daily_returns, index=dates, columns=asset_names)
    
    return returns_df, annual_returns, cov_matrix, asset_names
def main():
    """Main function to demonstrate portfolio optimization"""
    print("Generating sample data...")
    returns_df, expected_returns, cov_matrix, asset_names = generate_sample_data()
    
    # Create a portfolio builder
    builder = PortfolioBuilder(returns_df)
    
    # Step 1: Basic Mean-Variance Optimization
    print("\nStep 1: Basic Mean-Variance Optimization")
    mv_model = builder.create_mean_variance_model(risk_aversion=2.0, model_name="basic_mv")
    mv_model.build_model()
    mv_solution = mv_model.solve()
    
    print(f"Expected Return: {mv_solution['expected_return']:.4f}")
    print(f"Volatility: {mv_solution['portfolio_volatility']:.4f}")
    print(f"Sharpe Ratio: {mv_solution['sharpe_ratio']:.4f}")
    
    # Plot the portfolio weights
    fig1 = builder.plot_portfolio_weights("basic_mv")
    plt.savefig("basic_mv_weights.png")
    
    # Step 2: Add Linear Constraints
    print("\nStep 2: Add Linear Constraints")
    
    # Define sector mapping
    sector_mapping = {
        'Technology': ['Asset_1', 'Asset_2', 'Asset_3'],
        'Finance': ['Asset_4', 'Asset_5'],
        'Energy': ['Asset_6', 'Asset_7'],
        'Healthcare': ['Asset_8', 'Asset_9', 'Asset_10']
    }
    
    # Create constraints dictionary
    constraints = {
        'sector_constraints': {
            'mapping': {asset: sector for sector, assets in sector_mapping.items() for asset in assets},
            'min_allocation': {'Technology': 0.2, 'Finance': 0.1},
            'max_allocation': {'Technology': 0.5, 'Energy': 0.3}
        },
        'asset_bounds': {
            'min_weights': {'Asset_1': 0.05},
            'max_weights': {'Asset_5': 0.15, 'Asset_8': 0.2}
        }
    }
    
    # Build and solve constrained model
    constrained_model = builder.build_and_solve_model(
        model_type="constrained",
        risk_aversion=2.0,
        constraints=constraints,
        model_name="constrained_mv"
    )
    
    # Print results
    constrained_solution = constrained_model.solution
    print(f"Expected Return: {constrained_solution['expected_return']:.4f}")
    print(f"Volatility: {constrained_solution['portfolio_volatility']:.4f}")
    print(f"Sharpe Ratio: {constrained_solution['sharpe_ratio']:.4f}")
    
    # Plot the portfolio weights
    fig2 = builder.plot_portfolio_weights("constrained_mv")
    plt.savefig("constrained_mv_weights.png")
    
    # Step 3: Add Integer Constraints
    print("\nStep 3: Add Integer Constraints")
    
    # Add integer constraints
    int_constraints = constraints.copy()
    int_constraints.update({
        'max_assets': 5,
        'min_position_size': 0.05
    })
    
    # Build and solve integer model
    integer_model = builder.build_and_solve_model(
        model_type="integer",
        risk_aversion=2.0,
        constraints=int_constraints,
        model_name="integer_mv"
    )
    
    # Print results
    integer_solution = integer_model.solution
    print(f"Expected Return: {integer_solution['expected_return']:.4f}")
    print(f"Volatility: {integer_solution['portfolio_volatility']:.4f}")
    print(f"Sharpe Ratio: {integer_solution['sharpe_ratio']:.4f}")
    print(f"Number of assets: {integer_solution['num_assets_selected']}")
    print(f"Selected assets: {integer_solution['selected_asset_names']}")
    
    # Plot the portfolio weights
    fig3 = builder.plot_portfolio_weights("integer_mv")
    plt.savefig("integer_mv_weights.png")
    
    # Step 4: Compare Models
    print("\nStep 4: Compare Models")
    comparison = builder.compare_models()
    print(comparison)
    
    # Step 5: Plot Efficient Frontier
    print("\nStep 5: Plot Efficient Frontier")
    fig4 = builder.plot_efficient_frontier()
    plt.savefig("efficient_frontier.png")
    
    # Step 6: Backtest Portfolio
    print("\nStep 6: Backtest Portfolio")

    # Generate test data using SAME asset names
    test_df, _, _, _ = generate_sample_data(
        n_assets=10,  # Same number of assets
        n_periods=126  # 6 months
    )

    # Ensure test data has the same columns as the original data
    # This is important: rename test columns to match original asset names
    test_df.columns = builder.asset_names[:10]  

    # Add a benchmark (e.g., equal-weighted portfolio)
    test_df['Benchmark'] = test_df.mean(axis=1)

    # Run backtest
    backtest_results = builder.backtest_portfolio(
        model_name="integer_mv",
        test_data=test_df,
        benchmark='Benchmark'
    )
    
    # Print key performance metrics
    print(f"Annual Return: {backtest_results['annual_return']:.4f}")
    print(f"Annual Volatility: {backtest_results['annual_volatility']:.4f}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']:.4f}")
    print(f"Information Ratio: {backtest_results['information_ratio']:.4f}")
    
    # Plot backtest results
    fig5 = builder.plot_backtest_results(backtest_results)
    plt.savefig("backtest_results.png")
    
    # Step 7: Generate Report
    print("\nStep 7: Generate Report")
    builder.generate_report(output_file="portfolio_optimization_report.md")
    
    print("\nDone! All results and plots have been saved.")

if __name__ == "__main__":
    main()