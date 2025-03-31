import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from gurobipy import Model, GRB, quicksum

# Create visualisations directory if it doesn't exist
os.makedirs('visualisations', exist_ok=True)

# Define ETF descriptions for better labeling
ETF_DESCRIPTIONS = {
    'SPY': 'US Equities (S&P 500)',
    'QQQ': 'US Tech',
    'VWRA.L': 'All World Equities',
    'VGK': 'European Equities',
    'EEM': 'Emerging Markets',
    'VPL': 'Asia-Pacific Equities',
    'AGG': 'US Aggregate Bonds',
    'TLT': 'Long-Term Treasury Bonds',
    'LQD': 'Corporate Bonds',
    'GLD': 'Gold',
    'SLV': 'Silver',
    'VNQ': 'Real Estate',
    'GSG': 'Commodities'
}

# 1. Load Financial Data
def fetch_stock_data(tickers, start_date, end_date):
    """Fetch stock data using yfinance and save to CSV"""
    print(f"Fetching data for {len(tickers)} assets from {start_date} to {end_date}...")
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Save the data to a CSV file
    csv_filename = 'stock_data.csv'
    stock_data.to_csv(csv_filename)
    print(f"Stock price data saved to {csv_filename}")
    
    return stock_data

# 2. Calculate Returns and Covariance
def calculate_returns_and_covariance(stock_data):
    """Calculate daily returns, annual returns and covariance matrix"""
    # Calculate daily returns
    daily_returns = stock_data.pct_change().dropna()
    
    # Calculate average daily and annual returns (252 trading days)
    avg_daily_return = daily_returns.mean()
    avg_annual_return = avg_daily_return * 252
    
    # Calculate covariance matrix (annualized)
    cov_matrix = daily_returns.cov() * 252
    
    return daily_returns, avg_annual_return, cov_matrix

# 3. Portfolio Optimization for a single target return
def optimize_for_target_return(returns, cov_matrix, target_return, risk_free_rate=0.03):
    """
    Optimize portfolio weights to minimize risk for a specified target return
    
    Parameters:
    - returns: Expected annual returns for each asset
    - cov_matrix: Covariance matrix of returns
    - target_return: Target return level
    - risk_free_rate: Annual risk-free rate (default: 3%)
    
    Returns:
    - Dictionary of asset weights, return, risk, and sharpe ratio
    """
    n = len(returns)  # Number of assets
    assets = returns.index.tolist()
    
    print(f"\n=== Optimizing for target return: {target_return:.2%} ===")
    
    # Create optimization model
    m = Model("Portfolio_Optimization")
    m.setParam('OutputFlag', 0)  # Suppress Gurobi output
    
    # Add variables (portfolio weights)
    x = m.addVars(n, lb=0, name="weights")
    
    # Constraint: Weights sum to 1 (fully invested)
    m.addConstr(quicksum(x[i] for i in range(n)) == 1, "budget")
    
    # Constraint: Achieve target return
    m.addConstr(quicksum(returns.iloc[i] * x[i] for i in range(n)) >= target_return, "return")
    
    # Objective: Minimize portfolio variance
    portfolio_variance = quicksum(x[i] * x[j] * cov_matrix.iloc[i, j] 
                                for i in range(n) for j in range(n))
    m.setObjective(portfolio_variance, GRB.MINIMIZE)
    
    # Optimize the model
    try:
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            # Extract the optimal weights
            optimal_weights = pd.Series({assets[i]: x[i].X for i in range(n)})
            
            # Calculate portfolio performance metrics
            portfolio_return = sum(optimal_weights[asset] * returns[asset] for asset in assets)
            portfolio_risk = np.sqrt(sum(optimal_weights[i] * optimal_weights[j] * cov_matrix.loc[i, j]
                                    for i in assets for j in assets))
            
            # Calculate Sharpe ratio using risk-free rate
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            # Print detailed results
            print(f"Portfolio with target return {target_return:.2%}:")
            print(f"  Actual Return: {portfolio_return:.4%}")
            print(f"  Risk (Std Dev): {portfolio_risk:.4%}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.4f} (using risk-free rate: {risk_free_rate:.2%})")
            print("\n  Asset Allocation:")
            for asset, weight in sorted(optimal_weights.items(), key=lambda x: -x[1]):
                if weight > 0.001:  # Only show allocations > 0.1%
                    asset_desc = ETF_DESCRIPTIONS.get(asset, asset)
                    print(f"    {asset_desc}: {weight:.4%}")
            
            return {
                'weights': optimal_weights,
                'return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe': sharpe_ratio
            }
        else:
            print(f"Optimization failed with status {m.status}")
            return None
    except Exception as e:
        print(f"Optimization error: {e}")
        return None

# 4. Generate Efficient Frontier
def generate_efficient_frontier(returns, cov_matrix, risk_free_rate=0.03, points=20):
    """Generate the efficient frontier by solving for different target returns"""
    print("\n=== Generating Efficient Frontier ===")
    min_return = min(returns)
    max_return = max(returns)
    print(f"Return range: {min_return:.4%} to {max_return:.4%}")
    
    target_returns = np.linspace(min_return, max_return, points)
    
    efficient_portfolios = []
    all_sharpe_ratios = []
    
    print("\nCalculating portfolios along the efficient frontier:")
    for i, target in enumerate(target_returns):
        print(f"\nPoint {i+1}/{points} - Target Return: {target:.4%}")
        result = optimize_for_target_return(returns, cov_matrix, target, risk_free_rate)
        if result:
            efficient_portfolios.append(result)
            all_sharpe_ratios.append(result['sharpe'])
            print(f"  Achieved - Return: {result['return']:.4%}, Risk: {result['risk']:.4%}, Sharpe: {result['sharpe']:.4f}")
    
    return efficient_portfolios, all_sharpe_ratios

# 5. Find Maximum Sharpe Ratio Portfolio
def find_max_sharpe_portfolio(efficient_portfolios, all_sharpe_ratios, risk_free_rate=0.03):
    """Find the portfolio with the maximum Sharpe ratio"""
    print("\n=== Finding Maximum Sharpe Ratio Portfolio ===")
    max_sharpe_idx = np.argmax(all_sharpe_ratios)
    max_sharpe = all_sharpe_ratios[max_sharpe_idx]
    optimal_portfolio = efficient_portfolios[max_sharpe_idx]
    
    print(f"Maximum Sharpe Ratio: {max_sharpe:.4f} (using risk-free rate: {risk_free_rate:.2%})")
    print(f"At point {max_sharpe_idx+1} of the efficient frontier")
    print(f"Portfolio Return: {optimal_portfolio['return']:.4%}")
    print(f"Portfolio Risk: {optimal_portfolio['risk']:.4%}")
    print(f"Excess Return: {(optimal_portfolio['return'] - risk_free_rate):.4%}")
    
    print("\nOptimal Portfolio Weights:")
    for asset, weight in sorted(optimal_portfolio['weights'].items(), key=lambda x: -x[1]):
        if weight > 0.005:  # Only show allocations > 0.5%
            asset_desc = ETF_DESCRIPTIONS.get(asset, asset)
            print(f"  {asset_desc}: {weight:.4%}")
    
    return optimal_portfolio

# 6. Calculate Portfolio Statistics for a given set of weights
def calculate_portfolio_stats(weights, returns, cov_matrix, risk_free_rate=0.03):
    """Calculate return, risk, and Sharpe ratio for a given portfolio"""
    portfolio_return = sum(weights[asset] * returns[asset] for asset in weights.index)
    
    portfolio_risk = np.sqrt(
        sum(weights[i] * weights[j] * cov_matrix.loc[i, j]
            for i in weights.index for j in weights.index)
    )
    
    # Calculate Sharpe ratio using risk-free rate
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
    
    return portfolio_return, portfolio_risk, sharpe_ratio

# New visualization functions
def plot_returns_vs_risk(annual_returns, annual_risks):
    """Plot returns versus risk for individual assets"""
    plt.figure(figsize=(12, 8))
    
    # Convert to DataFrames for easier plotting
    returns_df = pd.Series(annual_returns)
    risks_df = pd.Series({ticker: annual_risks[i] for i, ticker in enumerate(returns_df.index)})
    
    # Create the scatter plot
    plt.scatter(risks_df, returns_df, s=100, alpha=0.7)
    
    # Add labels for each point
    for ticker in returns_df.index:
        plt.annotate(ETF_DESCRIPTIONS.get(ticker, ticker), 
                    (risks_df[ticker], returns_df[ticker]),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.xlabel('Annual Risk (Standard Deviation)')
    plt.ylabel('Annual Return')
    plt.title('Risk-Return Profile of Individual Assets')
    plt.grid(True)
    
    # Save the figure
    plt.savefig('visualisations/returns_vs_risk.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Returns vs Risk plot saved to 'visualisations/returns_vs_risk.png'")

def plot_correlation_matrix(corr_matrix):
    """Plot the correlation matrix as a heatmap with asset descriptions"""
    plt.figure(figsize=(14, 12))
    
    # Create a copy with descriptive labels
    corr_matrix_labeled = corr_matrix.copy()
    
    # Create mapping dictionaries for rows and columns
    new_labels = {ticker: ETF_DESCRIPTIONS.get(ticker, ticker) for ticker in corr_matrix.index}
    
    # Rename both index and columns with descriptions
    corr_matrix_labeled = corr_matrix_labeled.rename(index=new_labels, columns=new_labels)
    
    # Create the heatmap with larger font for the descriptions
    sns.heatmap(corr_matrix_labeled, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, fmt='.2f', cbar_kws={'label': 'Correlation'},
                annot_kws={"size": 8})
    
    plt.title('Asset Correlation Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('visualisations/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Correlation matrix plot saved to 'visualisations/correlation_matrix.png'")

    # Also save a version with ticker symbols for reference
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, fmt='.2f', cbar_kws={'label': 'Correlation'})
    plt.title('Asset Correlation Matrix (Ticker Symbols)')
    plt.tight_layout()
    plt.savefig('visualisations/correlation_matrix_tickers.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_efficient_frontier(efficient_portfolios, annual_returns, annual_risks, 
                           optimal_portfolio, moderate_portfolio, equal_return, equal_risk, risk_free_rate=0.03):
    """Plot the efficient frontier with key portfolios and the Capital Market Line"""
    # Extract risk and return values from efficient portfolios
    risks = [p['risk'] for p in efficient_portfolios]
    returns = [p['return'] for p in efficient_portfolios]
    
    plt.figure(figsize=(12, 8))
    
    # Plot efficient frontier
    plt.plot(risks, returns, 'b-', linewidth=2, label='Efficient Frontier')
    
    # Convert annual_risks to array for plotting individual assets
    risks_array = np.array([annual_risks[i] for i in range(len(annual_returns))])
    
    # Plot individual assets
    plt.scatter(risks_array, annual_returns, c='red', marker='o', s=100, label='Individual Assets')
    
    # Plot risk-free rate point
    plt.scatter(0, risk_free_rate, c='black', marker='o', s=100, label=f'Risk-Free Rate ({risk_free_rate:.2%})')
    
    # Plot Capital Market Line (CML)
    # The CML connects the risk-free rate to the tangency portfolio (max Sharpe)
    x_values = np.linspace(0, max(risks_array) * 1.2, 100)
    slope = (optimal_portfolio['return'] - risk_free_rate) / optimal_portfolio['risk']  # This is the Sharpe ratio
    y_values = risk_free_rate + slope * x_values
    plt.plot(x_values, y_values, 'g--', linewidth=2, label='Capital Market Line')
    
    # Add asset labels with descriptions
    for i, ticker in enumerate(annual_returns.index):
        plt.annotate(ETF_DESCRIPTIONS.get(ticker, ticker), 
                    (risks_array[i], annual_returns.iloc[i]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    # Plot optimal (max Sharpe) portfolio
    plt.scatter(optimal_portfolio['risk'], optimal_portfolio['return'], 
               c='gold', marker='*', s=200, label='Maximum Sharpe Ratio')
    
    # Plot moderate return portfolio
    plt.scatter(moderate_portfolio['risk'], moderate_portfolio['return'], 
               c='green', marker='*', s=200, label='Target Return')
    
    # Plot equal-weighted portfolio
    plt.scatter(equal_risk, equal_return, 
               c='purple', marker='*', s=200, label='Equal-Weighted')
    
    plt.title('Efficient Frontier with Capital Market Line')
    plt.xlabel('Annual Risk (Standard Deviation)')
    plt.ylabel('Annual Expected Return')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    plt.savefig('visualisations/efficient_frontier.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Efficient frontier plot saved to 'visualisations/efficient_frontier.png'")

def plot_optimal_allocation(weights, title='Optimal Portfolio Allocation'):
    """Plot the optimal portfolio allocation as a pie chart with descriptive labels"""
    # Filter out tiny allocations (less than 1%)
    filtered_weights = weights[weights > 0.01]
    
    # Create a series with descriptive labels
    weights_with_desc = pd.Series({ETF_DESCRIPTIONS.get(asset, asset): value 
                                 for asset, value in filtered_weights.items()})
    
    plt.figure(figsize=(12, 8))
    weights_with_desc.plot(kind='pie', autopct='%1.1f%%')
    plt.title(title)
    plt.ylabel('')  # Hide the ylabel
    
    # Save the figure
    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(f'visualisations/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{title} plot saved to 'visualisations/{filename}'")

def plot_sharpe_ratios(efficient_portfolios, all_sharpe_ratios, risk_free_rate=0.03):
    """Plot Sharpe ratios along the efficient frontier"""
    # Extract returns for x-axis
    returns = [p['return'] for p in efficient_portfolios]
    
    plt.figure(figsize=(12, 8))
    plt.plot(returns, all_sharpe_ratios, 'g-o', linewidth=2)
    
    # Find and mark the maximum Sharpe ratio
    max_sharpe_idx = np.argmax(all_sharpe_ratios)
    max_sharpe = all_sharpe_ratios[max_sharpe_idx]
    max_return = returns[max_sharpe_idx]
    
    plt.scatter(max_return, max_sharpe, c='red', marker='*', s=200)
    plt.annotate(f"Max Sharpe: {max_sharpe:.4f}\nReturn: {max_return:.2%}\nExcess Return: {(max_return - risk_free_rate):.2%}", 
                (max_return, max_sharpe),
                xytext=(10, -30),
                textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    
    plt.title(f'Sharpe Ratios Along the Efficient Frontier (Risk-Free Rate: {risk_free_rate:.2%})')
    plt.xlabel('Portfolio Return')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    
    # Save the figure
    plt.savefig('visualisations/sharpe_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Sharpe ratios plot saved to 'visualisations/sharpe_ratios.png'")

def plot_portfolio_comparison(portfolios, risk_free_rate=0.03):
    """Plot a comparison of portfolio metrics"""
    # Extract data
    names = list(portfolios.keys())
    returns = [portfolios[name]['return'] for name in names]
    risks = [portfolios[name]['risk'] for name in names]
    sharpes = [portfolios[name]['sharpe'] for name in names]
    excess_returns = [portfolios[name]['return'] - risk_free_rate for name in names]
    
    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot returns
    ax1.bar(names, returns, color='skyblue')
    ax1.set_title('Expected Annual Return')
    ax1.set_ylabel('Return')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage labels
    for i, v in enumerate(returns):
        ax1.text(i, v/2, f"{v:.2%}", ha='center', va='center', 
                fontweight='bold', color='darkblue')
    
    # Plot excess returns
    ax2.bar(names, excess_returns, color='lightblue')
    ax2.set_title(f'Excess Return (over Risk-Free Rate: {risk_free_rate:.2%})')
    ax2.set_ylabel('Excess Return')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage labels
    for i, v in enumerate(excess_returns):
        ax2.text(i, v/2, f"{v:.2%}", ha='center', va='center', 
                fontweight='bold', color='darkblue')
    
    # Plot risks
    ax3.bar(names, risks, color='salmon')
    ax3.set_title('Annual Risk (Standard Deviation)')
    ax3.set_ylabel('Risk')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage labels
    for i, v in enumerate(risks):
        ax3.text(i, v/2, f"{v:.2%}", ha='center', va='center', 
                fontweight='bold', color='darkred')
    
    # Plot Sharpe ratios
    ax4.bar(names, sharpes, color='lightgreen')
    ax4.set_title(f'Sharpe Ratio (using Risk-Free Rate: {risk_free_rate:.2%})')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels
    for i, v in enumerate(sharpes):
        ax4.text(i, v/2, f"{v:.4f}", ha='center', va='center', 
                fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('visualisations/portfolio_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Portfolio comparison plot saved to 'visualisations/portfolio_comparison.png'")

def print_asset_descriptions():
    """Print a summary of the assets being analyzed with their descriptions"""
    print("\n=== Asset Class Descriptions ===")
    print(f"{'Ticker':<6} {'Description':<30}")
    print(f"{'-'*6} {'-'*30}")
    
    # Group by asset class for better organization
    asset_classes = {
        'Equities': ['SPY', 'VGK', 'EEM', 'VPL'],
        'Fixed Income': ['AGG', 'TLT', 'LQD'],
        'Alternative Assets': ['GLD', 'SLV', 'VNQ', 'GSG']
    }
    
    for asset_class, tickers in asset_classes.items():
        print(f"\n{asset_class}:")
        for ticker in tickers:
            if ticker in ETF_DESCRIPTIONS:
                print(f"  {ticker:<6} {ETF_DESCRIPTIONS[ticker]:<30}")

# Main function
def main():
    # Define parameters
    tickers = ['SPY', 'QQQ', 'VWRA.L', 'VGK', 'EEM', 'VPL', 'AGG', 'TLT', 'LQD', 'GLD', 'SLV', 'VNQ', 'GSG']
    start_date = '2006-10-03'
    end_date = '2024-12-31'
    
    # Define risk-free rate (annual)
    # Using 5% as a default but this should be adjusted based on current rates
    # For example, you could use the 3-month Treasury Bill rate or 10-year Treasury yield
    risk_free_rate = 0.03  # 5%
    print(f"Using annual risk-free rate: {risk_free_rate:.2%}")
    
    # Print asset descriptions for reference
    print_asset_descriptions()
    
    # Step 1: Fetch data
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    
    # Step 2: Calculate returns and covariance
    print("\n=== Asset Return and Risk Analysis ===")
    daily_returns, annual_returns, cov_matrix = calculate_returns_and_covariance(stock_data)
    
    # Print annual returns
    print("\nAnnual Returns:")
    for ticker, ret in annual_returns.items():
        asset_desc = ETF_DESCRIPTIONS.get(ticker, ticker)
        print(f"  {asset_desc:<30}: {ret:.4%}")
    
    # Print annual standard deviations (risk)
    annual_risks = []
    print("\nAnnual Risk (Standard Deviation):")
    for i, ticker in enumerate(annual_returns.index):
        risk = np.sqrt(cov_matrix.iloc[i, i])
        annual_risks.append(risk)
        asset_desc = ETF_DESCRIPTIONS.get(ticker, ticker)
        print(f"  {asset_desc:<30}: {risk:.4%}")
    
    # Calculate and print Sharpe ratio for individual assets
    print("\nSharpe Ratios (Individual Assets):")
    for i, ticker in enumerate(annual_returns.index):
        excess_return = annual_returns[ticker] - risk_free_rate
        risk = annual_risks[i]
        sharpe = excess_return / risk if risk > 0 else 0
        asset_desc = ETF_DESCRIPTIONS.get(ticker, ticker)
        print(f"  {asset_desc:<30}: {sharpe:.4f}")
    
    # Plot returns vs risk
    plot_returns_vs_risk(annual_returns, annual_risks)
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    corr_matrix = daily_returns.corr()
    print(corr_matrix.round(4))
    
    # Plot correlation matrix
    plot_correlation_matrix(corr_matrix)
    
    # Print covariance matrix
    print("\nCovariance Matrix:")
    print(cov_matrix.round(6))
    
    # Step 3: Optimize for a single target return
    # Choose a moderate target return
    moderate_return = annual_returns.mean()
    moderate_portfolio = optimize_for_target_return(annual_returns, cov_matrix, moderate_return, risk_free_rate)
    
    # Plot the moderate portfolio allocation
    plot_optimal_allocation(moderate_portfolio['weights'], 'Target Return Portfolio Allocation')
    
    # Step 4: Generate efficient frontier
    efficient_portfolios, all_sharpe_ratios = generate_efficient_frontier(annual_returns, cov_matrix, risk_free_rate, points=20)
    
    # Plot Sharpe ratios along the efficient frontier
    plot_sharpe_ratios(efficient_portfolios, all_sharpe_ratios, risk_free_rate)
    
    # Step 5: Find optimal portfolio (maximum Sharpe ratio)
    optimal_portfolio = find_max_sharpe_portfolio(efficient_portfolios, all_sharpe_ratios, risk_free_rate)
    
    # Plot the optimal portfolio allocation
    plot_optimal_allocation(optimal_portfolio['weights'], 'Maximum Sharpe Portfolio Allocation')
    
    # Step 6: Compare with equal-weighted portfolio
    print("\n=== Comparing with Equal-Weighted Portfolio ===")
    equal_weights = pd.Series(1/len(tickers), index=annual_returns.index)
    equal_return, equal_risk, equal_sharpe = calculate_portfolio_stats(equal_weights, annual_returns, cov_matrix, risk_free_rate)
    
    print(f"Equal-Weighted Portfolio:")
    print(f"  Return: {equal_return:.4%}")
    print(f"  Risk: {equal_risk:.4%}")
    print(f"  Excess Return: {(equal_return - risk_free_rate):.4%}")
    print(f"  Sharpe Ratio: {equal_sharpe:.4f}")
    
    # Plot the equal-weighted portfolio allocation
    plot_optimal_allocation(equal_weights, 'Equal-Weighted Portfolio Allocation')
    
    # Plot the efficient frontier
    plot_efficient_frontier(efficient_portfolios, annual_returns, annual_risks, 
                           optimal_portfolio, moderate_portfolio, equal_return, equal_risk, risk_free_rate)
    
    # Print comparison
    print("\n=== Performance Comparison ===")
    print(f"{'Portfolio Type':<20} {'Return':<10} {'Risk':<10} {'Excess Ret.':<12} {'Sharpe':<10}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    print(f"{'Equal-Weighted':<20} {equal_return:.4%} {equal_risk:.4%} {(equal_return - risk_free_rate):.4%} {equal_sharpe:.4f}")
    print(f"{'Target Return':<20} {moderate_portfolio['return']:.4%} {moderate_portfolio['risk']:.4%} {(moderate_portfolio['return'] - risk_free_rate):.4%} {moderate_portfolio['sharpe']:.4f}")
    print(f"{'Maximum Sharpe':<20} {optimal_portfolio['return']:.4%} {optimal_portfolio['risk']:.4%} {(optimal_portfolio['return'] - risk_free_rate):.4%} {optimal_portfolio['sharpe']:.4f}")
    
    # Plot portfolio comparison
    portfolio_data = {
        'Equal-Weighted': {'return': equal_return, 'risk': equal_risk, 'sharpe': equal_sharpe},
        'Target Return': {'return': moderate_portfolio['return'], 'risk': moderate_portfolio['risk'], 'sharpe': moderate_portfolio['sharpe']},
        'Maximum Sharpe': {'return': optimal_portfolio['return'], 'risk': optimal_portfolio['risk'], 'sharpe': optimal_portfolio['sharpe']}
    }
    plot_portfolio_comparison(portfolio_data, risk_free_rate)
    
    print("\nAll visualizations have been saved in the 'visualisations' folder.")
    print(f"\nNote: All Sharpe ratio calculations use a risk-free rate of {risk_free_rate:.2%}")

if __name__ == "__main__":
    main()