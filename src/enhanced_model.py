import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from gurobipy import Model, GRB, quicksum

#######################################################
#           CONFIGURATION PARAMETERS                  #
#######################################################

# Date range for analysis
START_DATE = '2006-10-03'
END_DATE = '2024-12-31'

# Risk-free rate (annual)
RISK_FREE_RATE = 0.03  # 3%

# Enhanced optimization constraints
MAX_ASSETS = 8         # Maximum number of assets in the portfolio
MIN_POSITION = 0.02    # Minimum position size (2%)
EF_POINTS = 20         # Number of points on the efficient frontier

# Maximum allocation per sector
SECTOR_LIMITS = {
    'Equities': 0.70, # Max 70% in equities
    'Fixed Income': 0.60,  # Max 60% in fixed income
    'Alternative': 0.30   # Max 30% in alternatives
}

# Transaction costs (percentage)

# Set APPLY_TRANSACTION_COSTS to False to ignore transaction costs in optimization
APPLY_TRANSACTION_COSTS = True

# Base transaction cost as percentage (e.g., 0.05 = 0.05%)
BASE_TRANSACTION_COST = 0.05

# Transaction costs (percentage)
TRANSACTION_COST_FACTORS = {
    'SPY': 0.05, 'QQQ': 0.05, 'VWRA.L': 0.07, 'VGK': 0.06, 'EEM': 0.08, 
    'VPL': 0.07, 'AGG': 0.04, 'TLT': 0.04, 'LQD': 0.04, 
    'GLD': 0.06, 'SLV': 0.06, 'VNQ': 0.05, 'GSG': 0.06
}

#######################################################
# Original code below - DO NOT MODIFY                 #
#######################################################

# Tickers to include in portfolio optimization
TICKERS = ['SPY', 'QQQ', 'VWRA.L', 'VGK', 'EEM', 'VPL', 'AGG', 'TLT', 'LQD', 'GLD', 'SLV', 'VNQ', 'GSG']

# Define sector classification for ETFs
SECTORS = {
    'Equities': ['SPY', 'QQQ', 'VWRA.L', 'VGK', 'EEM', 'VPL'],  # Added QQQ here
    'Fixed Income': ['AGG', 'TLT', 'LQD'],
    'Alternative': ['GLD', 'SLV', 'VNQ', 'GSG']
}


# Create visualisations_model2 directory if it doesn't exist
os.makedirs('visualisations_model2', exist_ok=True)

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


# Generate transaction costs based on configuration
def get_transaction_costs():
    """Calculate transaction costs for each asset based on configuration"""
    transaction_costs = {}
    for ticker in TICKERS:
        factor = TRANSACTION_COST_FACTORS.get(ticker, 1.0)
        transaction_costs[ticker] = BASE_TRANSACTION_COST * factor
    return transaction_costs

# Get transaction costs
TRANSACTION_COSTS = get_transaction_costs()

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

# 3. Save and load portfolio weights and metrics
def save_portfolio_weights(portfolio_data, filename="optimal_portfolio.json"):
    """Save portfolio weights and metrics to a JSON file"""
    # Convert pandas Series to dictionaries for JSON serialization
    serializable_data = {}
    for portfolio_name, portfolio in portfolio_data.items():
        serializable_data[portfolio_name] = {
            'weights': portfolio['weights'].to_dict() if isinstance(portfolio['weights'], pd.Series) else portfolio['weights'],
            'return': portfolio['return'],
            'risk': portfolio['risk'],
            'sharpe': portfolio['sharpe']
        }
        if 'assets' in portfolio:
            serializable_data[portfolio_name]['assets'] = portfolio['assets']
        if 'transaction_costs' in portfolio:
            serializable_data[portfolio_name]['transaction_costs'] = portfolio['transaction_costs']
        if 'sector_allocations' in portfolio:
            serializable_data[portfolio_name]['sector_allocations'] = portfolio['sector_allocations']
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(serializable_data, f, indent=4)
    
    print(f"Portfolio data saved to {filename}")

def load_portfolio_weights(filename="optimal_portfolio.json"):
    """Load portfolio weights and metrics from a JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert dictionaries back to pandas Series
        portfolio_data = {}
        for portfolio_name, portfolio in data.items():
            portfolio_data[portfolio_name] = {
                'weights': pd.Series(portfolio['weights']),
                'return': portfolio['return'],
                'risk': portfolio['risk'],
                'sharpe': portfolio['sharpe']
            }
            if 'assets' in portfolio:
                portfolio_data[portfolio_name]['assets'] = portfolio['assets']
            if 'transaction_costs' in portfolio:
                portfolio_data[portfolio_name]['transaction_costs'] = portfolio['transaction_costs']
            if 'sector_allocations' in portfolio:
                portfolio_data[portfolio_name]['sector_allocations'] = portfolio['sector_allocations']
        
        print(f"Portfolio data loaded from {filename}")
        return portfolio_data
    except FileNotFoundError:
        print(f"File {filename} not found. No portfolio data loaded.")
        return None

# 4. Generate Enhanced Efficient Frontier with constraints
def generate_enhanced_efficient_frontier(returns, cov_matrix, initial_weights=None, 
                                     risk_free_rate=RISK_FREE_RATE, max_assets=MAX_ASSETS, 
                                     sector_limits=SECTOR_LIMITS, min_position=MIN_POSITION, points=EF_POINTS):
    """
    Generate efficient frontier with practical constraints:
    - Cardinality constraint (maximum number of assets)
    - Minimum position size (if invested)
    - Sector concentration limits
    - Transaction costs
    
    Parameters:
    - returns: Expected annual returns for each asset
    - cov_matrix: Covariance matrix of returns
    - initial_weights: Current portfolio weights (for calculating transaction costs)
    - risk_free_rate: Annual risk-free rate
    - max_assets: Maximum number of assets
    - sector_limits: Dictionary of maximum allocation per sector
    - min_position: Minimum position size
    - points: Number of points on the efficient frontier
    
    Returns:
    - List of portfolio dictionaries along the efficient frontier
    """
    n = len(returns)  # Number of assets
    assets = returns.index.tolist()
    
    # Default initial weights if not provided
    if initial_weights is None:
        initial_weights = pd.Series(0, index=assets)
    
    # Determine return range for efficient frontier
    min_return = min(returns) * 0.8  # Allow slightly lower than min asset return
    max_return = max(returns) * 0.9  # Slightly less than max asset return for feasibility
    
    # Generate target returns for efficient frontier
    target_returns = np.linspace(min_return, max_return, points)
    
    print("\n=== Generating Enhanced Efficient Frontier ===")
    print(f"Return range: {min_return:.4%} to {max_return:.4%}")
    print(f"Constraints: Max {max_assets} assets, Min position {min_position:.1%}, with sector limits")
    if APPLY_TRANSACTION_COSTS:
        print(f"Transaction costs: Base {BASE_TRANSACTION_COST:.2%} with asset-specific factors")
    else:
        print("Transaction costs: Disabled")
    
    efficient_portfolios = []
    sharpe_ratios = []
    
    for i, target_return in enumerate(target_returns):
        print(f"\nOptimizing for target return {target_return:.4%} (point {i+1}/{points})...")
        
        # Create optimization model
        m = Model("Enhanced_Efficient_Frontier")
        m.setParam('OutputFlag', 0)  # Suppress Gurobi output
        
        # Add variables (weights, selection, transaction costs)
        x = m.addVars(assets, lb=0, name="weights")
        z = m.addVars(assets, vtype=GRB.BINARY, name="selection")
        tc = m.addVars(assets, lb=0, name="transaction_costs")
        
        # Constraint: Weights sum to 1 (fully invested)
        m.addConstr(quicksum(x[asset] for asset in assets) == 1, "budget")
        
        # Cardinality constraint: Limit number of assets
        m.addConstr(quicksum(z[asset] for asset in assets) <= max_assets, "max_assets")
        
        # Minimum position size constraint
        for asset in assets:
            m.addConstr(x[asset] <= z[asset], f"select_{asset}")  # Can only invest if selected
            m.addConstr(x[asset] >= min_position * z[asset], f"min_pos_{asset}")  # Minimum position if selected
        
        # Sector constraints
        for sector, sector_assets in SECTORS.items():
            sector_assets_in_model = [a for a in sector_assets if a in assets]
            if sector_assets_in_model:  # Only add constraint if sector assets are in our universe
                m.addConstr(
                    quicksum(x[asset] for asset in sector_assets_in_model) <= sector_limits[sector],
                    f"sector_{sector}"
                )
        
        # Transaction costs (if enabled)
        total_tc = 0
        if APPLY_TRANSACTION_COSTS:
            # Define transaction cost variables (absolute difference between new and initial weights)
            for asset in assets:
                m.addConstr(tc[asset] >= x[asset] - initial_weights[asset], f"tc_pos_{asset}")
                m.addConstr(tc[asset] >= initial_weights[asset] - x[asset], f"tc_neg_{asset}")
            
            # Calculate total transaction cost
            total_tc = quicksum(tc[asset] * TRANSACTION_COSTS[asset] / 100 for asset in assets)
        
        # Target return constraint with transaction costs
        m.addConstr(
            quicksum(returns[asset] * x[asset] for asset in assets) - total_tc >= target_return,
            "target_return"
        )
        
        # Portfolio variance (quadratic objective)
        portfolio_variance = quicksum(
            x[i] * x[j] * cov_matrix.loc[i, j] for i in assets for j in assets
        )
        
        # Objective: Minimize risk
        m.setObjective(portfolio_variance, GRB.MINIMIZE)
        
        # Optimize the model
        try:
            m.optimize()
            
            if m.status == GRB.OPTIMAL:
                # Extract the optimal weights
                optimal_weights = pd.Series({asset: x[asset].X for asset in assets})
                
                # Calculate actual transaction costs
                actual_tc = 0
                if APPLY_TRANSACTION_COSTS:
                    actual_tc = sum(abs(optimal_weights[asset] - initial_weights[asset]) * 
                                  TRANSACTION_COSTS[asset] / 100 for asset in assets)
                
                # Calculate portfolio metrics
                portfolio_return = sum(optimal_weights[asset] * returns[asset] for asset in assets) - actual_tc
                portfolio_risk = np.sqrt(sum(optimal_weights[i] * optimal_weights[j] * cov_matrix.loc[i, j]
                                       for i in assets for j in assets))
                
                # Calculate Sharpe ratio
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                
                # Count actual number of assets used
                assets_used = sum(1 for asset, weight in optimal_weights.items() if weight > 0.001)
                
                # Calculate sector allocations
                sector_allocations = {}
                for sector, sector_assets in SECTORS.items():
                    sector_allocation = sum(optimal_weights[asset] for asset in sector_assets if asset in optimal_weights)
                    sector_allocations[sector] = sector_allocation
                
                # Create portfolio data
                portfolio = {
                    'weights': optimal_weights,
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe': sharpe_ratio,
                    'transaction_costs': actual_tc,
                    'assets_used': assets_used,
                    'sector_allocations': sector_allocations
                }
                
                # Add to our collection
                efficient_portfolios.append(portfolio)
                sharpe_ratios.append(sharpe_ratio)
                
                print(f"  Return: {portfolio_return:.4%}, Risk: {portfolio_risk:.4%}, Sharpe: {sharpe_ratio:.4f}")
                print(f"  Transaction costs: {actual_tc:.4%}, Assets used: {assets_used}")
                
            else:
                print(f"  Optimization failed with status {m.status} - skipping this point")
        
        except Exception as e:
            print(f"  Optimization error: {e} - skipping this point")
    
    print(f"\nGenerated {len(efficient_portfolios)} portfolios for the enhanced efficient frontier")
    
    return efficient_portfolios, sharpe_ratios

# 5. Find optimal enhanced portfolio (maximum Sharpe ratio)
def find_optimal_enhanced_portfolio(efficient_portfolios, sharpe_ratios, risk_free_rate=RISK_FREE_RATE):
    """Find the portfolio with the maximum Sharpe ratio from the enhanced efficient frontier"""
    if not efficient_portfolios:
        print("No portfolios in efficient frontier to evaluate.")
        return None
    
    # Find the index of the portfolio with the maximum Sharpe ratio
    max_sharpe_idx = np.argmax(sharpe_ratios)
    optimal_portfolio = efficient_portfolios[max_sharpe_idx]
    
    print("\n=== Enhanced Portfolio with Maximum Sharpe Ratio ===")
    print(f"Sharpe Ratio: {optimal_portfolio['sharpe']:.4f}")
    print(f"Return: {optimal_portfolio['return']:.4%}")
    print(f"Risk: {optimal_portfolio['risk']:.4%}")
    print(f"Transaction Costs: {optimal_portfolio['transaction_costs']:.4%}")
    print(f"Number of Assets: {optimal_portfolio['assets_used']}")
    
    print("\nSector Allocations:")
    for sector, allocation in optimal_portfolio['sector_allocations'].items():
        print(f"  {sector}: {allocation:.4%}")
    
    print("\nAsset Allocation:")
    for asset, weight in sorted(optimal_portfolio['weights'].items(), key=lambda x: -x[1]):
        if weight > 0.001:  # Only show allocations > 0.1%
            asset_desc = ETF_DESCRIPTIONS.get(asset, asset)
            print(f"  {asset_desc}: {weight:.4%}")
    
    return optimal_portfolio

# 6. Run comparative analysis with original optimal portfolio
def compare_with_original_optimal(returns, cov_matrix, original_optimal_portfolio, risk_free_rate=RISK_FREE_RATE):
    """
    Compare enhanced portfolio with the original optimal portfolio and equal-weighted
    
    Parameters:
    - returns: Expected annual returns for each asset
    - cov_matrix: Covariance matrix of returns
    - original_optimal_portfolio: Original optimal portfolio data from model 1
    - risk_free_rate: Annual risk-free rate
    
    Returns:
    - Dictionary with portfolio comparison data
    """
    assets = returns.index.tolist()
    
    # 1. Equal-weighted portfolio
    equal_weights = pd.Series(1/len(assets), index=assets)
    equal_return = sum(equal_weights[asset] * returns[asset] for asset in assets)
    equal_risk = np.sqrt(sum(equal_weights[i] * equal_weights[j] * cov_matrix.loc[i, j]
                          for i in assets for j in assets))
    equal_sharpe = (equal_return - risk_free_rate) / equal_risk if equal_risk > 0 else 0
    
    print("\n=== Equal-Weighted Portfolio ===")
    print(f"  Return: {equal_return:.4%}")
    print(f"  Risk: {equal_risk:.4%}")
    print(f"  Sharpe Ratio: {equal_sharpe:.4f}")
    print(f"  Number of assets: {len(assets)}")
    
    # 2. Original optimal portfolio (from model 1)
    original_weights = original_optimal_portfolio['weights']
    assets_in_weights = set(original_weights.index)
    assets_in_returns = set(returns.index)
    
    # Handle any potential differences in assets between the two models
    if assets_in_weights != assets_in_returns:
        print(f"Warning: Asset mismatch between original portfolio and current model.")
        # Fill missing assets with 0 weight
        for asset in assets_in_returns - assets_in_weights:
            original_weights[asset] = 0
        # Remove assets that aren't in our universe
        for asset in assets_in_weights - assets_in_returns:
            original_weights = original_weights.drop(asset)
            
        # Renormalize weights to sum to 1
        original_weights = original_weights / original_weights.sum()
    
    # Calculate metrics using current returns and covariance
    original_return = sum(original_weights[asset] * returns[asset] for asset in assets if asset in original_weights)
    original_risk = np.sqrt(sum(
        original_weights.get(i, 0) * original_weights.get(j, 0) * cov_matrix.loc[i, j]
        for i in assets for j in assets if i in original_weights and j in original_weights
    ))
    original_sharpe = (original_return - risk_free_rate) / original_risk if original_risk > 0 else 0
    original_assets_used = sum(1 for weight in original_weights if weight > 0.001)
    
    print("\n=== Original Optimal Portfolio (from Model 1) ===")
    print(f"  Return: {original_return:.4%}")
    print(f"  Risk: {original_risk:.4%}")
    print(f"  Sharpe Ratio: {original_sharpe:.4f}")
    print(f"  Number of assets: {original_assets_used}")
    
    # 3. Generate enhanced efficient frontier and find optimal portfolio
    # Use original weights as initial weights to calculate transaction costs
    efficient_portfolios, sharpe_ratios = generate_enhanced_efficient_frontier(
        returns, 
        cov_matrix, 
        initial_weights=original_weights,
        risk_free_rate=risk_free_rate,
        max_assets=MAX_ASSETS,
        min_position=MIN_POSITION,
        sector_limits=SECTOR_LIMITS,
        points=EF_POINTS
    )
    
    # Find the optimal portfolio with the maximum Sharpe ratio
    enhanced_portfolio = find_optimal_enhanced_portfolio(efficient_portfolios, sharpe_ratios, risk_free_rate)
    
    # Compile results for comparison
    if enhanced_portfolio:
        portfolio_data = {
            'Equal-Weighted': {
                'return': equal_return, 
                'risk': equal_risk, 
                'sharpe': equal_sharpe,
                'assets': len(assets),
                'weights': equal_weights
            },
            'Original Optimal': {
                'return': original_return, 
                'risk': original_risk, 
                'sharpe': original_sharpe,
                'assets': original_assets_used,
                'weights': original_weights
            },
            'Enhanced Optimal': {
                'return': enhanced_portfolio['return'], 
                'risk': enhanced_portfolio['risk'], 
                'sharpe': enhanced_portfolio['sharpe'],
                'assets': enhanced_portfolio['assets_used'],
                'weights': enhanced_portfolio['weights'],
                'transaction_costs': enhanced_portfolio['transaction_costs'],
                'sector_allocations': enhanced_portfolio['sector_allocations']
            }
        }
        
        # Plot the efficient frontier with the portfolios
        plot_enhanced_efficient_frontier(
            efficient_portfolios, 
            portfolio_data,
            risk_free_rate
        )
        
        # Visualize comparison
        plot_portfolio_comparison_enhanced(portfolio_data, risk_free_rate)
        
        # Visualize weights for all portfolios
        plot_weight_comparison(portfolio_data)
        
        # Visualize sector allocation for enhanced portfolio
        plot_sector_allocation(portfolio_data)
        
        return portfolio_data
    
    return None

# Visualization functions
def plot_enhanced_efficient_frontier(efficient_portfolios, portfolio_data, risk_free_rate=RISK_FREE_RATE):
    """Plot the enhanced efficient frontier with key portfolios"""
    # Extract risk and return values from efficient portfolios
    risks = [p['risk'] for p in efficient_portfolios]
    returns = [p['return'] for p in efficient_portfolios]
    
    plt.figure(figsize=(12, 8))
    
    # Plot enhanced efficient frontier
    plt.plot(risks, returns, 'b-', linewidth=2, label='Enhanced Efficient Frontier')
    
    # Plot the key portfolios
    for name, portfolio in portfolio_data.items():
        color = 'purple' if name == 'Equal-Weighted' else 'green' if name == 'Original Optimal' else 'gold'
        marker = 'o' if name == 'Equal-Weighted' else '^' if name == 'Original Optimal' else '*'
        size = 100 if name == 'Equal-Weighted' else 150 if name == 'Original Optimal' else 200
        
        plt.scatter(
            portfolio['risk'], 
            portfolio['return'], 
            color=color, 
            marker=marker, 
            s=size, 
            label=name
        )
    
    # Plot risk-free rate point
    plt.scatter(0, risk_free_rate, c='black', marker='o', s=100, label=f'Risk-Free Rate ({risk_free_rate:.2%})')
    
    # Plot Capital Market Line (CML) for the enhanced optimal portfolio
    enhanced_optimal = portfolio_data['Enhanced Optimal']
    x_values = np.linspace(0, max(risks) * 1.2, 100)
    slope = (enhanced_optimal['return'] - risk_free_rate) / enhanced_optimal['risk']
    y_values = risk_free_rate + slope * x_values
    plt.plot(x_values, y_values, 'g--', linewidth=2, label='Enhanced Capital Market Line')
    
    # Set labels and title
    plt.title('Enhanced Efficient Frontier with Key Portfolios')
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Portfolio Return')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations for key portfolios with adjusted positions to avoid overlap
    # Define offsets for each portfolio type to prevent overlap
    annotation_offsets = {
        'Equal-Weighted': (10, 0),  # right
        'Original Optimal': (-80, 30),  # up and left
        'Enhanced Optimal': (10, 30)  # up and right
    }
    
    for name, portfolio in portfolio_data.items():
        # Get the appropriate offset for this portfolio type
        offset = annotation_offsets.get(name, (10, 0))
        
        plt.annotate(
            f"{name}\nReturn: {portfolio['return']:.2%}\nRisk: {portfolio['risk']:.2%}\nSharpe: {portfolio['sharpe']:.3f}",
            xy=(portfolio['risk'], portfolio['return']),
            xytext=offset,
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=1, color='gray') if name != 'Equal-Weighted' else None
        )
    
    # Save the figure
    plt.savefig('visualisations_model2/enhanced_efficient_frontier.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Enhanced efficient frontier plot saved to 'visualisations_model2/enhanced_efficient_frontier.png'")

def plot_portfolio_comparison_enhanced(portfolio_data, risk_free_rate=RISK_FREE_RATE):
    """Plot enhanced comparison of portfolio strategies"""
    # Extract data
    names = list(portfolio_data.keys())
    returns = [portfolio_data[name]['return'] for name in names]
    risks = [portfolio_data[name]['risk'] for name in names]
    sharpes = [portfolio_data[name]['sharpe'] for name in names]
    asset_counts = [portfolio_data[name]['assets'] for name in names]
    
    # Transaction costs (only for enhanced portfolio)
    tc = portfolio_data['Enhanced Optimal'].get('transaction_costs', 0)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot returns
    axes[0, 0].bar(names, returns, color='skyblue')
    axes[0, 0].set_title('Expected Annual Return')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(returns):
        axes[0, 0].text(i, v/2, f"{v:.2%}", ha='center', va='center', fontweight='bold', color='darkblue')
    
    # Plot risks
    axes[0, 1].bar(names, risks, color='salmon')
    axes[0, 1].set_title('Annual Risk (Standard Deviation)')
    axes[0, 1].set_ylabel('Risk')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(risks):
        axes[0, 1].text(i, v/2, f"{v:.2%}", ha='center', va='center', fontweight='bold', color='darkred')
    
    # Plot Sharpe ratios
    axes[1, 0].bar(names, sharpes, color='lightgreen')
    axes[1, 0].set_title(f'Sharpe Ratio (risk-free rate: {risk_free_rate:.2%})')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(sharpes):
        axes[1, 0].text(i, v/2, f"{v:.4f}", ha='center', va='center', fontweight='bold', color='darkgreen')
    
    # Plot asset counts and transaction costs
    ax_count = axes[1, 1]
    ax_count.bar(names, asset_counts, color='goldenrod')
    ax_count.set_title('Number of Assets Used')
    ax_count.set_ylabel('Asset Count')
    ax_count.tick_params(axis='x', rotation=45)
    ax_count.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(asset_counts):
        ax_count.text(i, v/2, f"{v}", ha='center', va='center', fontweight='bold', color='darkgoldenrod')
    
    # Add transaction cost annotation for Enhanced portfolio
    if APPLY_TRANSACTION_COSTS:
        ax_count.annotate(
            f"Transaction Costs: {tc:.2%}",
            xy=(2, asset_counts[2]),
            xytext=(2, asset_counts[2] + 3),
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="pink", alpha=0.7)
        )
    
    plt.tight_layout()
    
    # Save the figure to the model2 visualizations folder
    plt.savefig('visualisations_model2/enhanced_portfolio_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Enhanced portfolio comparison saved to 'visualisations_model2/enhanced_portfolio_comparison.png'")

def plot_weight_comparison(portfolio_data):
    """Plot comparison of weights across portfolio strategies"""
    strategies = list(portfolio_data.keys())
    
    # Get a common set of assets - use the union of all portfolios' assets
    all_assets = set()
    for strategy in strategies:
        all_assets.update(portfolio_data[strategy]['weights'].index)
    all_assets = sorted(list(all_assets))
    
    # Create a dataframe for plotting with all assets
    weights_df = pd.DataFrame(0, index=all_assets, columns=strategies)
    
    # Fill in the weights
    for strategy in strategies:
        for asset in portfolio_data[strategy]['weights'].index:
            if asset in all_assets:  # Should always be true
                weights_df.loc[asset, strategy] = portfolio_data[strategy]['weights'][asset]
    
    # Replace asset tickers with descriptions
    weights_df = weights_df.rename(index={asset: ETF_DESCRIPTIONS.get(asset, asset) for asset in all_assets})
    
    # Sort by weights in the enhanced portfolio
    sorted_assets = weights_df.sort_values(by='Enhanced Optimal', ascending=False).index
    weights_df = weights_df.reindex(sorted_assets)
    
    # Create plot
    plt.figure(figsize=(15, 10))
    ax = plt.axes()
    
    # Plot heatmap
    sns.heatmap(
        weights_df.T,
        annot=True,
        fmt='.1%',
        cmap='YlGnBu',
        linewidths=0.5,
        cbar_kws={'label': 'Portfolio Weight'},
        ax=ax
    )
    
    plt.title('Portfolio Weights Comparison Across Strategies', fontsize=16)
    plt.tight_layout()
    
    # Save the figure to the model2 visualizations folder
    plt.savefig('visualisations_model2/portfolio_weights_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Portfolio weights comparison saved to 'visualisations_model2/portfolio_weights_comparison.png'")

def plot_sector_allocation(portfolio_data):
    """Plot sector allocation for the enhanced portfolio"""
    sector_allocations = portfolio_data['Enhanced Optimal']['sector_allocations']
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create pie chart
    plt.pie(
        sector_allocations.values(),
        labels=sector_allocations.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('Set3')
    )
    
    plt.title('Enhanced Portfolio Sector Allocation', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Save the figure to the model2 visualizations folder
    plt.savefig('visualisations_model2/sector_allocation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Sector allocation saved to 'visualisations_model2/sector_allocation.png'")

# Create a function to extract the optimal portfolio from model 1 results
def get_optimal_portfolio(file_path="optimal_portfolio.json"):
    """
    Try to load the optimal portfolio from file. If not found, create a dummy placeholder.
    """
    try:
        # Try to load from file
        portfolio_data = load_portfolio_weights(file_path)
        if portfolio_data and "Maximum Sharpe" in portfolio_data:
            return portfolio_data["Maximum Sharpe"]
        else:
            print("Could not find 'Maximum Sharpe' portfolio in loaded data.")
    except Exception as e:
        print(f"Error loading optimal portfolio: {e}")
    
    # Create a placeholder with reasonable values if file not found
    print("Using placeholder for original optimal portfolio.")
    return {
        'weights': pd.Series({
            'SPY': 0.25, 'QQQ': 0.15, 'AGG': 0.20, 'TLT': 0.15, 'GLD': 0.10, 
            'VNQ': 0.10, 'GSG': 0.05
        }),
        'return': 0.12,
        'risk': 0.08,
        'sharpe': 1.125,
        'assets': 7
    }

# Main function
def main():
    # Print configuration for reference
    print("\n" + "="*60)
    print("ENHANCED PORTFOLIO OPTIMIZATION - CONFIGURATION PARAMETERS")
    print("="*60)
    print(f"Risk-Free Rate: {RISK_FREE_RATE:.2%}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Maximum Assets: {MAX_ASSETS}")
    print(f"Minimum Position Size: {MIN_POSITION:.2%}")
    print(f"Efficient Frontier Points: {EF_POINTS}")
    
    # Print transaction cost settings
    if APPLY_TRANSACTION_COSTS:
        print(f"Transaction Costs: Enabled")
        print(f"  - Base Cost: {BASE_TRANSACTION_COST:.2%}")
        print(f"  - Asset-Specific Costs:")
        for ticker in TICKERS:
            cost = TRANSACTION_COSTS.get(ticker, BASE_TRANSACTION_COST)
            print(f"    {ticker}: {cost:.3%}")
    else:
        print("Transaction Costs: Disabled")
    
    print(f"Sector Limits:")
    for sector, limit in SECTOR_LIMITS.items():
        print(f"  - {sector}: {limit:.2%}")
    print(f"Tickers: {', '.join(TICKERS)}")
    print("="*60 + "\n")
    
    # 1. Fetch or load data
    try:
        # Try to load existing data first
        stock_data = pd.read_csv('stock_data.csv', index_col=0, parse_dates=True)
        print("Loaded stock data from file.")
    except FileNotFoundError:
        # Fetch new data if file doesn't exist
        stock_data = fetch_stock_data(TICKERS, START_DATE, END_DATE)
    
    # 2. Calculate returns and covariance
    daily_returns, annual_returns, cov_matrix = calculate_returns_and_covariance(stock_data)
    
    # 3. Get the optimal portfolio from model 1
    original_optimal = get_optimal_portfolio()
    
    # 4. Run comparative analysis with the original optimal portfolio
    portfolio_data = compare_with_original_optimal(
        annual_returns, 
        cov_matrix, 
        original_optimal,
        RISK_FREE_RATE
    )
    
    # 5. Save the enhanced portfolio results for potential future use
    if portfolio_data:
        save_portfolio_weights(
            {'Enhanced Optimal': portfolio_data['Enhanced Optimal']}, 
            "enhanced_optimal_portfolio.json"
        )
    
    print("\nAll visualizations have been saved in the 'visualisations_model2' folder.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()