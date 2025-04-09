import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch

# Set up fonts and styles
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Create output directory
os.makedirs('final_visualization', exist_ok=True)

def load_portfolio_data(filename):
    """Load portfolio data from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error parsing JSON from {filename}.")
        return None

def create_simple_charts(model1_data, model2_data, risk_free_rate=0.03):
    """Create a simplified chart set with exactly 4 bars in each chart"""
    if not model1_data or not model2_data:
        print("Missing portfolio data.")
        return None
    
    # Simplify to 4 portfolios as required
    simplified_portfolios = []
    
    # 1. Equal Weighted (Base Model)
    if 'Equal-Weighted' in model1_data:
        simplified_portfolios.append({
            'name': 'Equal Weighted',
            'return': model1_data['Equal-Weighted']['return'],
            'risk': model1_data['Equal-Weighted']['risk'],
            'sharpe': model1_data['Equal-Weighted']['sharpe'],
            'type': 'base',
            'color': '#80B1DE'  # Light blue
        })
    
    # 2. Min Variance with same return (Base Model)
    if 'Target Return' in model1_data:
        simplified_portfolios.append({
            'name': 'Min Var',
            'return': model1_data['Target Return']['return'],
            'risk': model1_data['Target Return']['risk'],
            'sharpe': model1_data['Target Return']['sharpe'],
            'type': 'base',
            'color': '#80B1DE'  # Light blue
        })
    
    # 3. Maximum Sharpe (Base Model)
    if 'Maximum Sharpe' in model1_data:
        simplified_portfolios.append({
            'name': 'Base Model',
            'return': model1_data['Maximum Sharpe']['return'],
            'risk': model1_data['Maximum Sharpe']['risk'],
            'sharpe': model1_data['Maximum Sharpe']['sharpe'],
            'type': 'base_optimal',
            'color': '#3D6CB2'  # Dark blue
        })
    
    # 4. Enhanced Optimal (Enhanced Model)
    enhanced_optimal = None
    for name, portfolio in model2_data.items():
        if name == "Enhanced Optimal":
            enhanced_optimal = portfolio
            break
    
    if enhanced_optimal:
        simplified_portfolios.append({
            'name': 'Enhanced Model',
            'return': enhanced_optimal['return'],
            'risk': enhanced_optimal['risk'],
            'sharpe': enhanced_optimal['sharpe'],
            'type': 'enhanced',
            'color': '#207F33'  # Dark green
        })
    
    # Extract data for plotting
    names = [p['name'] for p in simplified_portfolios]
    returns = [p['return'] for p in simplified_portfolios]
    risks = [p['risk'] for p in simplified_portfolios]
    sharpes = [p['sharpe'] for p in simplified_portfolios]
    colors = [p['color'] for p in simplified_portfolios]
    
    # Create a 3-panel layout
    fig = plt.figure(figsize=(18, 12))
    
    # Create 3 subplots
    ax1 = plt.subplot2grid((2, 2), (0, 0))  # Returns
    ax2 = plt.subplot2grid((2, 2), (0, 1))  # Risk
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)  # Sharpe (full width)
    
    # Common styling for all axes
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#f8f8f8')
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='-', alpha=0.15, color='#999999')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
    
    # 1. Returns Chart
    bars1 = ax1.bar(names, returns, color=colors, width=0.6)
    ax1.set_title('Expected Annual Return', fontsize=16, pad=15)
    ax1.set_ylabel('Return', fontsize=14)
    
    # Add percentage labels
    for i, v in enumerate(returns):
        label_color = 'white' if v > 0.10 else 'black'
        ax1.text(i, v/2, f"{v:.2%}", ha='center', va='center', 
                fontsize=14, fontweight='bold', color=label_color)
    
    # 2. Risk Chart
    bars2 = ax2.bar(names, risks, color=colors, width=0.6)
    ax2.set_title('Annual Risk (Standard Deviation)', fontsize=16, pad=15)
    ax2.set_ylabel('Risk', fontsize=14)
    
    # Add percentage labels for risks - force black for Equal Weighted and Min Var
    for i, v in enumerate(risks):
        if i < 2:  # First two bars (Equal Weighted and Min Var)
            label_color = 'black'
        else:
            label_color = 'white' if v > 0.10 else 'black'
        
        ax2.text(i, v/2, f"{v:.2%}", ha='center', va='center', 
                fontsize=14, fontweight='bold', color=label_color)
    
    # 3. Sharpe Ratio Chart (full width)
    bars3 = ax3.bar(names, sharpes, color=colors, width=0.4)
    ax3.set_title(f'Sharpe Ratio (Risk-Free Rate: {risk_free_rate:.2%})', fontsize=16, pad=15)
    ax3.set_ylabel('Sharpe Ratio', fontsize=14)
    
    # Add Sharpe ratio labels
    for i, v in enumerate(sharpes):
        label_color = 'white' if v > 0.60 else 'black'
        ax3.text(i, v/2, f"{v:.4f}", ha='center', va='center', 
                fontsize=14, fontweight='bold', color=label_color)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='#3D6CB2', label='Base Model (Optimal)'),
        Patch(facecolor='#207F33', label='Enhanced Model (Optimal)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=14, frameon=True, framealpha=0.9)
    
    # Set overall title and adjust layout
    fig.suptitle('Portfolio Comparison', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.patch.set_facecolor('white')
    
    return fig

def create_weight_comparison(model1_data, model2_data):
    """Create a simple weight comparison between the base optimal and enhanced models"""
    # Get Maximum Sharpe weights
    base_weights = model1_data.get('Maximum Sharpe', {}).get('weights', {})
    
    # Get Enhanced Optimal weights
    enhanced_weights = None
    for name, portfolio in model2_data.items():
        if 'weights' in portfolio:
            enhanced_weights = portfolio['weights']
            break
    
    if not base_weights or not enhanced_weights:
        print("Missing weight data for comparison.")
        return None
    
    # Convert to Series
    base_series = pd.Series(base_weights)
    enhanced_series = pd.Series(enhanced_weights)
    
    # Combine all assets
    all_assets = sorted(set(base_series.index) | set(enhanced_series.index))
    
    # Create DataFrame
    weights_df = pd.DataFrame(0, index=all_assets, columns=['Base Model', 'Enhanced Model'])
    
    # Fill in weights
    for asset in all_assets:
        if asset in base_series.index:
            weights_df.loc[asset, 'Base Model'] = base_series[asset]
        if asset in enhanced_series.index:
            weights_df.loc[asset, 'Enhanced Model'] = enhanced_series[asset]
    
    # Sort by importance
    weights_df['Total'] = weights_df.sum(axis=1)
    weights_df = weights_df.sort_values('Total', ascending=False)
    weights_df = weights_df.drop('Total', axis=1)
    
    # Keep only assets with significant weights
    weights_df = weights_df[(weights_df > 0.01).any(axis=1)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Style the chart
    ax.set_facecolor('#f8f8f8')
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', alpha=0.15, color='#999999')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot bar chart
    weights_df.plot(kind='bar', ax=ax, color=['#3D6CB2', '#207F33'], width=0.7)
    
    ax.set_title('Portfolio Weights Comparison', fontsize=18, pad=20)
    ax.set_xlabel('Asset', fontsize=14)
    ax.set_ylabel('Weight', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    
    # Add percentage labels
    for i, asset in enumerate(weights_df.index):
        base_weight = weights_df.loc[asset, 'Base Model']
        enhanced_weight = weights_df.loc[asset, 'Enhanced Model']
        
        if base_weight > 0.01:
            label_color = 'white' if base_weight > 0.15 else 'black'
            ax.text(i - 0.2, base_weight/2, f"{base_weight:.1%}", 
                    ha='center', va='center', fontsize=13, fontweight='bold', color=label_color)
        
        if enhanced_weight > 0.01:
            label_color = 'white' if enhanced_weight > 0.15 else 'black'
            ax.text(i + 0.2, enhanced_weight/2, f"{enhanced_weight:.1%}", 
                    ha='center', va='center', fontsize=13, fontweight='bold', color=label_color)
    
    plt.tight_layout()
    fig.patch.set_facecolor('white')
    
    return fig

def main():
    """Main function"""
    risk_free_rate = 0.03
    
    # Load data
    model1_data = load_portfolio_data('optimal_portfolio.json')
    model2_data = load_portfolio_data('enhanced_optimal_portfolio.json')
    
    if not model1_data or not model2_data:
        print("Failed to load portfolio data.")
        return
    
    # Create simplified charts
    performance_fig = create_simple_charts(model1_data, model2_data, risk_free_rate)
    
    if performance_fig:
        output_path = 'final_visualization/simplified_performance.png'
        performance_fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(performance_fig)
        print(f"Simplified performance chart saved to '{output_path}'")
    
    # Create weight comparison
    weights_fig = create_weight_comparison(model1_data, model2_data)
    
    if weights_fig:
        weights_path = 'final_visualization/portfolio_weights.png'
        weights_fig.savefig(weights_path, dpi=300, bbox_inches='tight')
        plt.close(weights_fig)
        print(f"Portfolio weights chart saved to '{weights_path}'")
    
    # Print performance summary
    print("\n=== Portfolio Performance Summary ===")
    print(f"{'Portfolio':<20} {'Return':<10} {'Risk':<10} {'Sharpe':<10}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    
    if 'Equal-Weighted' in model1_data:
        print(f"Equal Weighted{'':<7} {model1_data['Equal-Weighted']['return']:.4%} {model1_data['Equal-Weighted']['risk']:.4%} {model1_data['Equal-Weighted']['sharpe']:.4f}")
    
    if 'Target Return' in model1_data:
        print(f"Min Variance{'':<9} {model1_data['Target Return']['return']:.4%} {model1_data['Target Return']['risk']:.4%} {model1_data['Target Return']['sharpe']:.4f}")
    
    if 'Maximum Sharpe' in model1_data:
        print(f"Base Model{'':<11} {model1_data['Maximum Sharpe']['return']:.4%} {model1_data['Maximum Sharpe']['risk']:.4%} {model1_data['Maximum Sharpe']['sharpe']:.4f}")
    
    enhanced = None
    for name, portfolio in model2_data.items():
        if name == "Enhanced Optimal":
            enhanced = portfolio
            break
    
    if enhanced:
        print(f"Enhanced Model{'':<7} {enhanced['return']:.4%} {enhanced['risk']:.4%} {enhanced['sharpe']:.4f}")
        if 'transaction_costs' in enhanced:
            print(f"  Transaction Costs: {enhanced['transaction_costs']:.4%}")
        if 'sector_allocations' in enhanced:
            print("  Sector Allocations:")
            for sector, allocation in enhanced['sector_allocations'].items():
                print(f"    {sector}: {allocation:.4%}")

if __name__ == "__main__":
    main()