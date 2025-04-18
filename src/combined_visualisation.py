import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from matplotlib.patches import Patch
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set up fonts and styles
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Create output directory
os.makedirs('output/consolidated', exist_ok=True)

# ETF descriptions for better labeling
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

# Sector classification
SECTORS = {
    'Equities': ['SPY', 'QQQ', 'VWRA.L', 'VGK', 'EEM', 'VPL'],
    'Fixed Income': ['AGG', 'TLT', 'LQD'],
    'Alternative': ['GLD', 'SLV', 'VNQ', 'GSG']
}

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
        if name == "Enhanced Optimal" and 'weights' in portfolio:
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

def create_portfolio_heatmap(model1_data, model2_data):
    """Create a heatmap showing all 4 portfolio allocations"""
    # Extract weights for all portfolios
    portfolios = {
        'Equal-Weighted': model1_data.get('Equal-Weighted', {}).get('weights', {}),
        'Min Variance': model1_data.get('Target Return', {}).get('weights', {}),
        'Base Model': model1_data.get('Maximum Sharpe', {}).get('weights', {})
    }
    
    # Get Enhanced Model weights
    for name, portfolio in model2_data.items():
        if name == "Enhanced Optimal" and 'weights' in portfolio:
            portfolios['Enhanced Model'] = portfolio['weights']
            break
    
    # Check if we have all portfolios
    if len(portfolios) != 4 or any(not weights for weights in portfolios.values()):
        print("Missing portfolio weights for heatmap.")
        return None
    
    # Create a DataFrame for all assets across all portfolios
    all_assets = set()
    for weights in portfolios.values():
        all_assets.update(weights.keys())
    all_assets = sorted(all_assets)
    
    # Create DataFrame with all weights
    weights_df = pd.DataFrame(0, index=portfolios.keys(), columns=all_assets)
    
    # Fill in the weights
    for portfolio_name, weights in portfolios.items():
        for asset, weight in weights.items():
            weights_df.loc[portfolio_name, asset] = weight
    
    # Replace asset tickers with descriptions in columns
    weights_df.columns = [ETF_DESCRIPTIONS.get(asset, asset) for asset in weights_df.columns]
    
    # Sort columns by average weight (descending)
    avg_weights = weights_df.mean()
    weights_df = weights_df[avg_weights.sort_values(ascending=False).index]
    
    # Create figure
    plt.figure(figsize=(20, 10))
    
    # Create heatmap with seaborn
    ax = sns.heatmap(
        weights_df * 100,  # Convert to percentages
        annot=True,
        fmt='.1f',  # Show one decimal place
        cmap='YlGnBu',
        linewidths=1,
        cbar_kws={'label': 'Portfolio Weight (%)', 'shrink': 0.8}
    )
    
    # Set title and labels
    plt.title('Portfolio Weights Across All Strategies (%)', fontsize=18, pad=20)
    plt.ylabel('')
    plt.xlabel('')
    
    # Rotate x labels and adjust font size
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()

def create_allocation_visualization(model1_data, model2_data):
    """Create a clearer asset allocation visualization comparing the two models"""
    # Get Maximum Sharpe weights
    base_weights = model1_data.get('Maximum Sharpe', {}).get('weights', {})
    
    # Get Enhanced Optimal weights
    enhanced_weights = None
    for name, portfolio in model2_data.items():
        if name == "Enhanced Optimal" and 'weights' in portfolio:
            enhanced_weights = portfolio['weights']
            break
    
    if not base_weights or not enhanced_weights:
        print("Missing weight data for comparison.")
        return None
    
    # Convert to Series with descriptive labels
    base_series = pd.Series({ETF_DESCRIPTIONS.get(k, k): v for k, v in base_weights.items()})
    enhanced_series = pd.Series({ETF_DESCRIPTIONS.get(k, k): v for k, v in enhanced_weights.items()})
    
    # Filter to only include significant allocations (>1%)
    base_series = base_series[base_series > 0.01]
    enhanced_series = enhanced_series[enhanced_series > 0.01]
    
    # Create figure - 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Colors for pie charts - use a colorblind-friendly palette
    colors1 = plt.cm.Blues(np.linspace(0.5, 0.9, len(base_series)))
    colors2 = plt.cm.Greens(np.linspace(0.5, 0.9, len(enhanced_series)))
    
    # Plot pie charts
    wedges1, texts1, autotexts1 = ax1.pie(
        base_series.values, 
        labels=None,
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors1,
        pctdistance=0.85,
        wedgeprops={'width': 0.4, 'edgecolor': 'w', 'linewidth': 1}
    )
    
    wedges2, texts2, autotexts2 = ax2.pie(
        enhanced_series.values, 
        labels=None, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors2,
        pctdistance=0.85,
        wedgeprops={'width': 0.4, 'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Style the autoctext (percentage labels)
    for autotext in autotexts1 + autotexts2:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
        autotext.set_color('black')
    
    # Add legends
    ax1.legend(
        wedges1, 
        base_series.index,
        title="Base Model",
        loc="center left",
        bbox_to_anchor=(0, 0.5),
        fontsize=12
    )
    
    ax2.legend(
        wedges2, 
        enhanced_series.index,
        title="Enhanced Model",
        loc="center left",
        bbox_to_anchor=(0, 0.5),
        fontsize=12
    )
    
    # Set titles
    ax1.set_title('Base Model Allocation', fontsize=18, pad=20)
    ax2.set_title('Enhanced Model Allocation', fontsize=18, pad=20)
    
    # Ensure equal aspect ratios
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    # Set overall title
    fig.suptitle('Portfolio Allocation Comparison', fontsize=22, y=0.98)
    
    plt.tight_layout()
    fig.patch.set_facecolor('white')
    
    return fig

def create_sector_allocation_visualization(model1_data, model2_data):
    """Create a sector allocation visualization that groups ETFs by sector"""
    # Get Maximum Sharpe weights
    base_weights = model1_data.get('Maximum Sharpe', {}).get('weights', {})
    
    # Get Enhanced Optimal weights
    enhanced_weights = None
    for name, portfolio in model2_data.items():
        if name == "Enhanced Optimal" and 'weights' in portfolio:
            enhanced_weights = portfolio['weights']
            break
    
    if not base_weights or not enhanced_weights:
        print("Missing weight data for sector comparison.")
        return None
    
    # Create DataFrames to hold sector and asset allocations
    base_df = pd.DataFrame(columns=['Sector', 'Asset', 'Weight'])
    enhanced_df = pd.DataFrame(columns=['Sector', 'Asset', 'Weight'])
    
    # Fill DataFrames with sector and asset information
    for ticker, weight in base_weights.items():
        if weight < 0.01:  # Skip negligible weights
            continue
            
        # Find the sector for this ticker
        sector = "Other"
        for s, tickers in SECTORS.items():
            if ticker in tickers:
                sector = s
                break
        
        # Add to DataFrame
        base_df = pd.concat([base_df, pd.DataFrame({
            'Sector': [sector],
            'Asset': [ETF_DESCRIPTIONS.get(ticker, ticker)],
            'Weight': [weight]
        })])
    
    for ticker, weight in enhanced_weights.items():
        if weight < 0.01:  # Skip negligible weights
            continue
            
        # Find the sector for this ticker
        sector = "Other"
        for s, tickers in SECTORS.items():
            if ticker in tickers:
                sector = s
                break
        
        # Add to DataFrame
        enhanced_df = pd.concat([enhanced_df, pd.DataFrame({
            'Sector': [sector],
            'Asset': [ETF_DESCRIPTIONS.get(ticker, ticker)],
            'Weight': [weight]
        })])
    
    # Group by sector to get sector totals
    base_sectors = base_df.groupby('Sector')['Weight'].sum().reset_index()
    enhanced_sectors = enhanced_df.groupby('Sector')['Weight'].sum().reset_index()
    
    # Create figure with 4 subplots (sector and asset allocation for each model)
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Create 4 subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Base Model - Sectors
    ax2 = fig.add_subplot(gs[0, 1])  # Enhanced Model - Sectors
    ax3 = fig.add_subplot(gs[1, 0])  # Base Model - Assets
    ax4 = fig.add_subplot(gs[1, 1])  # Enhanced Model - Assets
    
    # Define colors for sectors
    sector_colors = {
        'Equities': '#1f77b4',       # Blue
        'Fixed Income': '#2ca02c',   # Green
        'Alternative': '#d62728',    # Red
        'Other': '#9467bd'           # Purple
    }
    
    # Plot sector allocations
    base_sector_colors = [sector_colors.get(sector, '#9467bd') for sector in base_sectors['Sector']]
    enhanced_sector_colors = [sector_colors.get(sector, '#9467bd') for sector in enhanced_sectors['Sector']]
    
    ax1.pie(
        base_sectors['Weight'], 
        labels=base_sectors['Sector'],
        autopct='%1.1f%%', 
        startangle=90, 
        colors=base_sector_colors,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    ax2.pie(
        enhanced_sectors['Weight'], 
        labels=enhanced_sectors['Sector'],
        autopct='%1.1f%%', 
        startangle=90, 
        colors=enhanced_sector_colors,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Plot asset allocations with color coded by sector
    for i, row in base_df.iterrows():
        sector = row['Sector']
        asset = row['Asset']
        weight = row['Weight']
        color = sector_colors.get(sector, '#9467bd')
        
        # Adjust color brightness for variety within same sector
        brightness_factor = 0.7 + (i % 3) * 0.1  # Vary brightness
        color_rgb = plt.cm.colors.to_rgb(color)
        adjusted_color = tuple(min(c * brightness_factor, 1.0) for c in color_rgb)
        
        ax3.barh(asset, weight, color=adjusted_color, edgecolor='white', linewidth=0.5)
    
    for i, row in enhanced_df.iterrows():
        sector = row['Sector']
        asset = row['Asset']
        weight = row['Weight']
        color = sector_colors.get(sector, '#9467bd')
        
        # Adjust color brightness for variety within same sector
        brightness_factor = 0.7 + (i % 3) * 0.1  # Vary brightness
        color_rgb = plt.cm.colors.to_rgb(color)
        adjusted_color = tuple(min(c * brightness_factor, 1.0) for c in color_rgb)
        
        ax4.barh(asset, weight, color=adjusted_color, edgecolor='white', linewidth=0.5)
    
    # Add percentage labels to the horizontal bars
    for ax in [ax3, ax4]:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=5, 
                        label_type='edge', fontsize=10, fontweight='bold')
    
    # Set titles and style
    ax1.set_title('Base Model - Sector Allocation', fontsize=16)
    ax2.set_title('Enhanced Model - Sector Allocation', fontsize=16)
    ax3.set_title('Base Model - Asset Allocation', fontsize=16)
    ax4.set_title('Enhanced Model - Asset Allocation', fontsize=16)
    
    # Ensure equal aspect ratios for pie charts
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    # Style horizontal bar charts
    for ax in [ax3, ax4]:
        ax.set_xlabel('Weight (%)', fontsize=12)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle='-', alpha=0.15, color='#999999')
    
    # Add labels to horizontal bar charts
    ax3.invert_yaxis()  # Invert y-axis to match standard ordering
    ax4.invert_yaxis()
    
    # Set overall title
    fig.suptitle('Portfolio Allocation Comparison by Sector and Asset', fontsize=20, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.patch.set_facecolor('white')
    
    return fig

def main():
    """Main function"""
    risk_free_rate = 0.03
    
    # Load data
    model1_data = load_portfolio_data('output/base_model/optimal_portfolio.json')
    model2_data = load_portfolio_data('output/enhanced_model/enhanced_optimal_portfolio.json')
    
    if not model1_data or not model2_data:
        print("Failed to load portfolio data.")
        return
    
    # Create simplified charts
    performance_fig = create_simple_charts(model1_data, model2_data, risk_free_rate)
    
    if performance_fig:
        output_path = 'output/consolidated/simplified_performance.png'
        performance_fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(performance_fig)
        print(f"Simplified performance chart saved to '{output_path}'")
    
    # Create portfolio heatmap for all 4 portfolios
    heatmap_fig = create_portfolio_heatmap(model1_data, model2_data)
    
    if heatmap_fig:
        heatmap_path = 'output/consolidated/portfolio_heatmap.png'
        heatmap_fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(heatmap_fig)
        print(f"Portfolio heatmap saved to '{heatmap_path}'")
    
    # Create weight comparison
    weights_fig = create_weight_comparison(model1_data, model2_data)
    
    if weights_fig:
        weights_path = 'output/consolidated/portfolio_weights.png'
        weights_fig.savefig(weights_path, dpi=300, bbox_inches='tight')
        plt.close(weights_fig)
        print(f"Portfolio weights chart saved to '{weights_path}'")
    
    # Create donut chart allocation visualization
    allocation_fig = create_allocation_visualization(model1_data, model2_data)
    
    if allocation_fig:
        allocation_path = 'output/consolidated/portfolio_allocation.png'
        allocation_fig.savefig(allocation_path, dpi=300, bbox_inches='tight')
        plt.close(allocation_fig)
        print(f"Portfolio allocation chart saved to '{allocation_path}'")
    
    # Create sector allocation visualization
    sector_fig = create_sector_allocation_visualization(model1_data, model2_data)
    
    if sector_fig:
        sector_path = 'output/consolidated/sector_allocation.png'
        sector_fig.savefig(sector_path, dpi=300, bbox_inches='tight')
        plt.close(sector_fig)
        print(f"Sector allocation chart saved to '{sector_path}'")
    
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