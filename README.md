# Portfolio Optimization Project

## Quick Start Guide

This project implements portfolio optimization models using Modern Portfolio Theory with both standard and enhanced implementations.

### Installation

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

   Key dependencies:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - yfinance
   - gurobipy (requires appropriate license)

### How to Run

Run the scripts in this order:

1. Base model (standard Markowitz optimization):
   ```
   python base_model.py
   ```

2. Enhanced model (with practical constraints):
   ```
   python enhanced_model.py
   ```

3. Combined visualization:
   ```
   python combined_visualisation.py
   ```

**Important**: For combined_visualisation.py to work correctly, you must first run base_model.py and enhanced_model.py to generate the JSON results files which the visualization script requires.

All outputs will be saved to the `output/` directory.

## Project Files

### Main Scripts

- **base_model.py**
  - Implements standard Markowitz portfolio optimization
  - Generates the efficient frontier and finds maximum Sharpe ratio portfolio
  - Saves outputs to `output/base_model/`
  - Saves optimal portfolio data to `output/base_model/optimal_portfolio.json`

- **enhanced_model.py**
  - Implements constrained portfolio optimization with:
    - Maximum number of assets constraint
    - Minimum position size
    - Sector allocation limits
    - Transaction costs
  - Uses the base model's optimal portfolio as a reference
  - Saves outputs to `output/enhanced_model/`
  - Saves enhanced portfolio data to `output/enhanced_model/enhanced_optimal_portfolio.json`

- **combined_visualisation.py**
  - Creates consolidated visualizations comparing both models
  - Reads data from both models' JSON outputs
  - Generates performance comparisons, allocation charts, and sector analysis
  - Saves outputs to `output/consolidated/`

### Supporting Files

- **requirements.txt** - Lists all required Python packages
- **stock_data.csv** - Contains historical price data (automatically generated on first run)

## Configurable Parameters

### Base Model Parameters (base_model.py)

```python
# Date range for analysis
START_DATE = '2006-10-03'
END_DATE = '2024-12-31'

# Risk-free rate (annual)
RISK_FREE_RATE = 0.03  # 3%

# Number of points on efficient frontier
POINTS = 20
```

### Enhanced Model Parameters (enhanced_model.py)

```python
# Same date range and risk-free rate as base model

# Enhanced optimization constraints
MAX_ASSETS = 8         # Maximum number of assets in the portfolio
MIN_POSITION = 0.02    # Minimum position size (2%)
EF_POINTS = 20         # Number of points on the efficient frontier

# Maximum allocation per sector
SECTOR_LIMITS = {
    'Equities': 0.70,  # Max 70% in equities
    'Fixed Income': 0.60,  # Max 60% in fixed income
    'Alternative': 0.30    # Max 30% in alternatives
}

# Transaction costs
APPLY_TRANSACTION_COSTS = True
BASE_TRANSACTION_COST = 0.05  # Base cost as percentage
```

## How to Modify Parameters

To experiment with different configurations:

1. **Adjust Time Period**:
   - Change `START_DATE` and `END_DATE` to analyze different time periods
   - Format: 'YYYY-MM-DD'

2. **Modify Risk-Free Rate**:
   - Change `RISK_FREE_RATE` (e.g., 0.02 for 2%, 0.04 for 4%)
   - Used in Sharpe ratio calculations

3. **Enhanced Model Constraints**:
   - Adjust `MAX_ASSETS` to allow more or fewer assets
   - Modify `MIN_POSITION` to change minimum position size
   - Update `SECTOR_LIMITS` to loosen or tighten sector constraints

4. **Transaction Costs**:
   - Set `APPLY_TRANSACTION_COSTS = False` to disable transaction costs
   - Adjust `BASE_TRANSACTION_COST` and individual asset factors

## Output Files

The project generates various output files:

### Base Model Outputs
- Correlation matrix
- Returns vs risk scatter plot
- Efficient frontier
- Optimal portfolio allocation charts
- Portfolio comparison charts
- JSON file with portfolio weights and metrics

### Enhanced Model Outputs
- Enhanced efficient frontier
- Portfolio comparison with base model
- Weight comparison across strategies
- Sector allocation charts
- JSON file with enhanced portfolio data

### Combined Visualization Outputs
- Simplified performance charts
- Portfolio weight heatmap
- Allocation comparison charts
- Sector breakdown analysis

## Asset Universe

The project analyzes the following ETFs, organized by sector:

**Equities:**
- SPY (US Equities - S&P 500)
- QQQ (US Tech)
- VWRA.L (All World Equities)
- VGK (European Equities)
- EEM (Emerging Markets)
- VPL (Asia-Pacific Equities)

**Fixed Income:**
- AGG (US Aggregate Bonds)
- TLT (Long-Term Treasury Bonds)
- LQD (Corporate Bonds)

**Alternatives:**
- GLD (Gold)
- SLV (Silver)
- VNQ (Real Estate)
- GSG (Commodities)
