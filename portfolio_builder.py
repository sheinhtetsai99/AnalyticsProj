# portfolio_builder.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.mean_variance import MeanVarianceModel
from models.constraints import ConstrainedMeanVarianceModel
from models.integer_model import IntegerConstrainedModel
# Import additional models as they are implemented

class PortfolioBuilder:
    """
    Unified interface for building and analyzing portfolio optimization models.
    Provides methods to easily create and compare different models.
    """
    
    def __init__(self, returns_data=None, expected_returns=None, covariance_matrix=None, asset_names=None):
        """
        Initialize the portfolio builder.
        
        Parameters:
        -----------
        returns_data : pd.DataFrame, optional
            Historical returns data with assets in columns and time in rows
        expected_returns : np.ndarray or pd.Series, optional
            Expected returns for each asset (not needed if returns_data is provided)
        covariance_matrix : np.ndarray or pd.DataFrame, optional
            Covariance matrix (not needed if returns_data is provided)
        asset_names : list, optional
            Names of assets (not needed if returns_data is provided with named columns)
        """
        self.returns_data = returns_data
        
        # If returns_data is provided, calculate expected returns and covariance
        if returns_data is not None:
            if isinstance(returns_data, pd.DataFrame):
                self.asset_names = returns_data.columns.tolist()
                self.expected_returns = returns_data.mean().values
                self.covariance_matrix = returns_data.cov().values
            else:
                raise ValueError("returns_data must be a pandas DataFrame")
        else:
            # Use provided expected returns and covariance
            if expected_returns is None or covariance_matrix is None:
                raise ValueError("If returns_data is not provided, both expected_returns and covariance_matrix must be provided")
                
            self.expected_returns = expected_returns
            self.covariance_matrix = covariance_matrix
            self.asset_names = asset_names if asset_names is not None else [
                f"Asset_{i}" for i in range(len(expected_returns))
            ]
            
        # Store the number of assets
        self.n_assets = len(self.expected_returns)
        
        # Dictionary to store created models
        self.models = {}
    
    def create_mean_variance_model(self, risk_aversion=1.0, model_name="mean_variance"):
        """
        Create a basic mean-variance optimization model.
        
        Parameters:
        -----------
        risk_aversion : float, optional
            Risk aversion parameter
        model_name : str, optional
            Name to identify this model
            
        Returns:
        --------
        MeanVarianceModel
            The created model
        """
        model = MeanVarianceModel(
            self.expected_returns,
            self.covariance_matrix,
            self.asset_names,
            risk_aversion
        )
        
        self.models[model_name] = model
        return model
    
    def create_constrained_model(self, risk_aversion=1.0, model_name="constrained"):
        """
        Create a constrained mean-variance optimization model.
        
        Parameters:
        -----------
        risk_aversion : float, optional
            Risk aversion parameter
        model_name : str, optional
            Name to identify this model
            
        Returns:
        --------
        ConstrainedMeanVarianceModel
            The created model
        """
        model = ConstrainedMeanVarianceModel(
            self.expected_returns,
            self.covariance_matrix,
            self.asset_names,
            risk_aversion
        )
        
        self.models[model_name] = model
        return model
    
    def create_integer_model(self, risk_aversion=1.0, model_name="integer"):
        """
        Create an integer-constrained optimization model.
        
        Parameters:
        -----------
        risk_aversion : float, optional
            Risk aversion parameter
        model_name : str, optional
            Name to identify this model
            
        Returns:
        --------
        IntegerConstrainedModel
            The created model
        """
        model = IntegerConstrainedModel(
            self.expected_returns,
            self.covariance_matrix,
            self.asset_names,
            risk_aversion
        )
        
        self.models[model_name] = model
        return model
    
    def build_and_solve_model(self, model_type="mean_variance", risk_aversion=1.0,
                             constraints=None, solve=True, verbose=False, model_name=None):
        """
        Create, build, and optionally solve a model with specified constraints.
        
        Parameters:
        -----------
        model_type : str, optional
            Type of model to create ("mean_variance", "constrained", "integer")
        risk_aversion : float, optional
            Risk aversion parameter
        constraints : dict, optional
            Dictionary of constraints to add
        solve : bool, optional
            Whether to solve the model
        verbose : bool, optional
            Whether to print solver output
        model_name : str, optional
            Name to identify this model. If None, auto-generated
            
        Returns:
        --------
        object
            The created model
        """
        # Generate model name if not provided
        if model_name is None:
            model_name = f"{model_type}_{len(self.models)}"
            
        # Create the model
        if model_type == "mean_variance":
            model = self.create_mean_variance_model(risk_aversion, model_name)
        elif model_type == "constrained":
            model = self.create_constrained_model(risk_aversion, model_name)
        elif model_type == "integer":
            model = self.create_integer_model(risk_aversion, model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Build the model
        model.build_model()
        
        # Add constraints if provided
        if constraints:
            self._add_constraints(model, constraints)
            
        # Solve the model if requested
        if solve:
            model.solve(verbose=verbose)
            
        return model
    
    def _add_constraints(self, model, constraints):
        """
        Add constraints to a model.
        
        Parameters:
        -----------
        model : object
            Model to add constraints to
        constraints : dict
            Dictionary of constraints to add
        """
        # Add constraints based on model type
        if isinstance(model, IntegerConstrainedModel):
            # Handle integer constraints
            if 'max_assets' in constraints:
                model.add_cardinality_constraint(constraints['max_assets'])
                
            if 'min_position_size' in constraints:
                model.add_min_position_size(constraints['min_position_size'])
                
            if 'buy_in_threshold' in constraints:
                model.add_buy_in_threshold(constraints['buy_in_threshold'])
                
            if 'min_assets' in constraints:
                model.add_diversification_constraint(constraints['min_assets'])
                
            if 'pre_selected_assets' in constraints:
                model.add_pre_selected_assets(constraints['pre_selected_assets'])
        
        if isinstance(model, (ConstrainedMeanVarianceModel, IntegerConstrainedModel)):
            # Handle linear constraints
            if 'asset_bounds' in constraints:
                bounds = constraints['asset_bounds']
                model.add_asset_bounds(
                    bounds.get('min_weights'),
                    bounds.get('max_weights')
                )
                
            if 'sector_constraints' in constraints:
                sector_info = constraints['sector_constraints']
                model.add_sector_constraints(
                    sector_info['mapping'],
                    sector_info.get('min_allocation'),
                    sector_info.get('max_allocation')
                )
                
            if 'group_constraints' in constraints:
                group_info = constraints['group_constraints']
                model.add_group_constraints(
                    group_info['groups'],
                    group_info.get('min_allocation'),
                    group_info.get('max_allocation')
                )
                
            if 'target_return' in constraints:
                model.add_target_return(constraints['target_return'])
                
            if 'max_risk' in constraints:
                model.add_target_risk(constraints['max_risk'])
    
    def compare_models(self, model_names=None, metrics=None):
        """
        Compare performance metrics for multiple models.
        
        Parameters:
        -----------
        model_names : list, optional
            Names of models to compare. If None, use all solved models
        metrics : list, optional
            Metrics to compare. If None, use standard metrics
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with metrics for each model
        """
        # If no model names provided, use all solved models
        if model_names is None:
            model_names = [name for name, model in self.models.items() 
                          if model.solution is not None]
            
        # If no metrics provided, use standard metrics
        if metrics is None:
            metrics = ['expected_return', 'portfolio_volatility', 'sharpe_ratio', 'num_assets_selected']
            
        # Collect metrics for each model
        data = []
        for name in model_names:
            if name not in self.models:
                print(f"Warning: Model '{name}' not found, skipping")
                continue
                
            model = self.models[name]
            if model.solution is None:
                print(f"Warning: Model '{name}' has not been solved, skipping")
                continue
                
            row = {'model_name': name, 'model_type': model.model_type}
            for metric in metrics:
                if metric in model.solution:
                    row[metric] = model.solution[metric]
                else:
                    row[metric] = None
                    
            data.append(row)
            
        # Create DataFrame
        if data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(columns=['model_name', 'model_type'] + metrics)
    
    def plot_portfolio_weights(self, model_name, min_weight=0.01, figsize=(12, 8), title=None):
        """
        Plot the portfolio weights for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to plot
        min_weight : float, optional
            Minimum weight to include in the plot
        figsize : tuple, optional
            Figure size
        title : str, optional
            Plot title. If None, auto-generated
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
            
        model = self.models[model_name]
        if model.solution is None:
            raise ValueError(f"Model '{model_name}' has not been solved")
            
        # Get weights
        weights = model.get_weights()
        
        # Filter small weights
        weights = weights[weights >= min_weight]
        
        # Sort by weight (descending)
        weights = weights.sort_values(ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot weights
        weights.plot(kind='bar', ax=ax)
        
        # Set title
        if title is None:
            title = f"Portfolio Weights for {model_name}"
        ax.set_title(title)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_efficient_frontier(self, model_name=None, num_points=20, 
                            show_assets=True, figsize=(12, 8), title=None):
        """
        Plot the efficient frontier for a model.
        
        Parameters:
        -----------
        model_name : str, optional
            Name of the model to use. If None, create a new mean-variance model
        num_points : int, optional
            Number of points on the efficient frontier
        show_assets : bool, optional
            Whether to show individual assets
        figsize : tuple, optional
            Figure size
        title : str, optional
            Plot title. If None, auto-generated
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Use specified model or create a new one
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")
            model = self.models[model_name]
        else:
            # Create a new model with explicit parameter naming
            model = self.create_mean_variance_model(
                risk_aversion=1.0, 
                model_name="efficient_frontier"
            )
            model.build_model()
            
        # Calculate efficient frontier
        frontier = model.get_efficient_frontier(n_points=num_points)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot efficient frontier
        ax.plot(frontier['risk'], frontier['return'], 'b-', linewidth=3, label="Efficient Frontier")
        ax.scatter(frontier['risk'], frontier['return'], c='blue', s=50, alpha=0.7)
        
        # Mark the solved model if available
        if model.solution is not None:
            vol = model.solution['portfolio_volatility']
            ret = model.solution['expected_return']
            ax.scatter([vol], [ret], c='green', s=200, marker='*', label=f"{model_name} Portfolio")
            
        # Show individual assets if requested
        if show_assets:
            asset_vols = [np.sqrt(self.covariance_matrix[i, i]) for i in range(self.n_assets)]
            asset_rets = self.expected_returns
            
            ax.scatter(asset_vols, asset_rets, s=100, c='red', alpha=0.5, label="Individual Assets")
            
            # Add asset labels
            for i, (vol, ret) in enumerate(zip(asset_vols, asset_rets)):
                ax.annotate(self.asset_names[i], (vol, ret), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
                
        # Set title
        if title is None:
            title = "Efficient Frontier"
        ax.set_title(title)
        
        # Labels
        ax.set_xlabel("Portfolio Volatility")
        ax.set_ylabel("Expected Return")
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend()
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def backtest_portfolio(self, model_name, test_data=None, benchmark=None, risk_free_rate=0.0):
        """
        Backtest a portfolio on historical returns data.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to backtest
        test_data : pd.DataFrame, optional
            Test returns data. If None, use self.returns_data
        benchmark : pd.Series or str, optional
            Benchmark returns or column name in test_data
        risk_free_rate : float, optional
            Annual risk-free rate
            
        Returns:
        --------
        dict
            Backtest results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
            
        model = self.models[model_name]
        if model.solution is None:
            raise ValueError(f"Model '{model_name}' has not been solved")
            
        # Use provided test data or self.returns_data
        if test_data is None:
            if self.returns_data is None:
                raise ValueError("No test data provided and no returns_data available")
            test_data = self.returns_data
            
        # Get weights
        weights_series = model.get_weights()
        
        # Ensure alignment between test_data columns and weights indices
        common_assets = [asset for asset in test_data.columns 
                        if asset in weights_series.index and asset != 'Benchmark']
        if not common_assets:
            raise ValueError("No common assets between test data and portfolio weights")
        
        # Create aligned weights series
        aligned_weights = pd.Series(0.0, index=common_assets)
        for asset in common_assets:
            aligned_weights[asset] = weights_series[asset]
        
        # Normalize weights to sum to 1
        if abs(aligned_weights.sum() - 1.0) > 1e-6:
            aligned_weights = aligned_weights / aligned_weights.sum()
        
        # Use only common assets in test data
        test_data_aligned = test_data[common_assets]
        
        # Calculate portfolio returns
        portfolio_returns = test_data_aligned.dot(aligned_weights)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        
        # Calculate benchmark returns if provided
        benchmark_returns = None
        benchmark_cumulative = None
        if benchmark is not None:
            if isinstance(benchmark, str):
                # Benchmark is a column name in test_data
                if benchmark in test_data.columns:
                    benchmark_returns = test_data[benchmark]
                else:
                    raise ValueError(f"Benchmark column '{benchmark}' not found in test_data")
            else:
                # Benchmark is a Series
                benchmark_returns = benchmark
                    
            # Calculate cumulative benchmark returns
            benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
                
        # Calculate performance metrics
        
        # Determine frequency for annualization
        if isinstance(test_data.index, pd.DatetimeIndex):
            freq = pd.infer_freq(test_data.index)
            if freq in ['D', 'B']:
                annualization_factor = 252
            elif freq in ['W', 'W-SUN', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT']:
                annualization_factor = 52
            elif freq in ['M', 'MS', 'BM', 'BMS']:
                annualization_factor = 12
            elif freq in ['Q', 'QS', 'BQ', 'BQS']:
                annualization_factor = 4
            elif freq in ['A', 'AS', 'BA', 'BAS']:
                annualization_factor = 1
            else:
                annualization_factor = 252  # Default to daily
        else:
            annualization_factor = 252  # Default to daily
            
        # Calculate annualized return and volatility
        annual_return = portfolio_returns.mean() * annualization_factor
        annual_volatility = portfolio_returns.std() * np.sqrt(annualization_factor)
        
        # Calculate Sharpe ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Calculate maximum drawdown
        running_max = np.maximum.accumulate(cumulative_returns + 1)
        drawdown = (cumulative_returns + 1) / running_max - 1
        max_drawdown = drawdown.min()
        
        # Prepare results
        results = {
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        # Add benchmark results if available
        if benchmark_returns is not None:
            benchmark_annual_return = benchmark_returns.mean() * annualization_factor
            benchmark_annual_volatility = benchmark_returns.std() * np.sqrt(annualization_factor)
            benchmark_sharpe = (benchmark_annual_return - risk_free_rate) / benchmark_annual_volatility
            
            benchmark_running_max = np.maximum.accumulate(benchmark_cumulative + 1)
            benchmark_drawdown = (benchmark_cumulative + 1) / benchmark_running_max - 1
            benchmark_max_drawdown = benchmark_drawdown.min()
            
            results.update({
                'benchmark_returns': benchmark_returns,
                'benchmark_cumulative': benchmark_cumulative,
                'benchmark_annual_return': benchmark_annual_return,
                'benchmark_annual_volatility': benchmark_annual_volatility,
                'benchmark_sharpe_ratio': benchmark_sharpe,
                'benchmark_max_drawdown': benchmark_max_drawdown,
                'tracking_error': np.sqrt(((portfolio_returns - benchmark_returns) ** 2).mean()) * np.sqrt(annualization_factor),
                'information_ratio': (annual_return - benchmark_annual_return) / 
                                    (np.sqrt(((portfolio_returns - benchmark_returns) ** 2).mean()) * np.sqrt(annualization_factor))
                if np.sqrt(((portfolio_returns - benchmark_returns) ** 2).mean()) > 0 else 0
            })
            
        return results
    
    def plot_backtest_results(self, backtest_results, figsize=(14, 10), title=None):
        """
        Plot backtest results.
        
        Parameters:
        -----------
        backtest_results : dict
            Results from backtest_portfolio()
        figsize : tuple, optional
            Figure size
        title : str, optional
            Plot title. If None, auto-generated
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create gridspec
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
        
        # Plot cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot portfolio cumulative returns
        backtest_results['cumulative_returns'].plot(ax=ax1, label="Portfolio", linewidth=2)
        
        # Plot benchmark if available
        if 'benchmark_cumulative' in backtest_results:
            backtest_results['benchmark_cumulative'].plot(ax=ax1, label="Benchmark", linewidth=2, linestyle='--')
            
        # Format y-axis as percentage
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Set title
        if title is None:
            title = "Portfolio Backtest Results"
        ax1.set_title(title)
        
        # Labels
        ax1.set_ylabel("Cumulative Return")
        
        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax1.legend()
        
        # Plot drawdowns
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Calculate drawdowns
        running_max = np.maximum.accumulate(backtest_results['cumulative_returns'] + 1)
        drawdown = (backtest_results['cumulative_returns'] + 1) / running_max - 1
        
        # Plot drawdowns
        drawdown.plot(ax=ax2, label="Portfolio", linewidth=2)
        
        # Plot benchmark drawdowns if available
        if 'benchmark_cumulative' in backtest_results:
            benchmark_running_max = np.maximum.accumulate(backtest_results['benchmark_cumulative'] + 1)
            benchmark_drawdown = (backtest_results['benchmark_cumulative'] + 1) / benchmark_running_max - 1
            benchmark_drawdown.plot(ax=ax2, label="Benchmark", linewidth=2, linestyle='--')
            
        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Set title
        ax2.set_title("Drawdowns")
        
        # Labels
        ax2.set_ylabel("Drawdown")
        
        # Add grid
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax2.legend()
        
        # Plot performance metrics
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Collect metrics
        metrics = {
            'Annual Return': backtest_results['annual_return'],
            'Annual Volatility': backtest_results['annual_volatility'],
            'Sharpe Ratio': backtest_results['sharpe_ratio'],
            'Max Drawdown': backtest_results['max_drawdown']
        }
        
        # Add benchmark metrics if available
        if 'benchmark_annual_return' in backtest_results:
            metrics.update({
                'Benchmark Return': backtest_results['benchmark_annual_return'],
                'Benchmark Volatility': backtest_results['benchmark_annual_volatility'],
                'Benchmark Sharpe': backtest_results['benchmark_sharpe_ratio'],
                'Benchmark Max DD': backtest_results['benchmark_max_drawdown'],
                'Tracking Error': backtest_results['tracking_error'],
                'Information Ratio': backtest_results['information_ratio']
            })
            
        # Create metrics table
        data = []
        for metric, value in metrics.items():
            if metric in ['Annual Return', 'Annual Volatility', 'Max Drawdown', 
                        'Benchmark Return', 'Benchmark Volatility', 'Benchmark Max DD', 
                        'Tracking Error']:
                formatted_value = f"{value:.2%}"
            else:
                formatted_value = f"{value:.2f}"
            data.append([metric, formatted_value])
            
        # Hide axes
        ax3.axis('off')
        
        # Create table
        table = ax3.table(
            cellText=data,
            colLabels=['Metric', 'Value'],
            cellLoc='center',
            loc='center'
        )
        
        # Set title
        ax3.set_title("Performance Metrics")
        
        # Adjust table style
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, model_names=None, output_file=None):
        """
        Generate a comprehensive report on the models.
        
        Parameters:
        -----------
        model_names : list, optional
            Names of models to include. If None, use all solved models
        output_file : str, optional
            Path to save the report. If None, print to console
            
        Returns:
        --------
        str
            Report content
        """
        # If no model names provided, use all solved models
        if model_names is None:
            model_names = [name for name, model in self.models.items() 
                          if model.solution is not None]
            
        # Generate report
        report = []
        report.append("# Portfolio Optimization Report")
        report.append(f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Asset information
        report.append("## Asset Information")
        report.append(f"Number of assets: {self.n_assets}")
        report.append("Top 5 assets by expected return:")
        top_returns = pd.Series(self.expected_returns, index=self.asset_names).sort_values(ascending=False).head(5)
        for asset, ret in top_returns.items():
            report.append(f"- {asset}: {ret:.2%}")
        report.append("")
        
        # Model comparison
        report.append("## Model Comparison")
        comparison = self.compare_models(model_names)
        if not comparison.empty:
            report.append("| Model | Type | Return | Risk | Sharpe | Assets |")
            report.append("|-------|------|--------|------|--------|--------|")
            for _, row in comparison.iterrows():
                model_name = row['model_name']
                model_type = row['model_type']
                ret = row.get('expected_return', "N/A")
                risk = row.get('portfolio_volatility', "N/A")
                sharpe = row.get('sharpe_ratio', "N/A")
                assets = row.get('num_assets_selected', "N/A")
                
                # Format values
                if isinstance(ret, (int, float)):
                    ret = f"{ret:.2%}"
                if isinstance(risk, (int, float)):
                    risk = f"{risk:.2%}"
                if isinstance(sharpe, (int, float)):
                    sharpe = f"{sharpe:.2f}"
                    
                report.append(f"| {model_name} | {model_type} | {ret} | {risk} | {sharpe} | {assets} |")
        else:
            report.append("No solved models to compare.")
        report.append("")
        
        # Individual model details
        report.append("## Model Details")
        for name in model_names:
            if name not in self.models:
                continue
                
            model = self.models[name]
            if model.solution is None:
                continue
                
            report.append(f"### {name}")
            report.append(f"Model type: {model.model_type}")
            report.append(f"Status: {model.status}")
            
            # Performance metrics
            metrics = model.get_performance_metrics()
            report.append("#### Performance Metrics")
            for metric, value in metrics.items():
                if metric in ['weights', 'selected_assets', 'selected_asset_names', 
                            'constraints_added', 'integer_constraints_added', 'solver_status']:
                    continue
                    
                if metric in ['expected_return', 'portfolio_volatility', 'portfolio_variance']:
                    report.append(f"- {metric}: {value:.2%}")
                else:
                    report.append(f"- {metric}: {value}")
            report.append("")
            
            # Portfolio composition
            weights = model.get_weights()
            non_zero_weights = weights[weights > 0.001]
            
            report.append("#### Portfolio Composition")
            if not non_zero_weights.empty:
                report.append("| Asset | Weight |")
                report.append("|-------|--------|")
                for asset, weight in non_zero_weights.sort_values(ascending=False).items():
                    report.append(f"| {asset} | {weight:.2%} |")
            else:
                report.append("No non-zero weights found.")
            report.append("")
            
            # Constraints
            if hasattr(model, 'constraints_added'):
                active_constraints = [c for c, v in model.constraints_added.items() if v]
                if active_constraints:
                    report.append("#### Active Constraints")
                    for constraint in active_constraints:
                        report.append(f"- {constraint}")
                    report.append("")
                    
            if hasattr(model, 'integer_constraints_added'):
                active_int_constraints = [c for c, v in model.integer_constraints_added.items() if v]
                if active_int_constraints:
                    report.append("#### Active Integer Constraints")
                    for constraint in active_int_constraints:
                        report.append(f"- {constraint}")
                    report.append("")
            
        # Join report lines
        report_text = "\n".join(report)
        
        # Save to file or print
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        else:
            print(report_text)
            
        return report_text