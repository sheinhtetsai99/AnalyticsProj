# models/mean_variance.py
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from .base_model import BasePortfolioModel

class MeanVarianceModel(BasePortfolioModel):
    """
    Implementation of the classic Markowitz mean-variance portfolio optimization model.
    """
    
    def __init__(self, expected_returns, covariance_matrix, asset_names=None, risk_aversion=1.0):
        """
        Initialize the mean-variance model.
        
        Parameters:
        -----------
        expected_returns : np.ndarray or pd.Series
            Expected returns for each asset
        covariance_matrix : np.ndarray or pd.DataFrame
            Covariance matrix of asset returns
        asset_names : list, optional
            Names of assets. If None, generic names are created
        risk_aversion : float, optional
            Risk aversion parameter (lambda) in the objective function
        """
        super().__init__(expected_returns, covariance_matrix, asset_names)
        self.risk_aversion = risk_aversion
        self.model_type = "mean-variance"
    
    def build_model(self):
        """
        Build the mean-variance optimization model.
        
        Returns:
        --------
        self : MeanVarianceModel
            The model instance (for method chaining)
        """
        # Create a new model
        self.model = gp.Model("Mean_Variance_Portfolio")
        
        # Add weight variables (0 to 1 for each asset)
        self.weight_vars = self.model.addVars(
            self.n_assets, 
            lb=0, 
            ub=1, 
            name="weights"
        )
        
        # Budget constraint (sum of weights = 1)
        self.model.addConstr(
            gp.quicksum(self.weight_vars[i] for i in range(self.n_assets)) == 1, 
            "budget_constraint"
        )
        
        # Define portfolio return
        portfolio_return = gp.quicksum(
            self.expected_returns[i] * self.weight_vars[i] 
            for i in range(self.n_assets)
        )
        
        # Define portfolio risk (variance)
        portfolio_risk = gp.quicksum(
            self.weight_vars[i] * self.covariance_matrix[i, j] * self.weight_vars[j]
            for i in range(self.n_assets) 
            for j in range(self.n_assets)
        )
        
        # Ensure risk_aversion is a scalar float
        risk_aversion = float(self.risk_aversion)
        
        # Set objective: maximize return - risk_aversion * risk
        self.model.setObjective(
            portfolio_return - risk_aversion * portfolio_risk, 
            GRB.MAXIMIZE
        )
        
        # Store expressions for later use
        self.portfolio_return_expr = portfolio_return
        self.portfolio_risk_expr = portfolio_risk
        
        self.status = "built"
        return self
    
    def solve(self, verbose=False, time_limit=None):
        """
        Solve the optimization model.
        
        Parameters:
        -----------
        verbose : bool, optional
            If True, print solver output
        time_limit : int, optional
            Time limit in seconds for solver
            
        Returns:
        --------
        dict
            Solution details
        """
        if self.model is None:
            self.build_model()
            
        # Set verbosity
        self.model.setParam('OutputFlag', 1 if verbose else 0)
        
        # Set time limit if specified
        if time_limit is not None:
            self.model.setParam('TimeLimit', time_limit)
            
        # Solve model
        self.model.optimize()
        
        # Check if a solution was found
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            # Extract optimal weights
            weights = [self.weight_vars[i].X for i in range(self.n_assets)]
            
            # Calculate performance metrics
            expected_return = sum(self.expected_returns[i] * weights[i] for i in range(self.n_assets))
            portfolio_variance = sum(weights[i] * self.covariance_matrix[i, j] * weights[j]
                                   for i in range(self.n_assets)
                                   for j in range(self.n_assets))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = expected_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Store solution
            self.solution = {
                'weights': weights,
                'expected_return': expected_return,
                'portfolio_variance': portfolio_variance,
                'portfolio_volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'objective_value': self.model.ObjVal,
                'solver_status': self.model.status
            }
            
            self.status = "solved"
            return self.solution
        else:
            self.status = "failed"
            raise ValueError(f"Optimization failed with status {self.model.status}")
    
    def get_efficient_frontier(self, min_return=None, max_return=None, n_points=20):
        """
        Calculate the efficient frontier.
        
        Parameters:
        -----------
        min_return : float, optional
            Minimum return for the efficient frontier
        max_return : float, optional
            Maximum return for the efficient frontier
        n_points : int, optional
            Number of points on the efficient frontier
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns for return, risk, and sharpe ratio
        """
        # Store original objective
        original_obj = None
        if self.model is not None:
            original_obj = self.model.getObjective()
            
        # Build model if not already built
        if self.model is None:
            self.build_model()
            
        # Determine min and max returns if not provided
        if min_return is None or max_return is None:
            # Save original objective
            original_obj = self.model.getObjective()
            
            # Find min variance portfolio
            self.model.setObjective(self.portfolio_risk_expr, GRB.MINIMIZE)
            self.model.optimize()
            if self.model.status == GRB.OPTIMAL:
                min_var_return = self.portfolio_return_expr.getValue()
            else:
                min_var_return = min(self.expected_returns)
                
            # Find max return portfolio
            self.model.setObjective(self.portfolio_return_expr, GRB.MAXIMIZE)
            self.model.optimize()
            if self.model.status == GRB.OPTIMAL:
                max_ret = self.portfolio_return_expr.getValue()
            else:
                max_ret = max(self.expected_returns)
                
            min_return = min_return if min_return is not None else min_var_return
            max_return = max_return if max_return is not None else max_ret
            
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        # Create return constraint with placeholder value
        return_constr = self.model.addConstr(
            self.portfolio_return_expr >= 0, 
            "return_constraint"
        )
        
        # Set objective to minimize variance
        self.model.setObjective(self.portfolio_risk_expr, GRB.MINIMIZE)
        
        # Compute efficient frontier
        frontier_points = []
        
        for target_ret in target_returns:
            # Update return constraint
            return_constr.rhs = target_ret
            
            # Solve model
            self.model.optimize()
            
            # Check if a solution was found
            if self.model.status == GRB.OPTIMAL:
                # Extract optimal weights
                weights = [self.weight_vars[i].X for i in range(self.n_assets)]
                
                # Calculate portfolio variance
                portfolio_variance = sum(weights[i] * self.covariance_matrix[i, j] * weights[j]
                                       for i in range(self.n_assets)
                                       for j in range(self.n_assets))
                                       
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = target_ret / portfolio_volatility if portfolio_volatility > 0 else 0
                
                frontier_points.append({
                    'return': target_ret,
                    'risk': portfolio_volatility,
                    'sharpe': sharpe_ratio,
                    'weights': weights
                })
        
        # Remove return constraint
        self.model.remove(return_constr)
        
        # Restore original objective if it exists
        if original_obj is not None:
            self.model.setObjective(original_obj)
            
        # Convert to DataFrame
        if frontier_points:
            df = pd.DataFrame(frontier_points)
            # Add asset weight columns
            for i, asset in enumerate(self.asset_names):
                df[f'weight_{asset}'] = df['weights'].apply(lambda w: w[i])
            # Drop the weights column (list of arrays)
            df = df.drop('weights', axis=1)
            return df
        else:
            return pd.DataFrame(columns=['return', 'risk', 'sharpe'])