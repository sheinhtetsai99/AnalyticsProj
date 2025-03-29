# models/integer_model.py
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from .constraints import ConstrainedMeanVarianceModel

class IntegerConstrainedModel(ConstrainedMeanVarianceModel):
    """
    Portfolio optimization model with integer constraints like:
    - Cardinality constraints (maximum number of assets)
    - Minimum position size constraints
    - Buy-in thresholds
    """
    
    def __init__(self, expected_returns, covariance_matrix, asset_names=None, risk_aversion=1.0):
        """
        Initialize the integer-constrained model.
        
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
        super().__init__(expected_returns, covariance_matrix, asset_names, risk_aversion)
        self.model_type = "integer-constrained"
        self.selection_vars = None
        self.integer_constraints_added = {
            'cardinality': False,
            'min_position_size': False,
            'buy_in_threshold': False,
            'diversification': False
        }
    
    def build_model(self):
        """
        Build the base model and add binary selection variables.
        
        Returns:
        --------
        self : IntegerConstrainedModel
            The model instance (for method chaining)
        """
        # First build the base mean-variance model
        super().build_model()
        
        # Add binary selection variables
        self.selection_vars = self.model.addVars(
            self.n_assets, 
            vtype=GRB.BINARY, 
            name="asset_selection"
        )
        
        return self
    
    def add_cardinality_constraint(self, max_assets):
        """
        Add a constraint on the maximum number of assets.
        
        Parameters:
        -----------
        max_assets : int
            Maximum number of assets to include in the portfolio
            
        Returns:
        --------
        self : IntegerConstrainedModel
            The model instance (for method chaining)
        """
        # Ensure model is built
        if self.model is None or self.selection_vars is None:
            self.build_model()
            
        # Add cardinality constraint
        self.model.addConstr(
            gp.quicksum(self.selection_vars[i] for i in range(self.n_assets)) <= max_assets,
            "cardinality_constraint"
        )
        
        self.integer_constraints_added['cardinality'] = True
        return self
    
    def add_min_position_size(self, min_position):
        """
        Add minimum position size constraints.
        
        Parameters:
        -----------
        min_position : float
            Minimum position size for an asset if it's included
            
        Returns:
        --------
        self : IntegerConstrainedModel
            The model instance (for method chaining)
        """
        # Ensure model is built
        if self.model is None or self.selection_vars is None:
            self.build_model()
            
        # Link weight and selection variables
        for i in range(self.n_assets):
            # If asset is selected, weight must be at least min_position
            self.model.addConstr(
                self.weight_vars[i] >= min_position * self.selection_vars[i],
                f"min_position_{i}"
            )
            
            # If asset is not selected, weight must be zero
            self.model.addConstr(
                self.weight_vars[i] <= self.selection_vars[i],
                f"link_weight_selection_{i}"
            )
            
        self.integer_constraints_added['min_position_size'] = True
        return self
    
    def add_buy_in_threshold(self, threshold):
        """
        Add buy-in threshold constraints without minimum position size.
        
        Parameters:
        -----------
        threshold : float
            Maximum weight an asset can have if it's not fully included
            
        Returns:
        --------
        self : IntegerConstrainedModel
            The model instance (for method chaining)
        """
        # Ensure model is built
        if self.model is None or self.selection_vars is None:
            self.build_model()
            
        # Add buy-in threshold constraints
        for i in range(self.n_assets):
            self.model.addConstr(
                self.weight_vars[i] <= threshold + (1 - threshold) * self.selection_vars[i],
                f"buy_in_threshold_{i}"
            )
            
        self.integer_constraints_added['buy_in_threshold'] = True
        return self
    
    def add_diversification_constraint(self, min_assets):
        """
        Add a constraint on the minimum number of assets.
        
        Parameters:
        -----------
        min_assets : int
            Minimum number of assets to include in the portfolio
            
        Returns:
        --------
        self : IntegerConstrainedModel
            The model instance (for method chaining)
        """
        # Ensure model is built
        if self.model is None or self.selection_vars is None:
            self.build_model()
            
        # Add diversification constraint
        self.model.addConstr(
            gp.quicksum(self.selection_vars[i] for i in range(self.n_assets)) >= min_assets,
            "diversification_constraint"
        )
        
        self.integer_constraints_added['diversification'] = True
        return self
    
    def add_pre_selected_assets(self, assets):
        """
        Force specific assets to be included in the portfolio.
        
        Parameters:
        -----------
        assets : list
            List of asset names or indices to include
            
        Returns:
        --------
        self : IntegerConstrainedModel
            The model instance (for method chaining)
        """
        # Ensure model is built
        if self.model is None or self.selection_vars is None:
            self.build_model()
            
        # Convert asset names to indices if needed
        indices = []
        for asset in assets:
            if isinstance(asset, str):
                indices.append(self.asset_names.index(asset))
            else:
                indices.append(asset)
                
        # Force selection of specified assets
        for idx in indices:
            self.model.addConstr(
                self.selection_vars[idx] == 1,
                f"pre_select_{idx}"
            )
            
        return self
    
    def solve(self, verbose=False, time_limit=None, gap=0.01):
        """
        Solve the model and return the solution.
        
        Parameters:
        -----------
        verbose : bool, optional
            If True, print solver output
        time_limit : int, optional
            Time limit in seconds for solver
        gap : float, optional
            Relative MIP gap tolerance
            
        Returns:
        --------
        dict
            Solution details including integer constraint info
        """
        # Ensure model is built
        if self.model is None:
            self.build_model()
            
        # Set verbosity
        self.model.setParam('OutputFlag', 1 if verbose else 0)
        
        # Set time limit if specified
        if time_limit is not None:
            self.model.setParam('TimeLimit', time_limit)
            
        # Set MIP gap
        self.model.setParam('MIPGap', gap)
        
        # Solve model
        self.model.optimize()
        
        # Check if a solution was found
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            # Extract optimal weights
            weights = [self.weight_vars[i].X for i in range(self.n_assets)]
            
            # Extract selected assets
            selected = [i for i in range(self.n_assets) if self.selection_vars[i].X > 0.5]
            selected_names = [self.asset_names[i] for i in selected]
            
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
                'solver_status': self.model.status,
                'selected_assets': selected,
                'selected_asset_names': selected_names,
                'num_assets_selected': len(selected),
                'integer_constraints_added': self.integer_constraints_added.copy(),
                'mip_gap': self.model.MIPGap
            }
            
            # Add constraint information to solution
            if hasattr(self, 'constraints_added'):
                self.solution['constraints_added'] = self.constraints_added.copy()
            
            self.status = "solved"
            return self.solution
        else:
            self.status = "failed"
            raise ValueError(f"Optimization failed with status {self.model.status}")