# models/constraints.py
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from .mean_variance import MeanVarianceModel

class ConstrainedMeanVarianceModel(MeanVarianceModel):
    """
    Mean-variance model with additional linear constraints like:
    - Individual asset weight bounds
    - Sector allocation constraints
    - Group constraints
    """
    
    def __init__(self, expected_returns, covariance_matrix, asset_names=None, risk_aversion=1.0):
        """
        Initialize the constrained mean-variance model.
        
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
        self.model_type = "constrained-mean-variance"
        self.constraints_added = {
            'asset_bounds': False,
            'sector_constraints': False,
            'target_return': False,
            'target_risk': False,
            'group_constraints': False
        }
    
    def add_asset_bounds(self, min_weights=None, max_weights=None):
        """
        Add minimum and maximum weight constraints for individual assets.
        
        Parameters:
        -----------
        min_weights : dict, list, or float, optional
            Minimum weights for assets. Can be:
            - A float (same min weight for all assets)
            - A list (one min weight per asset)
            - A dict (mapping asset names or indices to min weights)
        max_weights : dict, list, or float, optional
            Maximum weights for assets, in the same format as min_weights
            
        Returns:
        --------
        self : ConstrainedMeanVarianceModel
            The model instance (for method chaining)
        """
        # Ensure model is built
        if self.model is None:
            self.build_model()
            
        # Process min_weights
        if min_weights is not None:
            if isinstance(min_weights, (int, float)):
                # Single value for all assets
                for i in range(self.n_assets):
                    self.model.addConstr(
                        self.weight_vars[i] >= min_weights, 
                        f"min_weight_{i}"
                    )
            elif isinstance(min_weights, list):
                # List with one value per asset
                for i, min_w in enumerate(min_weights):
                    if min_w is not None:  # Skip None values
                        self.model.addConstr(
                            self.weight_vars[i] >= min_w, 
                            f"min_weight_{i}"
                        )
            elif isinstance(min_weights, dict):
                # Dict mapping asset indices or names to values
                for key, min_w in min_weights.items():
                    idx = key if isinstance(key, int) else self.asset_names.index(key)
                    self.model.addConstr(
                        self.weight_vars[idx] >= min_w, 
                        f"min_weight_{idx}"
                    )
        
        # Process max_weights (similar to min_weights)
        if max_weights is not None:
            if isinstance(max_weights, (int, float)):
                for i in range(self.n_assets):
                    self.model.addConstr(
                        self.weight_vars[i] <= max_weights, 
                        f"max_weight_{i}"
                    )
            elif isinstance(max_weights, list):
                for i, max_w in enumerate(max_weights):
                    if max_w is not None:  # Skip None values
                        self.model.addConstr(
                            self.weight_vars[i] <= max_w, 
                            f"max_weight_{i}"
                        )
            elif isinstance(max_weights, dict):
                for key, max_w in max_weights.items():
                    idx = key if isinstance(key, int) else self.asset_names.index(key)
                    self.model.addConstr(
                        self.weight_vars[idx] <= max_w, 
                        f"max_weight_{idx}"
                    )
                    
        self.constraints_added['asset_bounds'] = True
        return self
    
    def add_sector_constraints(self, sector_mapping, sector_min=None, sector_max=None):
        """
        Add sector allocation constraints.
        
        Parameters:
        -----------
        sector_mapping : dict or list
            Mapping of assets to sectors. Can be:
            - A list (sector for each asset)
            - A dict (mapping asset names or indices to sectors)
        sector_min : dict, optional
            Minimum allocation per sector {sector_name: min_allocation}
        sector_max : dict, optional
            Maximum allocation per sector {sector_name: max_allocation}
            
        Returns:
        --------
        self : ConstrainedMeanVarianceModel
            The model instance (for method chaining)
        """
        # Ensure model is built
        if self.model is None:
            self.build_model()
            
        # Standardize sector_mapping to a list
        if isinstance(sector_mapping, dict):
            # Convert dict to list
            mapping = [None] * self.n_assets
            for key, sector in sector_mapping.items():
                idx = key if isinstance(key, int) else self.asset_names.index(key)
                mapping[idx] = sector
            sector_mapping = mapping
            
        # Get unique sectors
        sectors = set(filter(None, sector_mapping))  # Filter out None values
        
        # Initialize default min/max if not provided
        sector_min = sector_min or {}
        sector_max = sector_max or {}
        
        # Add sector constraints
        for sector in sectors:
            # Get indices of assets in this sector
            sector_indices = [i for i, s in enumerate(sector_mapping) if s == sector]
            
            # Skip empty sectors
            if not sector_indices:
                continue
                
            # Calculate sector weight expression
            sector_weight = gp.quicksum(self.weight_vars[i] for i in sector_indices)
            
            # Add min constraint if specified
            if sector in sector_min:
                self.model.addConstr(
                    sector_weight >= sector_min[sector],
                    f"min_sector_{sector}"
                )
                
            # Add max constraint if specified
            if sector in sector_max:
                self.model.addConstr(
                    sector_weight <= sector_max[sector],
                    f"max_sector_{sector}"
                )
                
        self.constraints_added['sector_constraints'] = True
        return self
        
    def add_group_constraints(self, groups, group_min=None, group_max=None):
        """
        Add constraints on groups of assets (more general than sectors).
        
        Parameters:
        -----------
        groups : dict
            Dictionary where keys are group names and values are lists of asset indices or names
        group_min : dict, optional
            Minimum allocation per group {group_name: min_allocation}
        group_max : dict, optional
            Maximum allocation per group {group_name: max_allocation}
            
        Returns:
        --------
        self : ConstrainedMeanVarianceModel
            The model instance (for method chaining)
        """
        # Ensure model is built
        if self.model is None:
            self.build_model()
            
        # Initialize default min/max if not provided
        group_min = group_min or {}
        group_max = group_max or {}
        
        # Add group constraints
        for group_name, assets in groups.items():
            # Convert asset names to indices if needed
            indices = []
            for asset in assets:
                if isinstance(asset, str):
                    indices.append(self.asset_names.index(asset))
                else:
                    indices.append(asset)
                    
            # Calculate group weight expression
            group_weight = gp.quicksum(self.weight_vars[i] for i in indices)
            
            # Add min constraint if specified
            if group_name in group_min:
                self.model.addConstr(
                    group_weight >= group_min[group_name],
                    f"min_group_{group_name}"
                )
                
            # Add max constraint if specified
            if group_name in group_max:
                self.model.addConstr(
                    group_weight <= group_max[group_name],
                    f"max_group_{group_name}"
                )
                
        self.constraints_added['group_constraints'] = True
        return self
    
    def add_target_return(self, target_return):
        """
        Add a target return constraint.
        
        Parameters:
        -----------
        target_return : float
            Target portfolio return
            
        Returns:
        --------
        self : ConstrainedMeanVarianceModel
            The model instance (for method chaining)
        """
        # Ensure model is built
        if self.model is None:
            self.build_model()
            
        # Add target return constraint
        self.model.addConstr(
            self.portfolio_return_expr >= target_return,
            "target_return_constraint"
        )
        
        self.constraints_added['target_return'] = True
        return self
    
    def add_target_risk(self, max_risk):
        """
        Add a maximum risk/volatility constraint.
        
        Parameters:
        -----------
        max_risk : float
            Maximum portfolio volatility
            
        Returns:
        --------
        self : ConstrainedMeanVarianceModel
            The model instance (for method chaining)
        """
        # Ensure model is built
        if self.model is None:
            self.build_model()
            
        # Add risk constraint (risk is the square of volatility)
        max_variance = max_risk ** 2
        self.model.addConstr(
            self.portfolio_risk_expr <= max_variance,
            "max_risk_constraint"
        )
        
        self.constraints_added['target_risk'] = True
        return self
    
    def solve(self, verbose=False, time_limit=None):
        """
        Solve the model and return the solution.
        Overrides parent solve method to include constraint info in solution.
        
        Parameters:
        -----------
        verbose : bool, optional
            If True, print solver output
        time_limit : int, optional
            Time limit in seconds for solver
            
        Returns:
        --------
        dict
            Solution details including constraint status
        """
        # Call parent solve method
        solution = super().solve(verbose, time_limit)
        
        # Add constraint information to solution
        solution['constraints_added'] = self.constraints_added.copy()
        
        return solution