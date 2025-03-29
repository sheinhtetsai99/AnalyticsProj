# models/base_model.py
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import gurobipy as gp
from gurobipy import GRB

class BasePortfolioModel(ABC):
    """
    Abstract base class for all portfolio optimization models.
    Defines the common interface for all models.
    """
    
    def __init__(self, expected_returns, covariance_matrix, asset_names=None):
        """
        Initialize a portfolio optimization model.
        
        Parameters:
        -----------
        expected_returns : np.ndarray or pd.Series
            Expected returns for each asset
        covariance_matrix : np.ndarray or pd.DataFrame
            Covariance matrix of asset returns
        asset_names : list, optional
            Names of assets. If None, generic names are created
        """
        # Convert Series/DataFrame to numpy arrays
        if isinstance(expected_returns, pd.Series):
            self.asset_names = expected_returns.index.tolist() if asset_names is None else asset_names
            self.expected_returns = expected_returns.values
        else:
            self.expected_returns = expected_returns
            self.asset_names = asset_names if asset_names is not None else [
                f"Asset_{i}" for i in range(len(expected_returns))
            ]
            
        if isinstance(covariance_matrix, pd.DataFrame):
            self.covariance_matrix = covariance_matrix.values
        else:
            self.covariance_matrix = covariance_matrix
            
        # Validate dimensions
        if len(self.expected_returns) != len(self.asset_names):
            raise ValueError("Length of expected_returns must match the number of asset_names")
        if self.covariance_matrix.shape[0] != len(self.expected_returns) or \
           self.covariance_matrix.shape[1] != len(self.expected_returns):
            raise ValueError("Dimensions of covariance_matrix must match the length of expected_returns")
            
        # Store the number of assets
        self.n_assets = len(self.expected_returns)
        
        # Initialize model
        self.model = None
        self.weight_vars = None
        self.created_date = pd.Timestamp.now()
        self.status = "initialized"
        
        # Dictionary to store solution
        self.solution = None
    
    @abstractmethod
    def build_model(self):
        """
        Build the optimization model.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def solve(self, verbose=False):
        """
        Solve the optimization model.
        Must be implemented by subclasses.
        
        Parameters:
        -----------
        verbose : bool, optional
            If True, print solver output
            
        Returns:
        --------
        dict
            Solution details
        """
        pass
    
    def get_weights(self):
        """
        Get the optimal portfolio weights.
        
        Returns:
        --------
        pd.Series
            Optimal weights for each asset
        """
        if self.solution is None:
            raise ValueError("Model has not been solved yet")
            
        return pd.Series(self.solution['weights'], index=self.asset_names)
    
    def get_performance_metrics(self):
        """
        Get performance metrics for the optimal portfolio.
        
        Returns:
        --------
        dict
            Performance metrics
        """
        if self.solution is None:
            raise ValueError("Model has not been solved yet")
            
        return {k: v for k, v in self.solution.items() if k != 'weights'}
    
    def export_solution(self, filepath):
        """
        Export the solution to a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the CSV file
        """
        if self.solution is None:
            raise ValueError("Model has not been solved yet")
            
        weights_df = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': self.solution['weights']
        })
        
        weights_df.to_csv(filepath, index=False)
        print(f"Solution exported to {filepath}")
    
    def __str__(self):
        """String representation of the model"""
        model_type = self.__class__.__name__
        status = self.status
        n_assets = self.n_assets
        
        if self.solution is not None:
            exp_return = self.solution.get('expected_return', 'N/A')
            volatility = self.solution.get('portfolio_volatility', 'N/A')
            return f"{model_type} with {n_assets} assets: Status={status}, Return={exp_return}, Risk={volatility}"
        else:
            return f"{model_type} with {n_assets} assets: Status={status}, Not solved"