import numpy as np
import pandas as pd
import cvxpy as cp

def mean_variance_optimization(returns_df, target_return=None):
    """
    Perform Markowitz mean-variance optimization.
    returns_df: DataFrame of historical returns (each column is a stock).
    target_return: If specified, find the portfolio with at least that return.
    """
    cov_matrix = returns_df.cov()
    mean_returns = returns_df.mean()
    num_assets = len(mean_returns)
    
    cov_mat = cov_matrix.values
    mean_vec = mean_returns.values
    
    # Define optimization variables
    w = cp.Variable(num_assets, nonneg=True)
    portfolio_variance = cp.quad_form(w, cov_mat)
    
    objective = cp.Minimize(portfolio_variance)
    constraints = [cp.sum(w) == 1]  # Weights sum to 1 (fully invested)
    if target_return is not None:
        constraints.append(mean_vec @ w >= target_return)
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return w.value

if __name__ == '__main__':
    # Demo with random returns
    np.random.seed(42)
    dummy_returns = pd.DataFrame(np.random.randn(100, 4) / 100,
                                 columns=['AAPL', 'MSFT', 'GOOG', 'TSLA'])
    weights = mean_variance_optimization(dummy_returns, target_return=0.001)
    print("Optimized Weights:", weights)
