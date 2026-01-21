import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.linalg import sqrtm
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.covariance import LedoitWolf

class AdvancedPortfolioAnalytics:
    """
    Advanced quantitative finance algorithms for portfolio optimization and risk management.
    Implements modern portfolio theory, stochastic models, and advanced risk metrics.
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    # ==================== PORTFOLIO OPTIMIZATION ====================
    
    def markowitz_optimization(
        self, 
        returns: np.ndarray, 
        target_return: Optional[float] = None,
        constraints: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Markowitz Mean-Variance Optimization
        Finds optimal portfolio weights to minimize variance for a given return
        or maximize Sharpe ratio.
        
        Args:
            returns: Historical returns matrix (n_periods x n_assets)
            target_return: Target portfolio return (optional)
            constraints: Additional constraints (min/max weights, sector limits)
        
        Returns:
            Optimal weights, expected return, volatility, Sharpe ratio
        """
        n_assets = returns.shape[1]
        mean_returns = np.mean(returns, axis=0)
        
        # Use Ledoit-Wolf shrinkage estimator for more stable covariance
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns).covariance_
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Portfolio return and risk
        portfolio_return = mean_returns @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0  # No short selling (can be modified)
        ]
        
        # Add custom constraints
        if constraints:
            if 'min_weight' in constraints:
                constraints_list.append(weights >= constraints['min_weight'])
            if 'max_weight' in constraints:
                constraints_list.append(weights <= constraints['max_weight'])
        
        if target_return is not None:
            # Minimize variance for target return
            constraints_list.append(portfolio_return >= target_return)
            objective = cp.Minimize(portfolio_variance)
        else:
            # Maximize Sharpe ratio (equivalent to maximizing return/risk)
            objective = cp.Maximize(portfolio_return - 0.5 * portfolio_variance)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if weights.value is None:
            return None
        
        optimal_weights = weights.value
        optimal_return = float(mean_returns @ optimal_weights)
        optimal_volatility = float(np.sqrt(optimal_weights @ cov_matrix @ optimal_weights))
        sharpe_ratio = (optimal_return - self.risk_free_rate / 252) / optimal_volatility if optimal_volatility > 0 else 0
        
        return {
            'weights': optimal_weights,
            'expected_return': optimal_return * 252,  # Annualized
            'volatility': optimal_volatility * np.sqrt(252),  # Annualized
            'sharpe_ratio': sharpe_ratio * np.sqrt(252)
        }
    
    def black_litterman_optimization(
        self,
        returns: np.ndarray,
        market_caps: np.ndarray,
        views: Optional[Dict[int, float]] = None,
        view_confidence: float = 0.5
    ) -> Dict[str, any]:
        """
        Black-Litterman Model
        Combines market equilibrium with investor views to generate
        optimal portfolio allocation.
        
        Args:
            returns: Historical returns
            market_caps: Market capitalizations for equilibrium weights
            views: Dict of {asset_index: expected_return}
            view_confidence: Confidence in views (0-1)
        """
        n_assets = returns.shape[1]
        
        # Market equilibrium weights (proportional to market cap)
        market_weights = market_caps / np.sum(market_caps)
        
        # Covariance matrix
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns).covariance_
        
        # Implied equilibrium returns (reverse optimization)
        risk_aversion = 2.5  # Typical value
        pi = risk_aversion * cov_matrix @ market_weights
        
        if views is None or len(views) == 0:
            # No views, return market portfolio
            posterior_returns = pi
        else:
            # Incorporate views using Bayesian updating
            k = len(views)
            P = np.zeros((k, n_assets))
            Q = np.zeros(k)
            
            for i, (asset_idx, view_return) in enumerate(views.items()):
                P[i, asset_idx] = 1
                Q[i] = view_return
            
            # View uncertainty (tau * P * Sigma * P')
            tau = view_confidence
            omega = tau * P @ cov_matrix @ P.T
            
            # Black-Litterman formula
            M_inv = np.linalg.inv(tau * cov_matrix)
            posterior_cov_inv = M_inv + P.T @ np.linalg.inv(omega) @ P
            posterior_cov = np.linalg.inv(posterior_cov_inv)
            posterior_returns = posterior_cov @ (M_inv @ pi + P.T @ np.linalg.inv(omega) @ Q)
        
        # Optimize with posterior returns
        weights = cp.Variable(n_assets)
        portfolio_return = posterior_returns @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        objective = cp.Maximize(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        constraints = [cp.sum(weights) == 1, weights >= 0]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return {
            'weights': weights.value,
            'posterior_returns': posterior_returns * 252,
            'expected_return': float(posterior_returns @ weights.value) * 252,
            'volatility': float(np.sqrt(weights.value @ cov_matrix @ weights.value)) * np.sqrt(252)
        }
    
    # ==================== STOCHASTIC MODELS ====================
    
    def monte_carlo_simulation(
        self,
        initial_price: float,
        mu: float,
        sigma: float,
        days: int = 252,
        simulations: int = 10000
    ) -> Dict[str, any]:
        """
        Monte Carlo simulation using Geometric Brownian Motion
        
        dS = μS dt + σS dW
        
        Args:
            initial_price: Starting price
            mu: Expected return (drift)
            sigma: Volatility
            days: Number of days to simulate
            simulations: Number of simulation paths
        """
        dt = 1/252  # Daily time step
        
        # Generate random walks
        random_shocks = np.random.normal(0, 1, (simulations, days))
        
        # Geometric Brownian Motion
        price_paths = np.zeros((simulations, days + 1))
        price_paths[:, 0] = initial_price
        
        for t in range(1, days + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks[:, t-1]
            )
        
        # Calculate statistics
        final_prices = price_paths[:, -1]
        
        return {
            'paths': price_paths,
            'mean_final_price': np.mean(final_prices),
            'median_final_price': np.median(final_prices),
            'std_final_price': np.std(final_prices),
            'percentile_5': np.percentile(final_prices, 5),
            'percentile_95': np.percentile(final_prices, 95),
            'probability_profit': np.mean(final_prices > initial_price)
        }
    
    def heston_model_simulation(
        self,
        S0: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        T: float = 1.0,
        steps: int = 252,
        simulations: int = 10000
    ) -> np.ndarray:
        """
        Heston Stochastic Volatility Model
        
        dS = μS dt + √v S dW1
        dv = κ(θ - v) dt + σ√v dW2
        
        where dW1 and dW2 are correlated with correlation ρ
        """
        dt = T / steps
        
        # Initialize arrays
        S = np.zeros((simulations, steps + 1))
        v = np.zeros((simulations, steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        
        # Correlated random numbers
        for t in range(steps):
            Z1 = np.random.normal(0, 1, simulations)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, simulations)
            
            # Ensure variance stays positive (Milstein scheme)
            v[:, t+1] = np.maximum(
                v[:, t] + kappa * (theta - v[:, t]) * dt + sigma * np.sqrt(v[:, t] * dt) * Z2,
                0
            )
            
            S[:, t+1] = S[:, t] * np.exp(
                (self.risk_free_rate - 0.5 * v[:, t]) * dt + np.sqrt(v[:, t] * dt) * Z1
            )
        
        return S
    
    # ==================== ADVANCED RISK METRICS ====================
    
    def conditional_value_at_risk(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Conditional Value at Risk (CVaR) / Expected Shortfall
        Expected loss given that loss exceeds VaR
        
        More coherent risk measure than VaR
        """
        var = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = returns[returns <= var].mean()
        return float(cvar)
    
    def expected_shortfall(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Expected Shortfall with multiple confidence levels
        """
        levels = [0.90, 0.95, 0.99]
        results = {}
        
        for level in levels:
            var = np.percentile(returns, (1 - level) * 100)
            es = returns[returns <= var].mean()
            results[f'ES_{int(level*100)}'] = float(es)
        
        return results
    
    def maximum_drawdown_duration(self, prices: np.ndarray) -> Dict[str, any]:
        """
        Calculate maximum drawdown and its duration
        """
        cumulative = np.cumprod(1 + prices)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Find duration
        peak_idx = np.argmax(cumulative[:max_dd_idx]) if max_dd_idx > 0 else 0
        duration = max_dd_idx - peak_idx
        
        return {
            'max_drawdown': float(max_dd),
            'max_drawdown_duration': int(duration),
            'peak_date_index': int(peak_idx),
            'trough_date_index': int(max_dd_idx)
        }
    
    def downside_deviation(
        self,
        returns: np.ndarray,
        target_return: float = 0.0
    ) -> float:
        """
        Downside deviation (semi-deviation)
        Only considers returns below target
        """
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0.0
        return float(np.sqrt(np.mean((downside_returns - target_return)**2)))
    
    def sortino_ratio(
        self,
        returns: np.ndarray,
        target_return: float = 0.0
    ) -> float:
        """
        Sortino Ratio - risk-adjusted return using downside deviation
        Better than Sharpe for asymmetric returns
        """
        excess_return = np.mean(returns) - target_return
        downside_dev = self.downside_deviation(returns, target_return)
        
        if downside_dev == 0:
            return 0.0
        
        return float(excess_return / downside_dev * np.sqrt(252))
    
    def omega_ratio(
        self,
        returns: np.ndarray,
        threshold: float = 0.0
    ) -> float:
        """
        Omega Ratio - probability weighted ratio of gains vs losses
        """
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if len(losses) == 0 or np.sum(losses) == 0:
            return np.inf
        
        return float(np.sum(gains) / np.sum(losses))
    
    # ==================== TIME SERIES ANALYSIS ====================
    
    def garch_volatility_forecast(
        self,
        returns: np.ndarray,
        horizon: int = 5
    ) -> Dict[str, any]:
        """
        GARCH(1,1) model for volatility forecasting
        
        σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
        """
        # Fit GARCH(1,1) model
        model = arch_model(returns * 100, vol='Garch', p=1, q=1)
        results = model.fit(disp='off')
        
        # Forecast volatility
        forecast = results.forecast(horizon=horizon)
        
        return {
            'current_volatility': float(np.sqrt(results.conditional_volatility[-1]) / 100),
            'forecast_volatility': forecast.variance.values[-1, :] / 10000,
            'omega': float(results.params['omega']),
            'alpha': float(results.params['alpha[1]']),
            'beta': float(results.params['beta[1]']),
            'persistence': float(results.params['alpha[1]'] + results.params['beta[1]'])
        }
    
    def kalman_filter_estimate(
        self,
        prices: np.ndarray,
        observation_covariance: float = 1.0,
        transition_covariance: float = 0.1
    ) -> np.ndarray:
        """
        Kalman Filter for state estimation and noise reduction
        Useful for estimating true price from noisy observations
        """
        from pykalman import KalmanFilter
        
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=prices[0],
            initial_state_covariance=1,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance
        )
        
        state_means, _ = kf.filter(prices)
        return state_means.flatten()
    
    # ==================== FACTOR MODELS ====================
    
    def fama_french_three_factor(
        self,
        returns: np.ndarray,
        market_returns: np.ndarray,
        smb: np.ndarray,  # Small Minus Big
        hml: np.ndarray   # High Minus Low
    ) -> Dict[str, float]:
        """
        Fama-French Three-Factor Model
        
        R_i - R_f = α + β_1(R_m - R_f) + β_2*SMB + β_3*HML + ε
        """
        from sklearn.linear_model import LinearRegression
        
        # Prepare data
        X = np.column_stack([market_returns, smb, hml])
        y = returns
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R-squared
        r_squared = model.score(X, y)
        
        return {
            'alpha': float(model.intercept_ * 252),  # Annualized
            'beta_market': float(model.coef_[0]),
            'beta_smb': float(model.coef_[1]),
            'beta_hml': float(model.coef_[2]),
            'r_squared': float(r_squared)
        }
    
    def principal_component_analysis(
        self,
        returns: np.ndarray,
        n_components: int = 3
    ) -> Dict[str, any]:
        """
        PCA for factor extraction and dimensionality reduction
        Identifies main drivers of portfolio variance
        """
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(returns)
        
        return {
            'factors': factors,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_
        }
    
    # ==================== REGIME DETECTION ====================
    
    def hidden_markov_model(
        self,
        returns: np.ndarray,
        n_states: int = 2
    ) -> Dict[str, any]:
        """
        Hidden Markov Model for market regime detection
        Identifies bull/bear markets or high/low volatility regimes
        """
        from hmmlearn import hmm
        
        # Reshape for HMM
        X = returns.reshape(-1, 1)
        
        # Fit Gaussian HMM
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
        model.fit(X)
        
        # Predict states
        states = model.predict(X)
        
        # Calculate regime statistics
        regime_stats = {}
        for state in range(n_states):
            state_returns = returns[states == state]
            regime_stats[f'regime_{state}'] = {
                'mean_return': float(np.mean(state_returns)),
                'volatility': float(np.std(state_returns)),
                'probability': float(np.mean(states == state))
            }
        
        return {
            'states': states,
            'transition_matrix': model.transmat_,
            'regime_stats': regime_stats,
            'current_regime': int(states[-1])
        }

advanced_analytics = AdvancedPortfolioAnalytics()
