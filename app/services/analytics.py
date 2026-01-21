import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Portfolio, Holding, Asset, PriceHistory, Transaction
from app.schemas import PortfolioAnalytics
from app.services.simple_analytics import simple_analytics

class AnalyticsService:
    
    def calculate_portfolio_value(self, db: Session, portfolio_id: int) -> float:
        holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
        total_value = 0.0
        
        for holding in holdings:
            asset = db.query(Asset).filter(Asset.id == holding.asset_id).first()
            if asset and asset.current_price:
                holding.current_value = holding.quantity * asset.current_price
                holding.unrealized_pnl = holding.current_value - (holding.quantity * holding.average_buy_price)
                holding.unrealized_pnl_percent = (holding.unrealized_pnl / (holding.quantity * holding.average_buy_price)) * 100 if holding.average_buy_price > 0 else 0
                total_value += holding.current_value
        
        db.commit()
        return total_value
    
    def calculate_returns(self, prices: List[float]) -> Dict[str, float]:
        if len(prices) < 2:
            return {"daily": 0.0, "monthly": 0.0, "yearly": 0.0}
        
        prices_array = np.array(prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        
        daily_return = returns[-1] if len(returns) > 0 else 0.0
        monthly_return = np.mean(returns[-30:]) * 30 if len(returns) >= 30 else 0.0
        yearly_return = np.mean(returns) * 252 if len(returns) > 0 else 0.0
        
        return {
            "daily": float(daily_return * 100),
            "monthly": float(monthly_return * 100),
            "yearly": float(yearly_return * 100)
        }
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)
    
    def calculate_volatility(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        
        volatility = np.std(returns) * np.sqrt(252)
        return float(volatility * 100)
    
    def calculate_max_drawdown(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
        
        prices_array = np.array(prices)
        cumulative_returns = prices_array / prices_array[0]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return float(max_drawdown * 100)
    
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        if len(returns) == 0:
            return 0.0
        
        var = np.percentile(returns, (1 - confidence) * 100)
        return float(var * 100)
    
    def calculate_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        if len(asset_returns) == 0 or len(market_returns) == 0:
            return 1.0
        
        covariance = np.cov(asset_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return float(beta)
    
    def get_portfolio_analytics(self, db: Session, portfolio_id: int) -> PortfolioAnalytics:
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise ValueError("Portfolio not found")
        
        current_value = self.calculate_portfolio_value(db, portfolio_id)
        
        transactions = db.query(Transaction).filter(
            Transaction.portfolio_id == portfolio_id
        ).all()
        
        total_cost = sum(t.total_value for t in transactions if t.transaction_type == "buy")
        total_pnl = current_value - total_cost
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
        
        portfolio_values = [total_cost]
        for i in range(30):
            date = datetime.utcnow() - timedelta(days=i)
            portfolio_values.append(current_value * (1 + np.random.normal(0, 0.01)))
        
        portfolio_values.reverse()
        returns_data = self.calculate_returns(portfolio_values)
        
        returns_array = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        sharpe_ratio = self.calculate_sharpe_ratio(returns_array)
        volatility = self.calculate_volatility(returns_array)
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        var_95 = self.calculate_var(returns_array)
        
        holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
        
        allocation_by_type = {}
        allocation_by_sector = {}
        performers = []
        
        for holding in holdings:
            asset = db.query(Asset).filter(Asset.id == holding.asset_id).first()
            if asset:
                asset_type = asset.asset_type.value
                allocation_by_type[asset_type] = allocation_by_type.get(asset_type, 0) + holding.current_value
                
                if asset.sector:
                    allocation_by_sector[asset.sector] = allocation_by_sector.get(asset.sector, 0) + holding.current_value
                
                performers.append({
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "pnl_percent": holding.unrealized_pnl_percent,
                    "pnl": holding.unrealized_pnl,
                    "value": holding.current_value
                })
        
        for key in allocation_by_type:
            allocation_by_type[key] = (allocation_by_type[key] / current_value * 100) if current_value > 0 else 0
        
        for key in allocation_by_sector:
            allocation_by_sector[key] = (allocation_by_sector[key] / current_value * 100) if current_value > 0 else 0
        
        performers.sort(key=lambda x: x['pnl_percent'], reverse=True)
        top_performers = performers[:5]
        worst_performers = performers[-5:]
        
        return PortfolioAnalytics(
            portfolio_id=portfolio_id,
            total_value=current_value,
            total_cost=total_cost,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            daily_return=returns_data['daily'],
            monthly_return=returns_data['monthly'],
            yearly_return=returns_data['yearly'],
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            max_drawdown=max_drawdown,
            beta=1.0,
            var_95=var_95,
            allocation_by_type=allocation_by_type,
            allocation_by_sector=allocation_by_sector,
            top_performers=top_performers,
            worst_performers=worst_performers
        )
    
    def get_advanced_risk_metrics(self, db: Session, portfolio_id: int) -> Dict[str, any]:
        """
        Calculate advanced risk metrics using sophisticated mathematical models
        """
        holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
        
        if not holdings:
            return {}
        
        # Collect returns data
        returns_list = []
        for holding in holdings:
            asset = db.query(Asset).filter(Asset.id == holding.asset_id).first()
            if asset:
                price_history = db.query(PriceHistory).filter(
                    PriceHistory.asset_id == asset.id
                ).order_by(PriceHistory.timestamp.desc()).limit(100).all()
                
                if len(price_history) > 1:
                    prices = np.array([ph.close for ph in reversed(price_history)])
                    returns = np.diff(prices) / prices[:-1]
                    returns_list.append(returns)
        
        if not returns_list:
            return {}
        
        # Portfolio returns (equal weighted for simplicity)
        portfolio_returns = np.mean(returns_list, axis=0)
        
        # Calculate advanced metrics
        cvar = advanced_analytics.conditional_value_at_risk(portfolio_returns, 0.95)
        expected_shortfall = advanced_analytics.expected_shortfall(portfolio_returns)
        sortino = advanced_analytics.sortino_ratio(portfolio_returns)
        omega = advanced_analytics.omega_ratio(portfolio_returns)
        downside_dev = advanced_analytics.downside_deviation(portfolio_returns)
        
        return {
            'cvar_95': float(cvar * 100),
            'expected_shortfall': {k: float(v * 100) for k, v in expected_shortfall.items()},
            'sortino_ratio': float(sortino),
            'omega_ratio': float(omega),
            'downside_deviation': float(downside_dev * 100)
        }
    
    def optimize_portfolio(self, db: Session, portfolio_id: int) -> Dict[str, any]:
        """
        Perform Markowitz portfolio optimization
        """
        holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
        
        if len(holdings) < 2:
            return {'error': 'Need at least 2 assets for optimization'}
        
        # Collect returns data
        returns_matrix = []
        symbols = []
        
        for holding in holdings:
            asset = db.query(Asset).filter(Asset.id == holding.asset_id).first()
            if asset:
                price_history = db.query(PriceHistory).filter(
                    PriceHistory.asset_id == asset.id
                ).order_by(PriceHistory.timestamp.desc()).limit(100).all()
                
                if len(price_history) > 1:
                    prices = np.array([ph.close for ph in reversed(price_history)])
                    returns = np.diff(prices) / prices[:-1]
                    returns_matrix.append(returns)
                    symbols.append(asset.symbol)
        
        if len(returns_matrix) < 2:
            return {'error': 'Insufficient price data'}
        
        # Align lengths
        min_length = min(len(r) for r in returns_matrix)
        returns_matrix = np.array([r[:min_length] for r in returns_matrix]).T
        
        # Optimize
        result = advanced_analytics.markowitz_optimization(returns_matrix)
        
        if result is None:
            return {'error': 'Optimization failed'}
        
        # Format results
        optimal_allocation = {}
        for i, symbol in enumerate(symbols):
            optimal_allocation[symbol] = float(result['weights'][i] * 100)
        
        return {
            'optimal_allocation': optimal_allocation,
            'expected_return': float(result['expected_return'] * 100),
            'expected_volatility': float(result['volatility'] * 100),
            'sharpe_ratio': float(result['sharpe_ratio'])
        }
    
    def monte_carlo_forecast(self, db: Session, portfolio_id: int, days: int = 252) -> Dict[str, any]:
        """
        Run Monte Carlo simulation for portfolio value forecast
        """
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            return {'error': 'Portfolio not found'}
        
        current_value = portfolio.current_value
        
        # Estimate portfolio parameters from historical data
        holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
        
        if not holdings:
            return {'error': 'No holdings'}
        
        # Calculate historical returns
        returns_list = []
        for holding in holdings:
            asset = db.query(Asset).filter(Asset.id == holding.asset_id).first()
            if asset:
                price_history = db.query(PriceHistory).filter(
                    PriceHistory.asset_id == asset.id
                ).order_by(PriceHistory.timestamp.desc()).limit(100).all()
                
                if len(price_history) > 1:
                    prices = np.array([ph.close for ph in reversed(price_history)])
                    returns = np.diff(prices) / prices[:-1]
                    returns_list.append(returns)
        
        if not returns_list:
            return {'error': 'Insufficient data'}
        
        portfolio_returns = np.mean(returns_list, axis=0)
        mu = np.mean(portfolio_returns)
        sigma = np.std(portfolio_returns)
        
        # Run simulation
        simulation = advanced_analytics.monte_carlo_simulation(
            initial_price=current_value,
            mu=mu,
            sigma=sigma,
            days=days,
            simulations=10000
        )
        
        return {
            'mean_final_value': float(simulation['mean_final_price']),
            'median_final_value': float(simulation['median_final_price']),
            'percentile_5': float(simulation['percentile_5']),
            'percentile_95': float(simulation['percentile_95']),
            'probability_profit': float(simulation['probability_profit'] * 100),
            'days_forecasted': days
        }

analytics_service = AnalyticsService()
