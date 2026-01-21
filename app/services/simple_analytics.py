import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Portfolio, Holding, Asset, PriceHistory, Transaction

class SimpleAnalyticsService:
    """
    Basic analytics without pandas dependency
    """
    
    def calculate_portfolio_value(self, db: Session, portfolio_id: int) -> float:
        holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
        total_value = 0.0
        
        for holding in holdings:
            asset = db.query(Asset).filter(Asset.id == holding.asset_id).first()
            if asset and asset.current_price:
                total_value += holding.quantity * asset.current_price
        
        return total_value
    
    def calculate_returns(self, db: Session, portfolio_id: int) -> Dict[str, float]:
        """Calculate basic returns without pandas"""
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            return {'daily': 0.0, 'monthly': 0.0, 'yearly': 0.0}
        
        # Simple return calculation
        if portfolio.initial_value and portfolio.initial_value > 0:
            total_return = (portfolio.current_value - portfolio.initial_value) / portfolio.initial_value
            
            # Approximate time-based returns (simplified)
            days_held = (datetime.utcnow() - portfolio.created_at).days
            if days_held > 0:
                daily_return = total_return / days_held
                monthly_return = daily_return * 30
                yearly_return = daily_return * 365
            else:
                daily_return = monthly_return = yearly_return = 0.0
        else:
            daily_return = monthly_return = yearly_return = 0.0
        
        return {
            'daily': daily_return,
            'monthly': monthly_return,
            'yearly': yearly_return
        }
    
    def calculate_basic_risk_metrics(self, db: Session, portfolio_id: int) -> Dict[str, float]:
        """Calculate basic risk metrics without advanced math"""
        holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
        
        if not holdings:
            return {'volatility': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
        
        # Get recent price history for volatility
        all_returns = []
        for holding in holdings:
            asset = db.query(Asset).filter(Asset.id == holding.asset_id).first()
            if asset:
                price_history = db.query(PriceHistory).filter(
                    PriceHistory.asset_id == asset.id
                ).order_by(PriceHistory.timestamp.desc()).limit(30).all()
                
                if len(price_history) > 1:
                    prices = [ph.close for ph in reversed(price_history)]
                    for i in range(1, len(prices)):
                        ret = (prices[i] - prices[i-1]) / prices[i-1]
                        all_returns.append(ret)
        
        if all_returns:
            volatility = np.std(all_returns) * np.sqrt(252)  # Annualized
            mean_return = np.mean(all_returns) * 252
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
        else:
            volatility = sharpe_ratio = 0.0
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': 0.0  # Simplified
        }
    
    def get_portfolio_allocation(self, db: Session, portfolio_id: int) -> Dict[str, float]:
        """Get asset allocation by type"""
        holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
        allocation = {}
        total_value = 0.0
        
        # Calculate total value first
        for holding in holdings:
            asset = db.query(Asset).filter(Asset.id == holding.asset_id).first()
            if asset and asset.current_price:
                total_value += holding.quantity * asset.current_price
        
        # Calculate allocation percentages
        for holding in holdings:
            asset = db.query(Asset).filter(Asset.id == holding.asset_id).first()
            if asset and asset.current_price:
                value = holding.quantity * asset.current_price
                asset_type = asset.asset_type.value
                allocation[asset_type] = allocation.get(asset_type, 0.0) + (value / total_value * 100)
        
        return allocation

simple_analytics = SimpleAnalyticsService()
