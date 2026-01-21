import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Portfolio, Holding, Asset, PriceHistory, Transaction
from app.schemas import PortfolioAnalytics
from app.services.simple_analytics import simple_analytics

class AnalyticsService:
    """
    Simplified analytics service without pandas dependency
    """
    
    def calculate_portfolio_value(self, db: Session, portfolio_id: int) -> float:
        return simple_analytics.calculate_portfolio_value(db, portfolio_id)
    
    def calculate_returns(self, db: Session, portfolio_id: int) -> Dict[str, float]:
        return simple_analytics.calculate_returns(db, portfolio_id)
    
    def calculate_basic_risk_metrics(self, db: Session, portfolio_id: int) -> Dict[str, float]:
        return simple_analytics.calculate_basic_risk_metrics(db, portfolio_id)
    
    def get_portfolio_analytics(self, db: Session, portfolio_id: int) -> PortfolioAnalytics:
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            return None
        
        current_value = self.calculate_portfolio_value(db, portfolio_id)
        returns_data = self.calculate_returns(db, portfolio_id)
        risk_metrics = self.calculate_basic_risk_metrics(db, portfolio_id)
        allocation = simple_analytics.get_portfolio_allocation(db, portfolio_id)
        
        # Calculate P&L
        total_cost = sum(h.quantity * h.purchase_price for h in portfolio.holdings)
        total_pnl = current_value - total_cost
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        return PortfolioAnalytics(
            portfolio_id=portfolio_id,
            total_value=current_value,
            total_cost=total_cost,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            daily_return=returns_data['daily'],
            monthly_return=returns_data['monthly'],
            yearly_return=returns_data['yearly'],
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            volatility=risk_metrics['volatility'],
            max_drawdown=risk_metrics['max_drawdown'],
            beta=1.0,
            var_95=0.0,
            allocation_by_type=allocation,
            allocation_by_sector={},
            top_performers=[],
            worst_performers=[]
        )

analytics_service = AnalyticsService()
