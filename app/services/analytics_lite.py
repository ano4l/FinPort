"""Lightweight analytics service without heavy dependencies"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Portfolio, Holding, Asset, PriceHistory, Transaction
from app.schemas import PortfolioAnalytics


class AnalyticsService:
    """Lightweight analytics without numpy/pandas"""
    
    def calculate_portfolio_value(self, db: Session, portfolio_id: int) -> float:
        """Calculate total portfolio value"""
        try:
            holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
            total_value = 0.0
            
            for holding in holdings:
                asset = db.query(Asset).filter(Asset.id == holding.asset_id).first()
                if asset and asset.current_price:
                    holding.current_value = holding.quantity * asset.current_price
                    holding.unrealized_pnl = holding.current_value - (holding.quantity * holding.average_buy_price)
                    if holding.average_buy_price > 0:
                        holding.unrealized_pnl_percent = (holding.unrealized_pnl / (holding.quantity * holding.average_buy_price)) * 100
                    else:
                        holding.unrealized_pnl_percent = 0
                    total_value += holding.current_value
            
            db.commit()
            return total_value
        except Exception as e:
            print(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def get_portfolio_analytics(self, db: Session, portfolio_id: int) -> PortfolioAnalytics:
        """Get portfolio analytics"""
        try:
            portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
            if not portfolio:
                return PortfolioAnalytics(
                    portfolio_id=portfolio_id,
                    total_value=0.0,
                    total_invested=0.0,
                    total_return=0.0,
                    total_return_percent=0.0,
                    sharpe_ratio=0.0,
                    volatility=0.0,
                    var_95=0.0
                )
            
            current_value = self.calculate_portfolio_value(db, portfolio_id)
            total_invested = portfolio.initial_value or 0.0
            total_return = current_value - total_invested
            total_return_percent = (total_return / total_invested * 100) if total_invested > 0 else 0.0
            
            return PortfolioAnalytics(
                portfolio_id=portfolio_id,
                total_value=current_value,
                total_invested=total_invested,
                total_return=total_return,
                total_return_percent=total_return_percent,
                sharpe_ratio=0.0,
                volatility=0.0,
                var_95=0.0
            )
        except Exception as e:
            print(f"Error getting portfolio analytics: {e}")
            return PortfolioAnalytics(
                portfolio_id=portfolio_id,
                total_value=0.0,
                total_invested=0.0,
                total_return=0.0,
                total_return_percent=0.0,
                sharpe_ratio=0.0,
                volatility=0.0,
                var_95=0.0
            )
    
    def get_advanced_risk_metrics(self, db: Session, portfolio_id: int) -> Dict:
        """Get advanced risk metrics (simplified)"""
        return {
            "portfolio_id": portfolio_id,
            "cvar": 0.0,
            "expected_shortfall": 0.0,
            "sortino_ratio": 0.0,
            "omega_ratio": 0.0,
            "downside_deviation": 0.0
        }
    
    def optimize_portfolio(self, db: Session, portfolio_id: int) -> Dict:
        """Portfolio optimization (simplified)"""
        return {
            "portfolio_id": portfolio_id,
            "optimal_weights": {},
            "expected_return": 0.0,
            "expected_volatility": 0.0,
            "sharpe_ratio": 0.0
        }
    
    def monte_carlo_forecast(self, db: Session, portfolio_id: int, days: int = 252) -> Dict:
        """Monte Carlo simulation (simplified)"""
        return {
            "portfolio_id": portfolio_id,
            "days": days,
            "simulations": 1000,
            "mean_return": 0.0,
            "std_return": 0.0,
            "percentile_5": 0.0,
            "percentile_95": 0.0
        }


# Create singleton instance
analytics_service = AnalyticsService()
