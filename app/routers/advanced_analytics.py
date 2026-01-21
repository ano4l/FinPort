from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from app.database import get_db
from app.services.analytics_lite import analytics_service

router = APIRouter(prefix="/api/advanced-analytics", tags=["Advanced Analytics"])

@router.get("/risk-metrics/{portfolio_id}")
def get_advanced_risk_metrics(
    portfolio_id: int,
    db: Session = Depends(get_db)
):
    """
    Get advanced risk metrics including:
    - Conditional Value at Risk (CVaR)
    - Expected Shortfall
    - Sortino Ratio
    - Omega Ratio
    - Downside Deviation
    """
    metrics = analytics_service.get_advanced_risk_metrics(db, portfolio_id)
    
    if not metrics:
        raise HTTPException(status_code=404, detail="Insufficient data for risk metrics")
    
    return metrics

@router.get("/optimize/{portfolio_id}")
def optimize_portfolio(
    portfolio_id: int,
    target_return: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """
    Perform Markowitz Mean-Variance Optimization
    
    Returns optimal portfolio weights to maximize Sharpe ratio
    or minimize variance for a target return
    """
    result = analytics_service.optimize_portfolio(db, portfolio_id)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return result

@router.get("/monte-carlo/{portfolio_id}")
def monte_carlo_forecast(
    portfolio_id: int,
    days: int = Query(252, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Run Monte Carlo simulation for portfolio value forecast
    
    Uses Geometric Brownian Motion to simulate 10,000 possible
    future paths for portfolio value
    """
    result = analytics_service.monte_carlo_forecast(db, portfolio_id, days)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return result
