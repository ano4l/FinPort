from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models import Portfolio, Holding, Asset
from app.schemas import PortfolioCreate, Portfolio as PortfolioSchema, PortfolioUpdate

router = APIRouter(prefix="/api/portfolios", tags=["portfolios"])

# Mock user ID (since no auth)
MOCK_USER_ID = 1

@router.get("/", response_model=List[PortfolioSchema])
def get_portfolios(db: Session = Depends(get_db)):
    """Get all portfolios for mock user"""
    portfolios = db.query(Portfolio).filter(Portfolio.user_id == MOCK_USER_ID).all()
    return portfolios

@router.post("/", response_model=PortfolioSchema)
def create_portfolio(portfolio: PortfolioCreate, db: Session = Depends(get_db)):
    """Create new portfolio for mock user"""
    db_portfolio = Portfolio(
        user_id=MOCK_USER_ID,
        name=portfolio.name,
        description=portfolio.description,
        initial_value=portfolio.initial_value,
        current_value=portfolio.initial_value
    )
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

@router.get("/{portfolio_id}", response_model=PortfolioSchema)
def get_portfolio(portfolio_id: int, db: Session = Depends(get_db)):
    """Get specific portfolio"""
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == MOCK_USER_ID
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio

@router.put("/{portfolio_id}", response_model=PortfolioSchema)
def update_portfolio(portfolio_id: int, portfolio_update: PortfolioUpdate, db: Session = Depends(get_db)):
    """Update portfolio"""
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == MOCK_USER_ID
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    for field, value in portfolio_update.dict(exclude_unset=True).items():
        setattr(portfolio, field, value)
    
    db.commit()
    db.refresh(portfolio)
    return portfolio

@router.delete("/{portfolio_id}")
def delete_portfolio(portfolio_id: int, db: Session = Depends(get_db)):
    """Delete portfolio"""
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == MOCK_USER_ID
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    db.delete(portfolio)
    db.commit()
    return {"message": "Portfolio deleted successfully"}

@router.get("/{portfolio_id}/holdings")
def get_portfolio_holdings(portfolio_id: int, db: Session = Depends(get_db)):
    """Get holdings for a portfolio"""
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == MOCK_USER_ID
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
    return holdings
