from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models import Portfolio, Holding, Asset, Transaction
from app.schemas import (
    PortfolioCreate, Portfolio as PortfolioSchema, PortfolioUpdate,
    PortfolioWithHoldings, HoldingCreate, Holding as HoldingSchema,
    TransactionCreate, Transaction as TransactionSchema,
    PortfolioAnalytics
)
from app.services.analytics_lite import analytics_service
from app.services.market_data import market_data_service

router = APIRouter(prefix="/api/portfolios", tags=["Portfolios"])

@router.post("/", response_model=PortfolioSchema)
def create_portfolio(
    portfolio: PortfolioCreate,
    db: Session = Depends(get_db)
):
    db_portfolio = Portfolio(**portfolio.dict())
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

@router.get("/", response_model=List[PortfolioSchema])
def get_portfolios(
    db: Session = Depends(get_db)
):
    portfolios = db.query(Portfolio).all()
    
    for portfolio in portfolios:
        portfolio.current_value = analytics_service.calculate_portfolio_value(db, portfolio.id)
    
    db.commit()
    return portfolios

@router.get("/{portfolio_id}", response_model=PortfolioWithHoldings)
def get_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    portfolio.current_value = analytics_service.calculate_portfolio_value(db, portfolio.id)
    db.commit()
    
    return portfolio

@router.put("/{portfolio_id}", response_model=PortfolioSchema)
def update_portfolio(
    portfolio_id: int,
    portfolio_update: PortfolioUpdate,
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    update_data = portfolio_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(portfolio, field, value)
    
    db.commit()
    db.refresh(portfolio)
    return portfolio

@router.delete("/{portfolio_id}")
def delete_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    db.delete(portfolio)
    db.commit()
    return {"message": "Portfolio deleted successfully"}

@router.post("/{portfolio_id}/holdings", response_model=HoldingSchema)
async def add_holding(
    portfolio_id: int,
    holding: HoldingCreate,
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    asset = db.query(Asset).filter(Asset.symbol == holding.asset_symbol).first()
    
    if not asset:
        asset_data = await market_data_service.search_assets(holding.asset_symbol)
        if not asset_data:
            raise HTTPException(status_code=404, detail="Asset not found")
        
        asset = Asset(
            symbol=asset_data[0]['symbol'],
            name=asset_data[0]['name'],
            asset_type=asset_data[0]['asset_type'],
            sector=asset_data[0].get('sector'),
            current_price=asset_data[0].get('current_price'),
            market_cap=asset_data[0].get('market_cap')
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)
    
    existing_holding = db.query(Holding).filter(
        Holding.portfolio_id == portfolio_id,
        Holding.asset_id == asset.id
    ).first()
    
    if existing_holding:
        total_quantity = existing_holding.quantity + holding.quantity
        total_cost = (existing_holding.quantity * existing_holding.average_buy_price) + (holding.quantity * holding.purchase_price)
        existing_holding.average_buy_price = total_cost / total_quantity
        existing_holding.quantity = total_quantity
        db.commit()
        db.refresh(existing_holding)
        db_holding = existing_holding
    else:
        db_holding = Holding(
            portfolio_id=portfolio_id,
            asset_id=asset.id,
            quantity=holding.quantity,
            average_buy_price=holding.purchase_price
        )
        db.add(db_holding)
        db.commit()
        db.refresh(db_holding)
    
    transaction = Transaction(
        portfolio_id=portfolio_id,
        asset_symbol=asset.symbol,
        transaction_type="buy",
        quantity=holding.quantity,
        price=holding.purchase_price,
        total_value=holding.quantity * holding.purchase_price
    )
    db.add(transaction)
    db.commit()
    
    return db_holding

@router.get("/{portfolio_id}/analytics", response_model=PortfolioAnalytics)
def get_portfolio_analytics(
    portfolio_id: int,
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    analytics = analytics_service.get_portfolio_analytics(db, portfolio_id)
    return analytics

@router.get("/{portfolio_id}/transactions", response_model=List[TransactionSchema])
def get_transactions(
    portfolio_id: int,
    db: Session = Depends(get_db)
):
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    transactions = db.query(Transaction).filter(
        Transaction.portfolio_id == portfolio_id
    ).order_by(Transaction.transaction_date.desc()).all()
    
    return transactions
