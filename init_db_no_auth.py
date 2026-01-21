#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import engine, Base
from app.models import Portfolio, Asset, UserPreferences, AssetType, RiskLevel
from sqlalchemy.orm import Session

def init_database():
    """Initialize database with sample data without authentication requirements"""
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")
    
    # Create session
    db = Session(engine)
    
    try:
        # Check if data already exists
        existing_portfolio = db.query(Portfolio).first()
        if existing_portfolio:
            print("Database already contains data. Skipping initialization.")
            return
        
        # Create sample portfolio
        sample_portfolio = Portfolio(
            name="Sample Portfolio",
            description="A diversified portfolio for demonstration",
            initial_value=10000.0,
            current_value=10000.0
        )
        db.add(sample_portfolio)
        db.commit()
        db.refresh(sample_portfolio)
        print(f"Created sample portfolio: {sample_portfolio.name}")
        
        # Create sample assets
        assets = [
            Asset(symbol="AAPL", name="Apple Inc.", asset_type=AssetType.STOCK, current_price=150.0),
            Asset(symbol="GOOGL", name="Alphabet Inc.", asset_type=AssetType.STOCK, current_price=2500.0),
            Asset(symbol="BTC", name="Bitcoin", asset_type=AssetType.CRYPTO, current_price=35000.0),
            Asset(symbol="ETH", name="Ethereum", asset_type=AssetType.CRYPTO, current_price=2000.0),
            Asset(symbol="SPY", name="SPDR S&P 500 ETF", asset_type=AssetType.ETF, current_price=400.0),
        ]
        
        for asset in assets:
            db.add(asset)
        
        db.commit()
        print(f"Created {len(assets)} sample assets")
        
        # Create default preferences
        preferences = UserPreferences(
            risk_tolerance=RiskLevel.MEDIUM,
            preferred_assets=["stock", "etf"],
            investment_horizon_years=5,
            auto_rebalance=False,
            notification_enabled=True,
            theme="dark"
        )
        db.add(preferences)
        db.commit()
        print("Created default user preferences")
        
        print("\nDatabase initialization completed successfully!")
        print("Summary:")
        print(f"   - 1 portfolio created")
        print(f"   - {len(assets)} assets created")
        print(f"   - Default preferences set")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    init_database()
