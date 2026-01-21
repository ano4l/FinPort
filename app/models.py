from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import enum
from datetime import datetime

class AssetType(str, enum.Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    BOND = "bond"
    ETF = "etf"
    REAL_ESTATE = "real_estate"
    FOREX = "forex"
    OTHER = "other"

class RiskLevel(str, enum.Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class UserPreferences(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    risk_tolerance = Column(Enum(RiskLevel), default=RiskLevel.MEDIUM)
    preferred_assets = Column(JSON, default=list)
    investment_horizon_years = Column(Integer, default=5)
    auto_rebalance = Column(Boolean, default=False)
    notification_enabled = Column(Boolean, default=True)
    theme = Column(String, default="dark")

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    initial_value = Column(Float, default=0.0)
    current_value = Column(Float, default=0.0)
    target_allocation = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="portfolio", cascade="all, delete-orphan")

class Asset(Base):
    __tablename__ = "assets"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    asset_type = Column(Enum(AssetType), nullable=False)
    current_price = Column(Float, nullable=True)
    asset_metadata = Column(JSON, nullable=True)  # Additional asset data (renamed)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    holdings = relationship("Holding", back_populates="asset")
    price_history = relationship("PriceHistory", back_populates="asset")

class Holding(Base):
    __tablename__ = "holdings"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    asset_id = Column(Integer, ForeignKey("assets.id"))
    quantity = Column(Float, nullable=False)
    average_buy_price = Column(Float, nullable=False)
    current_value = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    unrealized_pnl_percent = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    portfolio = relationship("Portfolio", back_populates="holdings")
    asset = relationship("Asset", back_populates="holdings")

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    asset_symbol = Column(String, nullable=False)
    transaction_type = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    fees = Column(Float, default=0.0)
    notes = Column(String)
    transaction_date = Column(DateTime(timezone=True), server_default=func.now())
    
    portfolio = relationship("Portfolio", back_populates="transactions")

class PriceHistory(Base):
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"))
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Float)
    
    asset = relationship("Asset", back_populates="price_history")

class Recommendation(Base):
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_symbol = Column(String, nullable=False)
    asset_type = Column(Enum(AssetType), nullable=False)
    recommendation_type = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    expected_return = Column(Float)
    reasoning = Column(JSON, default=dict)
    model_version = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
