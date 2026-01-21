from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models import AssetType, RiskLevel

class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserPreferencesBase(BaseModel):
    risk_tolerance: RiskLevel = RiskLevel.MEDIUM
    preferred_assets: List[str] = []
    investment_horizon_years: int = 5
    auto_rebalance: bool = False
    notification_enabled: bool = True
    theme: str = "dark"

class UserPreferencesCreate(UserPreferencesBase):
    pass

class UserPreferences(UserPreferencesBase):
    id: int
    user_id: int
    
    class Config:
        from_attributes = True

class PortfolioBase(BaseModel):
    name: str
    description: Optional[str] = None
    initial_value: float = 0.0
    target_allocation: Dict[str, float] = {}

class PortfolioCreate(PortfolioBase):
    pass

class PortfolioUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    target_allocation: Optional[Dict[str, float]] = None

class Portfolio(PortfolioBase):
    id: int
    user_id: int
    current_value: float
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class PortfolioWithHoldings(Portfolio):
    holdings: List['Holding'] = []

class AssetBase(BaseModel):
    symbol: str
    name: str
    asset_type: AssetType
    sector: Optional[str] = None

class AssetCreate(AssetBase):
    pass

class Asset(AssetBase):
    id: int
    current_price: Optional[float]
    market_cap: Optional[float]
    volume_24h: Optional[float]
    metadata: Dict[str, Any] = {}
    last_updated: Optional[datetime]
    
    class Config:
        from_attributes = True

class HoldingBase(BaseModel):
    asset_id: int
    quantity: float
    average_buy_price: float

class HoldingCreate(BaseModel):
    portfolio_id: int
    asset_symbol: str
    quantity: float
    purchase_price: float

class Holding(HoldingBase):
    id: int
    portfolio_id: int
    current_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    created_at: datetime
    updated_at: Optional[datetime]
    asset: Optional[Asset] = None
    
    class Config:
        from_attributes = True

class TransactionBase(BaseModel):
    asset_symbol: str
    transaction_type: str
    quantity: float
    price: float
    fees: float = 0.0
    notes: Optional[str] = None

class TransactionCreate(TransactionBase):
    portfolio_id: int

class Transaction(TransactionBase):
    id: int
    portfolio_id: int
    total_value: float
    transaction_date: datetime
    
    class Config:
        from_attributes = True

class PriceHistoryBase(BaseModel):
    timestamp: datetime
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: float
    volume: Optional[float]

class PriceHistory(PriceHistoryBase):
    id: int
    asset_id: int
    
    class Config:
        from_attributes = True

class RecommendationBase(BaseModel):
    asset_symbol: str
    asset_type: AssetType
    recommendation_type: str
    confidence_score: float
    risk_score: float
    expected_return: Optional[float]
    reasoning: Dict[str, Any] = {}

class Recommendation(RecommendationBase):
    id: int
    user_id: int
    model_version: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class PortfolioAnalytics(BaseModel):
    portfolio_id: int
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_percent: float
    daily_return: float
    monthly_return: float
    yearly_return: float
    sharpe_ratio: float
    volatility: float
    max_drawdown: float
    beta: float
    var_95: float
    allocation_by_type: Dict[str, float]
    allocation_by_sector: Dict[str, float]
    top_performers: List[Dict[str, Any]]
    worst_performers: List[Dict[str, Any]]

class MarketData(BaseModel):
    symbol: str
    price: float
    change_24h: float
    change_percent_24h: float
    volume_24h: float
    market_cap: Optional[float]
    timestamp: datetime
