import yfinance as yf
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from app.models import Asset, AssetType, PriceHistory
from sqlalchemy.orm import Session
import httpx

class MarketDataService:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        
    async def get_stock_price(self, symbol: str) -> Optional[Dict]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            history = ticker.history(period="1d")
            
            if history.empty:
                return None
                
            current_price = history['Close'].iloc[-1]
            
            return {
                "symbol": symbol,
                "price": float(current_price),
                "change_24h": float(history['Close'].iloc[-1] - history['Open'].iloc[0]) if len(history) > 0 else 0,
                "volume_24h": float(history['Volume'].iloc[-1]) if len(history) > 0 else 0,
                "market_cap": info.get('marketCap'),
                "timestamp": datetime.utcnow()
            }
        except Exception as e:
            print(f"Error fetching stock price for {symbol}: {e}")
            return None
    
    async def get_crypto_price(self, symbol: str) -> Optional[Dict]:
        try:
            crypto_symbol = f"{symbol}-USD"
            ticker = yf.Ticker(crypto_symbol)
            history = ticker.history(period="1d")
            
            if history.empty:
                return None
                
            current_price = history['Close'].iloc[-1]
            
            return {
                "symbol": symbol,
                "price": float(current_price),
                "change_24h": float(history['Close'].iloc[-1] - history['Open'].iloc[0]) if len(history) > 0 else 0,
                "volume_24h": float(history['Volume'].iloc[-1]) if len(history) > 0 else 0,
                "market_cap": None,
                "timestamp": datetime.utcnow()
            }
        except Exception as e:
            print(f"Error fetching crypto price for {symbol}: {e}")
            return None
    
    async def get_asset_price(self, symbol: str, asset_type: AssetType) -> Optional[Dict]:
        if asset_type == AssetType.CRYPTO:
            return await self.get_crypto_price(symbol)
        else:
            return await self.get_stock_price(symbol)
    
    async def get_historical_data(self, symbol: str, asset_type: AssetType, period: str = "1y") -> List[Dict]:
        try:
            if asset_type == AssetType.CRYPTO:
                symbol = f"{symbol}-USD"
            
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period)
            
            data = []
            for index, row in history.iterrows():
                data.append({
                    "timestamp": index,
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": float(row['Volume'])
                })
            
            return data
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    async def update_asset_price(self, db: Session, asset: Asset) -> bool:
        try:
            price_data = await self.get_asset_price(asset.symbol, asset.asset_type)
            
            if price_data:
                asset.current_price = price_data['price']
                asset.volume_24h = price_data['volume_24h']
                asset.market_cap = price_data['market_cap']
                asset.last_updated = datetime.utcnow()
                db.commit()
                return True
            return False
        except Exception as e:
            print(f"Error updating asset price: {e}")
            db.rollback()
            return False
    
    async def search_assets(self, query: str, asset_type: Optional[AssetType] = None) -> List[Dict]:
        results = []
        
        try:
            ticker = yf.Ticker(query.upper())
            info = ticker.info
            
            if info and 'symbol' in info:
                results.append({
                    "symbol": info.get('symbol', query.upper()),
                    "name": info.get('longName', info.get('shortName', query)),
                    "asset_type": AssetType.STOCK if info.get('quoteType') == 'EQUITY' else AssetType.ETF,
                    "sector": info.get('sector'),
                    "current_price": info.get('currentPrice'),
                    "market_cap": info.get('marketCap')
                })
        except Exception as e:
            print(f"Error searching for asset {query}: {e}")
        
        return results

market_data_service = MarketDataService()
