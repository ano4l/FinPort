from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database import get_db
from app.models import Asset, AssetType
from app.schemas import Asset as AssetSchema, MarketData
from app.services.market_data import market_data_service

router = APIRouter(prefix="/api/assets", tags=["Assets"])

@router.get("/search", response_model=List[AssetSchema])
async def search_assets(
    query: str = Query(..., min_length=1),
    asset_type: Optional[AssetType] = None,
    db: Session = Depends(get_db)
):
    results = await market_data_service.search_assets(query, asset_type)
    
    asset_schemas = []
    for result in results:
        asset_schemas.append(AssetSchema(**result))
    
    return asset_schemas

@router.get("/{symbol}", response_model=AssetSchema)
async def get_asset(
    symbol: str,
    db: Session = Depends(get_db)
):
    asset = db.query(Asset).filter(Asset.symbol == symbol.upper()).first()
    
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    await market_data_service.update_asset_price(db, asset)
    
    return asset

@router.get("/{symbol}/price", response_model=MarketData)
async def get_asset_price(
    symbol: str,
    asset_type: AssetType = AssetType.STOCK,
    db: Session = Depends(get_db)
):
    price_data = await market_data_service.get_asset_price(symbol.upper(), asset_type)
    
    if not price_data:
        raise HTTPException(status_code=404, detail="Price data not available")
    
    return MarketData(**price_data, change_percent_24h=0.0)
