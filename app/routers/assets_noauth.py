from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database import get_db
from app.models import Asset, AssetType
from app.schemas import Asset as AssetSchema

router = APIRouter(prefix="/api/assets", tags=["assets"])

@router.get("/", response_model=List[AssetSchema])
def get_assets(
    skip: int = 0,
    limit: int = 100,
    asset_type: Optional[AssetType] = None,
    db: Session = Depends(get_db)
):
    """Get all assets with optional filtering"""
    query = db.query(Asset)
    
    if asset_type:
        query = query.filter(Asset.asset_type == asset_type)
    
    assets = query.offset(skip).limit(limit).all()
    return assets

@router.get("/search/{symbol}")
def search_asset(symbol: str, db: Session = Depends(get_db)):
    """Search asset by symbol"""
    asset = db.query(Asset).filter(Asset.symbol.ilike(f"%{symbol}%")).first()
    if not asset:
        # Create mock asset for demo
        asset = Asset(
            symbol=symbol.upper(),
            name=f"{symbol.upper()} Stock",
            asset_type=AssetType.STOCK,
            current_price=100.0
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)
    return asset

@router.get("/{asset_id}", response_model=AssetSchema)
def get_asset(asset_id: int, db: Session = Depends(get_db)):
    """Get specific asset"""
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset

@router.get("/types/list")
def get_asset_types():
    """Get available asset types"""
    return [asset_type.value for asset_type in AssetType]
