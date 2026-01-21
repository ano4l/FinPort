from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database import get_db
from app.models import Asset, AssetType, PriceHistory, Recommendation
from app.schemas import Recommendation as RecommendationSchema
from app.services.ml_recommendations_lite import ml_engine

router = APIRouter(prefix="/api/recommendations", tags=["Recommendations"])

@router.get("/", response_model=List[RecommendationSchema])
async def get_recommendations(
    asset_types: Optional[List[AssetType]] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    recommendations = await ml_engine.generate_recommendations(
        db=db,
        user_id=1,  # Default user ID since we removed auth
        asset_types=asset_types,
        limit=limit
    )
    
    return recommendations

@router.get("/diversified", response_model=List[RecommendationSchema])
def get_diversified_recommendations(
    db: Session = Depends(get_db)
):
    recommendations = ml_engine.get_diversified_recommendations(
        db=db,
        user_id=1,  # Default user ID since we removed auth
        total_recommendations=10
    )
    
    return recommendations
