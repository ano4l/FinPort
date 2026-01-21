from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models import AssetType, Recommendation
from app.schemas import Recommendation as RecommendationSchema
from app.services.ml_recommendations_simple import ml_engine

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])

# Mock user ID (since no auth)
MOCK_USER_ID = 1

@router.get("/", response_model=List[RecommendationSchema])
def get_recommendations(
    limit: int = 10,
    asset_type: AssetType = None,
    db: Session = Depends(get_db)
):
    """Get investment recommendations for mock user"""
    recommendations = db.query(Recommendation).filter(
        Recommendation.user_id == MOCK_USER_ID,
        Recommendation.expires_at > Recommendation.created_at
    ).order_by(Recommendation.confidence_score.desc()).limit(limit).all()
    
    return recommendations

@router.post("/generate")
async def generate_recommendations(
    limit: int = 10,
    asset_types: List[AssetType] = None,
    db: Session = Depends(get_db)
):
    """Generate new recommendations"""
    if asset_types is None:
        asset_types = [AssetType.STOCK, AssetType.CRYPTO, AssetType.ETF]
    
    recommendations = await ml_engine.generate_recommendations(
        db, MOCK_USER_ID, asset_types, limit
    )
    
    return recommendations

@router.get("/diversified")
def get_diversified_recommendations(
    total_recommendations: int = 10,
    db: Session = Depends(get_db)
):
    """Get diversified recommendations across asset types"""
    recommendations = ml_engine.get_diversified_recommendations(
        db, MOCK_USER_ID, total_recommendations
    )
    
    return recommendations
