import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models import Asset, AssetType, PriceHistory, User, UserPreferences, Recommendation
from app.schemas import Recommendation as RecommendationSchema
from app.services.simple_ml import simple_ml_engine

class MLRecommendationEngine:
    """
    Simplified ML recommendation engine without pandas/sklearn dependencies
    """
    
    def __init__(self):
        self.model_version = "1.0.0"
    
    async def generate_recommendations(
        self, 
        db: Session, 
        user_id: int, 
        asset_types: List[AssetType] = None,
        limit: int = 10
    ) -> List[RecommendationSchema]:
        """Delegate to simple ML engine"""
        return simple_ml_engine.generate_simple_recommendations(
            db, user_id, asset_types, limit
        )
    
    def get_diversified_recommendations(
        self,
        db: Session,
        user_id: int,
        total_recommendations: int = 10
    ) -> List[RecommendationSchema]:
        
        asset_distribution = {
            AssetType.STOCK: 4,
            AssetType.CRYPTO: 2,
            AssetType.ETF: 2,
            AssetType.COMMODITY: 1,
            AssetType.BOND: 1
        }
        
        all_recommendations = []
        
        for asset_type, count in asset_distribution.items():
            recs = db.query(Recommendation).filter(
                Recommendation.user_id == user_id,
                Recommendation.asset_type == asset_type,
                Recommendation.expires_at > datetime.utcnow()
            ).order_by(Recommendation.confidence_score.desc()).limit(count).all()
            
            all_recommendations.extend(recs)
        
        return all_recommendations

ml_engine = MLRecommendationEngine()
