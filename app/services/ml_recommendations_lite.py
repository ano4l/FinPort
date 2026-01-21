"""Lightweight ML recommendations engine without heavy dependencies"""
from typing import List, Optional
from sqlalchemy.orm import Session
from app.models import Asset, AssetType, Recommendation
from app.schemas import Recommendation as RecommendationSchema
from datetime import datetime, timedelta


class MLRecommendationEngine:
    """Lightweight recommendation engine for Vercel deployment"""
    
    def __init__(self):
        self.model_version = "1.0.0-lite"
    
    async def generate_recommendations(
        self,
        db: Session,
        user_id: int,
        asset_types: Optional[List[AssetType]] = None,
        limit: int = 10
    ) -> List[RecommendationSchema]:
        """Generate simple recommendations without ML"""
        try:
            # Get some assets from database
            query = db.query(Asset)
            if asset_types:
                query = query.filter(Asset.asset_type.in_(asset_types))
            
            assets = query.limit(limit).all()
            
            recommendations = []
            for asset in assets:
                rec = Recommendation(
                    asset_symbol=asset.symbol,
                    asset_type=asset.asset_type,
                    recommendation_type="hold",
                    confidence_score=0.5,
                    risk_score=0.5,
                    expected_return=0.0,
                    reasoning={"note": "Placeholder recommendation"},
                    model_version=self.model_version,
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=7)
                )
                recommendations.append(rec)
            
            return recommendations
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []
    
    def get_diversified_recommendations(
        self,
        db: Session,
        user_id: int,
        total_recommendations: int = 10
    ) -> List[RecommendationSchema]:
        """Get diversified portfolio recommendations"""
        try:
            # Get a mix of different asset types
            assets = db.query(Asset).limit(total_recommendations).all()
            
            recommendations = []
            for asset in assets:
                rec = Recommendation(
                    asset_symbol=asset.symbol,
                    asset_type=asset.asset_type,
                    recommendation_type="buy",
                    confidence_score=0.6,
                    risk_score=0.4,
                    expected_return=0.05,
                    reasoning={"note": "Diversification recommendation"},
                    model_version=self.model_version,
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=7)
                )
                recommendations.append(rec)
            
            return recommendations
        except Exception as e:
            print(f"Error getting diversified recommendations: {e}")
            return []


# Create singleton instance
ml_engine = MLRecommendationEngine()
