import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models import Asset, AssetType, PriceHistory, User, UserPreferences, Recommendation
from app.schemas import Recommendation as RecommendationSchema

class SimpleMLRecommendationEngine:
    """
    Basic ML recommendations without complex dependencies
    """
    
    def __init__(self):
        self.model_version = "1.0.0"
    
    def calculate_simple_indicators(self, prices: List[float]) -> Dict[str, float]:
        """Calculate basic technical indicators"""
        if len(prices) < 20:
            return {}
        
        # Simple moving average
        sma_20 = np.mean(prices[-20:])
        current_price = prices[-1]
        
        # Price momentum
        momentum = (current_price - sma_20) / sma_20 * 100
        
        # Simple volatility
        returns = np.diff(prices[-20:]) / prices[-20:-1]
        volatility = np.std(returns) * 100
        
        # Simple RSI approximation
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        return {
            'sma_20': float(sma_20),
            'momentum': float(momentum),
            'volatility': float(volatility),
            'rsi': float(rsi)
        }
    
    def generate_simple_recommendations(
        self, 
        db: Session, 
        user_id: int, 
        asset_types: List[AssetType] = None,
        limit: int = 10
    ) -> List[RecommendationSchema]:
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return []
        
        preferences = db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
        risk_tolerance = preferences.risk_tolerance.value if preferences else "medium"
        
        if asset_types is None:
            asset_types = [AssetType.STOCK, AssetType.CRYPTO, AssetType.ETF]
        
        assets = db.query(Asset).filter(Asset.asset_type.in_(asset_types)).limit(20).all()
        
        recommendations = []
        
        for asset in assets:
            price_history = db.query(PriceHistory).filter(
                PriceHistory.asset_id == asset.id
            ).order_by(PriceHistory.timestamp.desc()).limit(30).all()
            
            if len(price_history) < 20:
                continue
            
            prices = [ph.close for ph in reversed(price_history)]
            indicators = self.calculate_simple_indicators(prices)
            
            if not indicators:
                continue
            
            # Simple recommendation logic
            rsi = indicators['rsi']
            momentum = indicators['momentum']
            
            if rsi < 30 and momentum > -5:
                rec_type = "BUY"
                confidence = 0.8
            elif rsi > 70 and momentum < 5:
                rec_type = "SELL"
                confidence = 0.7
            else:
                rec_type = "HOLD"
                confidence = 0.5
            
            # Simple risk score based on asset type and volatility
            base_risk = {
                AssetType.STOCK: 0.5,
                AssetType.CRYPTO: 0.8,
                AssetType.ETF: 0.4,
                AssetType.BOND: 0.2,
                AssetType.COMMODITY: 0.6
            }
            
            risk_score = base_risk.get(asset.asset_type, 0.5)
            risk_score += indicators['volatility'] / 100 * 0.3
            risk_score = min(max(risk_score, 0.0), 1.0)
            
            # Simple expected return
            if rec_type == "BUY":
                expected_return = abs(momentum) * 0.5 + np.random.normal(2, 1)
            elif rec_type == "SELL":
                expected_return = np.random.normal(-2, 1)
            else:
                expected_return = np.random.normal(0, 0.5)
            
            recommendation = Recommendation(
                user_id=user_id,
                asset_symbol=asset.symbol,
                asset_type=asset.asset_type,
                recommendation_type=rec_type,
                confidence_score=confidence,
                risk_score=risk_score,
                expected_return=expected_return,
                reasoning=indicators,
                model_version=self.model_version,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )
            
            db.add(recommendation)
            recommendations.append(recommendation)
        
        db.commit()
        
        # Sort by confidence and risk-adjusted score
        recommendations.sort(key=lambda x: x.confidence_score * (1 - x.risk_score), reverse=True)
        
        return recommendations[:limit]

simple_ml_engine = SimpleMLRecommendationEngine()
