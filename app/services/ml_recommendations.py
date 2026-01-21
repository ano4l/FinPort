import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models import Asset, AssetType, PriceHistory, User, UserPreferences, Recommendation
from app.schemas import Recommendation as RecommendationSchema
from app.services.simple_ml import simple_ml_engine

class MLRecommendationEngine:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.model_version = "1.0.0"
    
    def calculate_technical_indicators(self, prices: pd.DataFrame) -> Dict[str, float]:
        if len(prices) < 20:
            return {}
        
        close_prices = prices['close'].values
        
        sma_20 = np.mean(close_prices[-20:])
        sma_50 = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else sma_20
        
        rsi = self.calculate_rsi(close_prices)
        
        macd, signal = self.calculate_macd(close_prices)
        
        volatility = np.std(close_prices[-30:]) / np.mean(close_prices[-30:]) if len(close_prices) >= 30 else 0
        
        momentum = (close_prices[-1] - close_prices[-10]) / close_prices[-10] if len(close_prices) >= 10 else 0
        
        return {
            "sma_20": float(sma_20),
            "sma_50": float(sma_50),
            "rsi": float(rsi),
            "macd": float(macd),
            "signal": float(signal),
            "volatility": float(volatility * 100),
            "momentum": float(momentum * 100),
            "current_price": float(close_prices[-1])
        }
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def calculate_macd(self, prices: np.ndarray) -> tuple:
        if len(prices) < 26:
            return 0.0, 0.0
        
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        
        macd = ema_12 - ema_26
        signal = self.calculate_ema(np.array([macd]), 9)
        
        return float(macd), float(signal)
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return float(ema)
    
    def calculate_risk_score(self, indicators: Dict[str, float], asset_type: AssetType) -> float:
        base_risk = {
            AssetType.STOCK: 0.5,
            AssetType.CRYPTO: 0.8,
            AssetType.BOND: 0.2,
            AssetType.ETF: 0.4,
            AssetType.COMMODITY: 0.6,
            AssetType.REAL_ESTATE: 0.3,
            AssetType.FOREX: 0.7,
            AssetType.OTHER: 0.5
        }
        
        risk = base_risk.get(asset_type, 0.5)
        
        volatility = indicators.get('volatility', 0)
        risk += (volatility / 100) * 0.3
        
        risk = min(max(risk, 0.0), 1.0)
        
        return float(risk)
    
    def generate_recommendation_type(self, indicators: Dict[str, float], risk_tolerance: str) -> str:
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        signal = indicators.get('signal', 0)
        momentum = indicators.get('momentum', 0)
        
        buy_signals = 0
        sell_signals = 0
        
        if rsi < 30:
            buy_signals += 2
        elif rsi > 70:
            sell_signals += 2
        
        if macd > signal:
            buy_signals += 1
        else:
            sell_signals += 1
        
        if momentum > 5:
            buy_signals += 1
        elif momentum < -5:
            sell_signals += 1
        
        if buy_signals > sell_signals:
            return "BUY"
        elif sell_signals > buy_signals:
            return "SELL"
        else:
            return "HOLD"
    
    def calculate_expected_return(self, indicators: Dict[str, float], recommendation_type: str) -> float:
        momentum = indicators.get('momentum', 0)
        volatility = indicators.get('volatility', 0)
        
        if recommendation_type == "BUY":
            base_return = abs(momentum) * 0.5
            expected_return = base_return + np.random.normal(5, volatility / 10)
        elif recommendation_type == "SELL":
            expected_return = np.random.normal(-3, volatility / 10)
        else:
            expected_return = np.random.normal(2, volatility / 10)
        
        return float(expected_return)
    
    async def generate_recommendations(
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
            asset_types = [AssetType.STOCK, AssetType.CRYPTO, AssetType.ETF, AssetType.COMMODITY]
        
        assets = db.query(Asset).filter(Asset.asset_type.in_(asset_types)).limit(50).all()
        
        recommendations = []
        
        for asset in assets:
            price_history = db.query(PriceHistory).filter(
                PriceHistory.asset_id == asset.id
            ).order_by(PriceHistory.timestamp.desc()).limit(100).all()
            
            if len(price_history) < 20:
                continue
            
            prices_df = pd.DataFrame([{
                'timestamp': ph.timestamp,
                'close': ph.close,
                'volume': ph.volume
            } for ph in reversed(price_history)])
            
            indicators = self.calculate_technical_indicators(prices_df)
            
            if not indicators:
                continue
            
            risk_score = self.calculate_risk_score(indicators, asset.asset_type)
            
            recommendation_type = self.generate_recommendation_type(indicators, risk_tolerance)
            
            expected_return = self.calculate_expected_return(indicators, recommendation_type)
            
            confidence_score = min(abs(indicators.get('rsi', 50) - 50) / 50 + 0.5, 1.0)
            
            reasoning = {
                "technical_indicators": indicators,
                "signals": {
                    "rsi_signal": "oversold" if indicators.get('rsi', 50) < 30 else "overbought" if indicators.get('rsi', 50) > 70 else "neutral",
                    "macd_signal": "bullish" if indicators.get('macd', 0) > indicators.get('signal', 0) else "bearish",
                    "momentum_signal": "positive" if indicators.get('momentum', 0) > 0 else "negative"
                }
            }
            
            recommendation = Recommendation(
                user_id=user_id,
                asset_symbol=asset.symbol,
                asset_type=asset.asset_type,
                recommendation_type=recommendation_type,
                confidence_score=confidence_score,
                risk_score=risk_score,
                expected_return=expected_return,
                reasoning=reasoning,
                model_version=self.model_version,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=1)
            )
            
            db.add(recommendation)
            recommendations.append(recommendation)
        
        db.commit()
        
        recommendations.sort(key=lambda x: x.confidence_score * (1 - x.risk_score), reverse=True)
        
        return recommendations[:limit]
    
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

    async def generate_advanced_predictions(
        self,
        db: Session,
        asset_id: int,
        forecast_horizon: int = 5
    ) -> Dict[str, any]:
        """
        Generate advanced predictions using multiple ML models
        """
        # Get price history
        price_history = db.query(PriceHistory).filter(
            PriceHistory.asset_id == asset_id
        ).order_by(PriceHistory.timestamp.desc()).limit(200).all()
        
        if len(price_history) < 100:
            return {'error': 'Insufficient data'}
        
        prices = np.array([ph.close for ph in reversed(price_history)])
        
        # Calculate features
        returns = np.diff(prices) / prices[:-1]
        
        # Technical indicators as features
        features = []
        for i in range(20, len(prices)):
            window = prices[i-20:i]
            feature_vector = [
                np.mean(window),  # SMA
                np.std(window),   # Volatility
                (prices[i] - np.min(window)) / (np.max(window) - np.min(window)),  # Stochastic
                prices[i] / np.mean(window) - 1  # Price momentum
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        target = returns[20:]
        
        # XGBoost prediction
        xgb_result = advanced_ml_engine.xgboost_price_prediction(
            features, target, forecast_horizon
        )
        
        # LSTM prediction
        lstm_result = advanced_ml_engine.lstm_price_forecasting(
            prices, sequence_length=60, forecast_horizon=forecast_horizon
        )
        
        # ARIMA prediction
        arima_result = advanced_ml_engine.auto_arima_forecasting(
            prices, forecast_horizon=forecast_horizon
        )
        
        # Ensemble prediction (weighted average)
        ensemble_forecast = (
            0.4 * lstm_result['forecast'] +
            0.3 * (prices[-1] * (1 + xgb_result['predictions'])) +
            0.3 * arima_result['forecast']
        )
        
        return {
            'ensemble_forecast': ensemble_forecast.tolist(),
            'lstm_forecast': lstm_result['forecast'].tolist(),
            'xgboost_returns': xgb_result['predictions'].tolist(),
            'arima_forecast': arima_result['forecast'].tolist(),
            'confidence_interval': arima_result['confidence_interval'].tolist(),
            'current_price': float(prices[-1]),
            'forecast_horizon': forecast_horizon
        }
    
    async def detect_market_regime(
        self,
        db: Session,
        asset_id: int
    ) -> Dict[str, any]:
        """
        Detect current market regime using Hidden Markov Model
        """
        from app.services.advanced_analytics import advanced_analytics
        
        price_history = db.query(PriceHistory).filter(
            PriceHistory.asset_id == asset_id
        ).order_by(PriceHistory.timestamp.desc()).limit(200).all()
        
        if len(price_history) < 100:
            return {'error': 'Insufficient data'}
        
        prices = np.array([ph.close for ph in reversed(price_history)])
        returns = np.diff(prices) / prices[:-1]
        
        # Detect regimes
        hmm_result = advanced_analytics.hidden_markov_model(returns, n_states=2)
        
        return {
            'current_regime': int(hmm_result['current_regime']),
            'regime_stats': hmm_result['regime_stats'],
            'transition_matrix': hmm_result['transition_matrix'].tolist()
        }

ml_engine = MLRecommendationEngine()
