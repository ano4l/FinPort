import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

class AdvancedMLEngine:
    """
    Advanced Machine Learning models for financial prediction and portfolio optimization.
    Includes deep learning, ensemble methods, and time series forecasting.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
    
    # ==================== GRADIENT BOOSTING MODELS ====================
    
    def xgboost_price_prediction(
        self,
        features: np.ndarray,
        target: np.ndarray,
        forecast_horizon: int = 5
    ) -> Dict[str, any]:
        """
        XGBoost for price prediction with advanced hyperparameters
        
        Uses gradient boosting with regularization to prevent overfitting
        """
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # XGBoost parameters optimized for financial data
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'early_stopping_rounds': 50
        }
        
        # Train model
        model = xgb.XGBRegressor(**params)
        
        # Cross-validation scores
        cv_scores = []
        for train_idx, val_idx in tscv.split(features):
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = target[train_idx], target[val_idx]
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            score = model.score(X_val, y_val)
            cv_scores.append(score)
        
        # Final model on all data
        model.fit(features, target)
        
        # Feature importance
        importance = model.feature_importances_
        
        # Forecast
        last_features = features[-1:].copy()
        predictions = []
        
        for _ in range(forecast_horizon):
            pred = model.predict(last_features)[0]
            predictions.append(pred)
            # Update features for next prediction (rolling forecast)
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
        
        return {
            'predictions': np.array(predictions),
            'feature_importance': importance,
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'model': model
        }
    
    def gradient_boosting_ensemble(
        self,
        features: np.ndarray,
        target: np.ndarray
    ) -> Dict[str, any]:
        """
        Ensemble of gradient boosting models with different configurations
        Combines XGBoost, LightGBM-style, and CatBoost-style approaches
        """
        models = []
        predictions = []
        
        # XGBoost configuration
        xgb_model = xgb.XGBRegressor(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8
        )
        xgb_model.fit(features, target)
        models.append(('xgboost', xgb_model))
        predictions.append(xgb_model.predict(features))
        
        # Gradient Boosting (sklearn)
        gb_model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8
        )
        gb_model.fit(features, target)
        models.append(('gradient_boost', gb_model))
        predictions.append(gb_model.predict(features))
        
        # Random Forest for diversity
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5
        )
        rf_model.fit(features, target)
        models.append(('random_forest', rf_model))
        predictions.append(rf_model.predict(features))
        
        # Ensemble prediction (weighted average)
        weights = np.array([0.4, 0.4, 0.2])  # XGB and GB get more weight
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return {
            'models': models,
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': predictions,
            'weights': weights
        }
    
    # ==================== DEEP LEARNING MODELS ====================
    
    def lstm_price_forecasting(
        self,
        prices: np.ndarray,
        sequence_length: int = 60,
        forecast_horizon: int = 5,
        epochs: int = 100
    ) -> Dict[str, any]:
        """
        LSTM (Long Short-Term Memory) neural network for time series forecasting
        
        Captures long-term dependencies in price movements
        """
        # Normalize data
        scaler = StandardScaler()
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(len(prices_scaled) - sequence_length - forecast_horizon + 1):
            X.append(prices_scaled[i:i+sequence_length])
            y.append(prices_scaled[i+sequence_length:i+sequence_length+forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, 1)),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(forecast_horizon)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        
        # Forecast
        last_sequence = prices_scaled[-sequence_length:].reshape(1, sequence_length, 1)
        forecast_scaled = model.predict(last_sequence, verbose=0)[0]
        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        
        return {
            'forecast': forecast,
            'test_loss': test_loss,
            'training_history': history.history,
            'model': model
        }
    
    def attention_based_forecasting(
        self,
        features: np.ndarray,
        target: np.ndarray,
        sequence_length: int = 30
    ) -> Dict[str, any]:
        """
        Transformer-based model with attention mechanism
        
        Learns which time steps are most important for prediction
        """
        # Prepare sequences
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Build attention model
        inputs = layers.Input(shape=(sequence_length, X.shape[-1]))
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32
        )(inputs, inputs)
        
        attention_output = layers.Dropout(0.1)(attention_output)
        attention_output = layers.LayerNormalization()(attention_output + inputs)
        
        # Feed-forward network
        ffn_output = layers.Dense(128, activation='relu')(attention_output)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        ffn_output = layers.Dense(X.shape[-1])(ffn_output)
        ffn_output = layers.LayerNormalization()(ffn_output + attention_output)
        
        # Global pooling and output
        pooled = layers.GlobalAveragePooling1D()(ffn_output)
        outputs = layers.Dense(1)(pooled)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train
        history = model.fit(
            X_scaled, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        return {
            'model': model,
            'training_history': history.history
        }
    
    def variational_autoencoder_anomaly(
        self,
        returns: np.ndarray,
        latent_dim: int = 10
    ) -> Dict[str, any]:
        """
        Variational Autoencoder for anomaly detection in returns
        
        Identifies unusual market conditions or potential risks
        """
        # Normalize
        returns_scaled = self.scaler.fit_transform(returns.reshape(-1, 1))
        
        # Encoder
        encoder_inputs = layers.Input(shape=(1,))
        x = layers.Dense(32, activation='relu')(encoder_inputs)
        x = layers.Dense(16, activation='relu')(x)
        
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)
        
        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling)([z_mean, z_log_var])
        
        # Decoder
        decoder_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(16, activation='relu')(decoder_inputs)
        x = layers.Dense(32, activation='relu')(x)
        decoder_outputs = layers.Dense(1)(x)
        
        # Models
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z])
        decoder = keras.Model(decoder_inputs, decoder_outputs)
        
        # VAE
        vae_outputs = decoder(encoder(encoder_inputs)[2])
        vae = keras.Model(encoder_inputs, vae_outputs)
        
        # Loss
        reconstruction_loss = keras.losses.mse(encoder_inputs, vae_outputs)
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        vae_loss = reconstruction_loss + kl_loss
        
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        
        # Train
        vae.fit(returns_scaled, epochs=100, batch_size=32, verbose=0)
        
        # Detect anomalies
        reconstructed = vae.predict(returns_scaled, verbose=0)
        reconstruction_error = np.abs(returns_scaled - reconstructed)
        
        # Anomaly threshold (95th percentile)
        threshold = np.percentile(reconstruction_error, 95)
        anomalies = reconstruction_error > threshold
        
        return {
            'anomalies': anomalies.flatten(),
            'reconstruction_error': reconstruction_error.flatten(),
            'threshold': float(threshold),
            'vae_model': vae
        }
    
    # ==================== TIME SERIES MODELS ====================
    
    def auto_arima_forecasting(
        self,
        prices: np.ndarray,
        forecast_horizon: int = 5
    ) -> Dict[str, any]:
        """
        Automatic ARIMA model selection and forecasting
        
        Finds optimal (p, d, q) parameters automatically
        """
        # Auto ARIMA
        model = auto_arima(
            prices,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        # Forecast
        forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
        
        # Model diagnostics
        aic = model.aic()
        bic = model.bic()
        
        return {
            'forecast': forecast,
            'confidence_interval': conf_int,
            'order': model.order,
            'aic': float(aic),
            'bic': float(bic),
            'model': model
        }
    
    def arima_garch_combined(
        self,
        returns: np.ndarray,
        forecast_horizon: int = 5
    ) -> Dict[str, any]:
        """
        Combined ARIMA-GARCH model
        
        ARIMA for mean, GARCH for volatility
        More accurate for financial returns
        """
        from arch import arch_model
        
        # Fit ARIMA for mean
        arima_model = auto_arima(
            returns,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True
        )
        
        # Get residuals
        residuals = arima_model.resid()
        
        # Fit GARCH on residuals
        garch_model = arch_model(
            residuals * 100,
            vol='Garch',
            p=1,
            q=1
        )
        garch_results = garch_model.fit(disp='off')
        
        # Forecast mean
        mean_forecast = arima_model.predict(n_periods=forecast_horizon)
        
        # Forecast volatility
        vol_forecast = garch_results.forecast(horizon=forecast_horizon)
        
        return {
            'mean_forecast': mean_forecast,
            'volatility_forecast': np.sqrt(vol_forecast.variance.values[-1, :]) / 100,
            'arima_order': arima_model.order,
            'garch_params': {
                'omega': float(garch_results.params['omega']),
                'alpha': float(garch_results.params['alpha[1]']),
                'beta': float(garch_results.params['beta[1]'])
            }
        }
    
    # ==================== REINFORCEMENT LEARNING ====================
    
    def q_learning_portfolio_optimization(
        self,
        returns: np.ndarray,
        n_assets: int,
        episodes: int = 1000
    ) -> Dict[str, any]:
        """
        Q-Learning for dynamic portfolio allocation
        
        Learns optimal rebalancing strategy through trial and error
        """
        # State: current portfolio weights
        # Action: rebalance to new weights
        # Reward: portfolio return
        
        n_states = 100  # Discretized state space
        n_actions = 10  # Discretized action space
        
        # Initialize Q-table
        Q = np.zeros((n_states, n_actions))
        alpha = 0.1  # Learning rate
        gamma = 0.95  # Discount factor
        epsilon = 0.1  # Exploration rate
        
        # Discretize weights
        def discretize_weights(weights):
            return int(np.sum(weights * np.arange(n_assets)) * 10) % n_states
        
        # Training
        total_rewards = []
        
        for episode in range(episodes):
            # Initialize random portfolio
            weights = np.random.dirichlet(np.ones(n_assets))
            state = discretize_weights(weights)
            
            episode_reward = 0
            
            for t in range(len(returns) - 1):
                # Choose action (epsilon-greedy)
                if np.random.random() < epsilon:
                    action = np.random.randint(n_actions)
                else:
                    action = np.argmax(Q[state])
                
                # Execute action (rebalance)
                new_weights = np.random.dirichlet(np.ones(n_assets))
                
                # Calculate reward
                portfolio_return = np.dot(weights, returns[t])
                reward = portfolio_return
                
                # Next state
                next_state = discretize_weights(new_weights)
                
                # Update Q-value
                Q[state, action] = Q[state, action] + alpha * (
                    reward + gamma * np.max(Q[next_state]) - Q[state, action]
                )
                
                state = next_state
                weights = new_weights
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        # Extract optimal policy
        optimal_actions = np.argmax(Q, axis=1)
        
        return {
            'Q_table': Q,
            'optimal_policy': optimal_actions,
            'training_rewards': total_rewards,
            'average_reward': np.mean(total_rewards[-100:])
        }
    
    # ==================== ENSEMBLE PREDICTION ====================
    
    def meta_learning_ensemble(
        self,
        features: np.ndarray,
        target: np.ndarray,
        forecast_horizon: int = 5
    ) -> Dict[str, any]:
        """
        Meta-learning ensemble that combines multiple models
        
        Uses stacking with a meta-learner to optimize predictions
        """
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import Ridge
        
        # Base models
        base_models = [
            ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=5)),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=10)),
            ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=4))
        ]
        
        # Meta-learner
        meta_learner = Ridge(alpha=1.0)
        
        # Stacking ensemble
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5
        )
        
        # Train
        stacking_model.fit(features, target)
        
        # Predictions
        predictions = stacking_model.predict(features[-forecast_horizon:])
        
        return {
            'predictions': predictions,
            'model': stacking_model,
            'base_models': base_models
        }

advanced_ml_engine = AdvancedMLEngine()
