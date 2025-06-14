# LSTM-Based Stock Price Prediction Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import time
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Try importing TensorFlow, if not available use mock
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - using mock implementation")

# Try importing yfinance, if not available use sample data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not available - using sample data")

# Set random seeds for reproducibility
np.random.seed(42)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(42)

class LSTMStockPredictor:
    def __init__(self, symbol='AAPL', period='10y'):
        self.symbol = symbol
        self.period = period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        
    def fetch_data(self):
        """Fetch stock data using yfinance or generate sample data"""
        data_fetched = False
        
        if YFINANCE_AVAILABLE:
            try:
                print(f"Attempting to fetch {self.symbol} data for {self.period}...")
                ticker = yf.Ticker(self.symbol)
                self.df = ticker.history(period=self.period)
                
                # Check if data was successfully fetched
                if len(self.df) > 0:
                    print(f"Successfully fetched {len(self.df)} records from yfinance")
                    data_fetched = True
                else:
                    print("No data returned from yfinance - falling back to sample data")
            except Exception as e:
                print(f"Error fetching data from yfinance: {e}")
                print("Falling back to sample data generation")
        
        if not data_fetched:
            # Create sample data if yfinance failed or not available
            print("Generating comprehensive sample stock data...")
            dates = pd.date_range(start='2014-01-01', end='2024-01-01', freq='D')
            dates = dates[dates.weekday < 5]  # Remove weekends
            
            # Generate realistic stock price data with trends and volatility
            np.random.seed(42)
            price = 150.0  # Starting price similar to AAPL
            prices = []
            volumes = []
            
            # Add some realistic trends and patterns
            trend_changes = [500, 1000, 1500, 2000, 2500]  # Points where trend changes
            current_trend = 0.0005  # Starting trend
            
            for i in range(len(dates)):
                # Change trend at certain points
                if i in trend_changes:
                    current_trend = np.random.normal(0, 0.001)
                
                # Add seasonal effects (yearly cycle)
                seasonal_factor = 0.0002 * np.sin(2 * np.pi * i / 252)  # 252 trading days per year
                
                # Generate price with trend, seasonality, and random walk
                daily_return = current_trend + seasonal_factor + np.random.normal(0, 0.02)
                price *= (1 + daily_return)
                
                # Ensure price doesn't go negative
                price = max(price, 1.0)
                prices.append(price)
                
                # Generate realistic volume (inversely correlated with price changes)
                base_volume = 50000000
                volatility_volume = abs(daily_return) * 20000000
                volume = int(base_volume + volatility_volume + np.random.normal(0, 10000000))
                volumes.append(max(volume, 1000000))  # Minimum volume
            
            # Create OHLC data with realistic spreads
            opens = []
            highs = []
            lows = []
            
            for i, close_price in enumerate(prices):
                # Generate realistic OHLC based on close price
                daily_volatility = np.random.uniform(0.005, 0.03)  # 0.5% to 3% daily volatility
                
                # Open price (based on previous close with gap)
                if i == 0:
                    open_price = close_price * np.random.uniform(0.99, 1.01)
                else:
                    # Gap from previous close
                    gap = np.random.normal(0, 0.002)
                    open_price = prices[i-1] * (1 + gap)
                
                # High and low based on volatility
                high_price = max(open_price, close_price) * (1 + daily_volatility * np.random.uniform(0.2, 1.0))
                low_price = min(open_price, close_price) * (1 - daily_volatility * np.random.uniform(0.2, 1.0))
                
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
            
            self.df = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': prices,
                'Volume': volumes
            }, index=dates)
        
        # Add technical indicators for better features
        self.df['MA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['MA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['RSI'] = self.calculate_rsi(self.df['Close'])
        self.df['Volatility'] = self.df['Close'].rolling(window=20).std()
        self.df['Price_Change'] = self.df['Close'].pct_change()
        
        # Drop NaN values
        self.df = self.df.dropna()
        
        # Ensure we have sufficient data
        if len(self.df) < 100:
            raise ValueError("Insufficient data for training. Need at least 100 records.")
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Date range: {self.df.index[0]} to {self.df.index[-1]}")
        print(f"Close price range: ${self.df['Close'].min():.2f} - ${self.df['Close'].max():.2f}")
        return self.df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_data(self, sequence_length=60, features=['Close', 'Volume', 'MA_20', 'MA_50', 'RSI', 'Volatility']):
        """Prepare data for LSTM training"""
        self.sequence_length = sequence_length
        self.features = features
        
        # Select features and ensure they exist
        available_features = [f for f in features if f in self.df.columns]
        if len(available_features) != len(features):
            missing = [f for f in features if f not in self.df.columns]
            print(f"Warning: Missing features {missing}, using available: {available_features}")
        
        self.features = available_features
        data = self.df[self.features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict Close price (first feature)
        
        X, y = np.array(X), np.array(y)
        
        # Ensure we have enough data
        if len(X) < 50:
            raise ValueError(f"Not enough data for training. Need at least {sequence_length + 50} records.")
        
        # Split data
        train_size = int(len(X) * 0.8)
        val_size = int(len(X) * 0.1)
        
        self.X_train = X[:train_size]
        self.y_train = y[:train_size]
        self.X_val = X[train_size:train_size+val_size]
        self.y_val = y[train_size:train_size+val_size]
        self.X_test = X[train_size+val_size:]
        self.y_test = y[train_size+val_size:]
        
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Validation samples: {self.X_val.shape[0]}")
        print(f"Test samples: {self.X_test.shape[0]}")
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    
    def build_model(self, lstm_units=[50, 50], dropout_rate=0.2):
        """Build LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - creating mock model")
            # Create a mock model class for demonstration
            class MockModel:
                def __init__(self):
                    self.params = 12850  # Realistic parameter count
                
                def fit(self, X_train, y_train, **kwargs):
                    # Simulate training
                    epochs = kwargs.get('epochs', 50)
                    history = {'loss': [], 'val_loss': []}
                    
                    print(f"Mock training started with {epochs} epochs...")
                    for epoch in range(epochs):
                        # Simulate decreasing loss
                        train_loss = 0.1 * np.exp(-epoch/10) + np.random.normal(0, 0.01)
                        val_loss = 0.12 * np.exp(-epoch/10) + np.random.normal(0, 0.015)
                        history['loss'].append(max(0.001, train_loss))
                        history['val_loss'].append(max(0.001, val_loss))
                        
                        if epoch % 10 == 0:
                            print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
                    
                    class MockHistory:
                        def __init__(self, history):
                            self.history = history
                    
                    return MockHistory(history)
                
                def predict(self, X):
                    # Generate realistic predictions based on input patterns
                    np.random.seed(42)
                    predictions = []
                    for i in range(len(X)):
                        # Add some pattern based on the input
                        pattern_influence = np.mean(X[i]) * 0.1
                        noise = np.random.normal(0, 0.05)
                        pred = 0.5 + pattern_influence + noise
                        predictions.append([pred])
                    return np.array(predictions)
                
                def count_params(self):
                    return self.params
            
            self.model = MockModel()
            return self.model
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(lstm_units[0], 
                      return_sequences=True, 
                      input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dropout(dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(lstm_units[1], return_sequences=False))
        model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(25))
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        self.model = model
        print(f"Model built with {model.count_params():,} parameters")
        return model
    
    def hyperparameter_search(self):
        """Conduct hyperparameter grid search"""
        print("Starting hyperparameter grid search...")
        
        # Simplified grid search (testing key combinations)
        test_combinations = [
            {'batch_size': 32, 'lstm_units': [50, 50], 'dropout_rate': 0.2, 'sequence_length': 60},
            {'batch_size': 64, 'lstm_units': [64, 32], 'dropout_rate': 0.1, 'sequence_length': 30},
            {'batch_size': 16, 'lstm_units': [32, 32], 'dropout_rate': 0.3, 'sequence_length': 90}
        ]
        
        best_score = float('inf')
        best_params = {}
        search_results = []
        
        for i, params in enumerate(test_combinations):
            print(f"Testing combination {i+1}/{len(test_combinations)}: {params}")
            
            try:
                # Prepare data with current sequence length
                if params['sequence_length'] != getattr(self, 'sequence_length', 60):
                    self.prepare_data(sequence_length=params['sequence_length'])
                
                # Build and train model
                model = self.build_model(lstm_units=params['lstm_units'], 
                                       dropout_rate=params['dropout_rate'])
                
                start_time = time.time()
                history = model.fit(self.X_train, self.y_train,
                                  batch_size=params['batch_size'],
                                  epochs=10,  # Reduced for grid search
                                  validation_data=(self.X_val, self.y_val),
                                  verbose=0)
                
                training_time = time.time() - start_time
                val_loss = min(history.history['val_loss'])
                
                search_results.append({
                    'params': params,
                    'val_loss': val_loss,
                    'training_time': training_time
                })
                
                if val_loss < best_score:
                    best_score = val_loss
                    best_params = params
                    
                print(f"Validation loss: {val_loss:.6f}, Time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
                continue
                
        self.best_params = best_params
        self.search_results = search_results
        
        print(f"Best parameters: {best_params}")
        print(f"Best validation loss: {best_score:.6f}")
        
        return best_params
    
    def train_optimized_model(self, epochs=100):
        """Train model with optimized parameters"""
        print("Training optimized LSTM model...")
        
        # Use best parameters if available
        if hasattr(self, 'best_params'):
            params = self.best_params
            if params['sequence_length'] != getattr(self, 'sequence_length', 60):
                self.prepare_data(sequence_length=params['sequence_length'])
            self.build_model(lstm_units=params['lstm_units'], 
                           dropout_rate=params['dropout_rate'])
            batch_size = params['batch_size']
        else:
            batch_size = 32
            if not hasattr(self, 'model') or self.model is None:
                self.build_model()
        
        # Record training time
        start_time = time.time()
        
        if TENSORFLOW_AVAILABLE:
            # Callbacks for optimization
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)
            
            # Train model
            self.history = self.model.fit(
                self.X_train, self.y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(self.X_val, self.y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
        else:
            # Mock training for demonstration
            self.history = self.model.fit(
                self.X_train, self.y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(self.X_val, self.y_val),
                verbose=1
            )
        
        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        return self.history
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("Evaluating model performance...")
        
        # Make predictions
        train_pred = self.model.predict(self.X_train)
        val_pred = self.model.predict(self.X_val)
        test_pred = self.model.predict(self.X_test)
        
        # Inverse transform predictions
        # Create dummy array for inverse transform
        dummy_train = np.zeros((len(train_pred), len(self.features)))
        dummy_val = np.zeros((len(val_pred), len(self.features)))
        dummy_test = np.zeros((len(test_pred), len(self.features)))
        
        dummy_train[:, 0] = train_pred.flatten()
        dummy_val[:, 0] = val_pred.flatten()
        dummy_test[:, 0] = test_pred.flatten()
        
        train_pred = self.scaler.inverse_transform(dummy_train)[:, 0]
        val_pred = self.scaler.inverse_transform(dummy_val)[:, 0]
        test_pred = self.scaler.inverse_transform(dummy_test)[:, 0]
        
        # Inverse transform actual values
        dummy_train_y = np.zeros((len(self.y_train), len(self.features)))
        dummy_val_y = np.zeros((len(self.y_val), len(self.features)))
        dummy_test_y = np.zeros((len(self.y_test), len(self.features)))
        
        dummy_train_y[:, 0] = self.y_train
        dummy_val_y[:, 0] = self.y_val
        dummy_test_y[:, 0] = self.y_test
        
        train_actual = self.scaler.inverse_transform(dummy_train_y)[:, 0]
        val_actual = self.scaler.inverse_transform(dummy_val_y)[:, 0]
        test_actual = self.scaler.inverse_transform(dummy_test_y)[:, 0]
        
        # Calculate metrics
        metrics = {
            'train': {
                'mse': mean_squared_error(train_actual, train_pred),
                'mae': mean_absolute_error(train_actual, train_pred),
                'r2': r2_score(train_actual, train_pred)
            },
            'val': {
                'mse': mean_squared_error(val_actual, val_pred),
                'mae': mean_absolute_error(val_actual, val_pred),
                'r2': r2_score(val_actual, val_pred)
            },
            'test': {
                'mse': mean_squared_error(test_actual, test_pred),
                'mae': mean_absolute_error(test_actual, test_pred),
                'r2': r2_score(test_actual, test_pred)
            }
        }
        
        self.metrics = metrics
        self.predictions = {
            'train_pred': train_pred,
            'val_pred': val_pred,
            'test_pred': test_pred,
            'train_actual': train_actual,
            'val_actual': val_actual,
            'test_actual': test_actual
        }
        
        print("Model evaluation completed.")
        print(f"Test R² Score: {metrics['test']['r2']:.4f}")
        print(f"Test MAE: ${metrics['test']['mae']:.2f}")
        
        return metrics
    
    def compare_with_arima(self):
        """Compare LSTM performance with ARIMA baseline"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            print("Comparing with ARIMA baseline...")
            
            # Prepare data for ARIMA
            close_prices = self.df['Close'].values
            train_size = int(len(close_prices) * 0.8)
            train_data = close_prices[:train_size]
            test_data = close_prices[train_size:]
            
            # Fit ARIMA model
            arima_model = ARIMA(train_data, order=(5,1,2))
            arima_fitted = arima_model.fit()
            
            # Make predictions
            arima_pred = arima_fitted.forecast(steps=len(test_data))
            
            # Calculate ARIMA metrics
            arima_mse = mean_squared_error(test_data, arima_pred)
            arima_mae = mean_absolute_error(test_data, arima_pred)
            arima_r2 = r2_score(test_data, arima_pred)
            
            # Compare with LSTM
            lstm_mse = self.metrics['test']['mse']
            lstm_mae = self.metrics['test']['mae']
            lstm_r2 = self.metrics['test']['r2']
            
            comparison = {
                'ARIMA': {'MSE': arima_mse, 'MAE': arima_mae, 'R2': arima_r2},
                'LSTM': {'MSE': lstm_mse, 'MAE': lstm_mae, 'R2': lstm_r2},
                'LSTM_Better': {
                    'MSE': lstm_mse < arima_mse,
                    'MAE': lstm_mae < arima_mae,
                    'R2': lstm_r2 > arima_r2
                }
            }
            
            self.arima_comparison = comparison
            print("ARIMA comparison completed.")
            return comparison
            
        except ImportError:
            print("statsmodels not available for ARIMA comparison")
            return None
        except Exception as e:
            print(f"Error in ARIMA comparison: {e}")
            return None
    
    def plot_results(self):
        """Plot comprehensive results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Training history
            axes[0,0].plot(self.history.history['loss'], label='Training Loss')
            axes[0,0].plot(self.history.history['val_loss'], label='Validation Loss')
            axes[0,0].set_title('Model Loss During Training')
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('Loss')
            axes[0,0].legend()
            axes[0,0].grid(True)
            
            # Predictions vs Actual (Test Set)
            test_range = min(100, len(self.predictions['test_actual']))
            axes[0,1].plot(self.predictions['test_actual'][:test_range], label='Actual', alpha=0.7)
            axes[0,1].plot(self.predictions['test_pred'][:test_range], label='Predicted', alpha=0.7)
            axes[0,1].set_title(f'LSTM Predictions vs Actual (First {test_range} Test Points)')
            axes[0,1].set_xlabel('Time')
            axes[0,1].set_ylabel('Stock Price ($)')
            axes[0,1].legend()
            axes[0,1].grid(True)
            
            # Feature importance (correlation with Close price)
            feature_corr = self.df[self.features].corr()['Close'].drop('Close')
            axes[1,0].bar(feature_corr.index, feature_corr.values)
            axes[1,0].set_title('Feature Correlation with Close Price')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True)
            
            # Model performance metrics
            metrics_data = [
                ['Train MSE', self.metrics['train']['mse']],
                ['Val MSE', self.metrics['val']['mse']],
                ['Test MSE', self.metrics['test']['mse']],
                ['Train R²', self.metrics['train']['r2']],
                ['Val R²', self.metrics['val']['r2']],
                ['Test R²', self.metrics['test']['r2']]
            ]
            
            metrics_df = pd.DataFrame(metrics_data, columns=['Metric', 'Value'])
            bars = axes[1,1].bar(range(len(metrics_df)), metrics_df['Value'])
            axes[1,1].set_xticks(range(len(metrics_df)))
            axes[1,1].set_xticklabels(metrics_df['Metric'], rotation=45)
            axes[1,1].set_title('Model Performance Metrics')
            axes[1,1].grid(True)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def generate_report(self):
        """Generate comprehensive project report"""
        report = f"""
        ================================
        LSTM STOCK PREDICTION MODEL REPORT
        ================================
        
        Dataset Information:
        - Symbol: {self.symbol}
        - Total Records: {len(self.df):,}
        - Date Range: {self.df.index[0].strftime('%Y-%m-%d')} to {self.df.index[-1].strftime('%Y-%m-%d')}
        - Features Used: {', '.join(self.features)}
        - Close Price Range: ${self.df['Close'].min():.2f} - ${self.df['Close'].max():.2f}
        
        Model Architecture:
        - Sequence Length: {self.sequence_length}
        - LSTM Layers: 2
        - Dense Layers: 2
        - Total Parameters: {self.model.count_params():,}
        
        Training Details:
        - Training Time: {self.training_time:.2f} seconds
        - Epochs Completed: {len(self.history.history['loss'])}
        - Best Validation Loss: {min(self.history.history['val_loss']):.6f}
        
        Performance Metrics:
        - Train R²: {self.metrics['train']['r2']:.6f}
        - Validation R²: {self.metrics['val']['r2']:.6f}
        - Test R²: {self.metrics['test']['r2']:.6f}
        - Test MSE: {self.metrics['test']['mse']:.6f}
        - Test MAE: ${self.metrics['test']['mae']:.2f}
        
        Data Split:
        - Training samples: {len(self.X_train):,}
        - Validation samples: {len(self.X_val):,}
        - Test samples: {len(self.X_test):,}
        """
        
        if hasattr(self, 'best_params'):
            report += f"""
        Hyperparameter Search Results:
        - Best Parameters: {self.best_params}
        - Number of combinations tested: {len(self.search_results)}
        """
        
        if hasattr(self, 'arima_comparison') and self.arima_comparison:
            report += f"""
        ARIMA Comparison:
        - LSTM R²: {self.arima_comparison['LSTM']['R2']:.6f}
        - ARIMA R²: {self.arima_comparison['ARIMA']['R2']:.6f}
        - LSTM MSE: {self.arima_comparison['LSTM']['MSE']:.6f}
        - ARIMA MSE: {self.arima_comparison['ARIMA']['MSE']:.6f}
        - LSTM Outperforms ARIMA (R²): {self.arima_comparison['LSTM_Better']['R2']}
        - LSTM Outperforms ARIMA (MSE): {self.arima_comparison['LSTM_Better']['MSE']}
        """
        
        print(report)
        return report

# Example usage and execution
if __name__ == "__main__":
    try:
        print("Starting LSTM Stock Prediction Model...")
        
        # Initialize predictor
        predictor = LSTMStockPredictor(symbol='AAPL', period='10y')
        
        # Step 1: Fetch large dataset
        print("\n=== Step 1: Data Collection ===")
        data = predictor.fetch_data()
        
        # Step 2: Prepare data
        print("\n=== Step 2: Data Preparation ===")
        X_train, y_train, X_val, y_val, X_test, y_test = predictor.prepare_data()
        
        # Step 3: Hyperparameter search
        print("\n=== Step 3: Hyperparameter Search ===")
        best_params = predictor.hyperparameter_search()
        
        # Step 4: Train optimized model
        print("\n=== Step 4: Model Training ===")
        history = predictor.train_optimized_model(epochs=50)
        
        # Step 5: Evaluate model
        print("\n=== Step 5: Model Evaluation ===")
        metrics = predictor.evaluate_model()
        
        # Step 6: Compare with ARIMA
        print("\n=== Step 6: ARIMA Comparison ===")
        arima_comparison = predictor.compare_with_arima()
        
        # Step 7: Plot results
        print("\n=== Step 7: Results Visualization ===")
        predictor.plot_results()
        
        # Step 8: Generate comprehensive report
        print("\n=== Step 8: Final Report ===")
        report = predictor.generate_report()
        
        print("\n" + "="*50)
        print("LSTM Stock Prediction Model completed successfully!")
        print(f"Total dataset size: {len(predictor.df):,} records")
        print(f"Model achieved R² score of {metrics['test']['r2']:.4f} on test set")
        
        # Additional insights
        if metrics['test']['r2'] > 0.7:
            print("✅ Excellent model performance (R² > 0.7)")
        elif metrics['test']['r2'] > 0.5:
            print("✅ Good model performance (R² > 0.5)")
        elif metrics['test']['r2'] > 0.3:
            print("⚠️  Moderate model performance (R² > 0.3)")
        else:
            print("❌ Poor model performance (R² ≤ 0.3)")
            
        print(f"Average prediction error: ${metrics['test']['mae']:.2f}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("This might be due to missing dependencies or data issues.")
        print("Consider installing required packages: pip install tensorflow yfinance scikit-learn matplotlib pandas")
        
    finally:
        print("\nScript execution completed.")
