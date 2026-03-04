"""
LSTM-Based Stock Price Prediction Model
- 5+ years historical stock data via yfinance
- MinMaxScaler + 60-day sliding window
- RMSE, MAE, R² evaluation
- Matplotlib visualization
"""

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================================================================
# CONFIG
# ==============================================================================
TICKER = 'AAPL'
PERIOD = '5y'
SEQ_LENGTH = 60  # 60-day sliding window

# ==============================================================================
# 1. FETCH DATA (5+ years)
# ==============================================================================
print(f"\n[1] Fetching {PERIOD} of {TICKER} data...")
df = yf.download(TICKER, period=PERIOD, progress=False, multi_level_index=False)
prices = df[['Close']].values
print(f"    Records: {len(prices)}")

# ==============================================================================
# 2. PREPROCESSING — MinMax scaling + 60-day sliding window
# ==============================================================================
print("\n[2] Preprocessing...")

# 70/30 chronological split
train_size = int(len(prices) * 0.70)

# Fit scaler on training data only
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(prices[:train_size])
scaled = scaler.transform(prices)

# Create 60-day sliding window sequences
X, y = [], []
for i in range(SEQ_LENGTH, len(scaled)):
    X.append(scaled[i - SEQ_LENGTH:i])
    y.append(scaled[i])

X, y = np.array(X), np.array(y)

# Split sequences
split = int(len(X) * 0.70)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

# ==============================================================================
# 3. BUILD & TRAIN LSTM
# ==============================================================================
print("\n[3] Building LSTM model...")
model = Sequential([
    Input(shape=(SEQ_LENGTH, 1)),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

print("\n[4] Training...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50, batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# ==============================================================================
# 4. EVALUATE — RMSE, MAE, R²
# ==============================================================================
print("\n[5] Evaluating...")
preds_scaled = model.predict(X_test)
preds = scaler.inverse_transform(preds_scaled)
actuals = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(actuals, preds))
mae = mean_absolute_error(actuals, preds)
r2 = r2_score(actuals, preds)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R²   : {r2:.4f}")
print("=" * 50)

# ==============================================================================
# 5. VISUALIZE — Actual vs Predicted
# ==============================================================================
plt.figure(figsize=(12, 5))
plt.plot(actuals, label='Actual Price', color='blue', linewidth=1.2)
plt.plot(preds, label='Predicted Price', color='red', linewidth=1.2, alpha=0.8)
plt.title(f'{TICKER} Stock Price — Actual vs Predicted (LSTM)')
plt.xlabel('Test Samples')
plt.ylabel('Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('lstm_prediction.png', dpi=200)
plt.show()
print("\nPlot saved as lstm_prediction.png")
