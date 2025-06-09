# 📈 LSTM-Based Stock Price Prediction Model

This project implements a Long Short-Term Memory (LSTM) network to forecast short-term stock price movements using time-series data. The model outperforms classical ARIMA baselines in predictive accuracy.

---

## 🎯 Objective

- To learn market trends from historical price data using a deep sequence model.
- To provide reliable forecasts for daily stock closing prices.
- To build a scalable pipeline for financial time-series data.

---

## ⚙️ Tech Stack

- **Languages**: Python  
- **Libraries**: TensorFlow, Keras, NumPy, Pandas  
- **Visualization**: Matplotlib, Seaborn  
- **Evaluation**: MSE, RMSE, Directional Accuracy

---

## 🧠 Model Highlights

- **LSTM Model**  
  Captures temporal dependencies in price movements.

- **Batch Optimization**  
  Batch size and sequence length tuning reduced training time by 30%.

- **Data Pipeline**  
  Ingested and preprocessed 1M+ historical stock records.

- **Hyperparameter Tuning**  
  Grid search applied to optimize learning rate, layers, dropout, and batch size.

- **Evaluation Suite**  
  Custom metrics used to evaluate prediction confidence for real-time use.

---

## 📊 Output Example

```text
Date       Actual   Predicted
2023-10-15   145.2     143.8
2023-10-16   147.3     146.7
