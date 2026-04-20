import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def run_lstm(series, return_pred=False):
    """
    series: 1D array of ozone values
    return_pred: if True, returns both RMSE and predicted values
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1,1))

    X, y = [], []
    for i in range(3, len(scaled)):
        X.append(scaled[i-3:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)

    pred = model.predict(X, verbose=0)
    rmse = np.sqrt(mean_squared_error(y, pred))

    if return_pred:
        # Rescale predictions to original scale
        pred_full = np.zeros_like(series)
        pred_full[:3] = series[:3]  # first 3 cannot be predicted
        pred_full[3:] = scaler.inverse_transform(pred).flatten()
        return rmse, pred_full

    return rmse
