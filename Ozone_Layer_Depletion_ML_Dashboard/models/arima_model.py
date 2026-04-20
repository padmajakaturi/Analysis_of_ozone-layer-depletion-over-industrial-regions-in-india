from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

def run_arima(series, return_pred=False):
    """
    series: 1D array of ozone values
    return_pred: if True, returns both RMSE and predicted values
    """
    model = ARIMA(series, order=(1,1,1))
    model_fit = model.fit()
    pred = model_fit.predict()
    rmse = np.sqrt(mean_squared_error(series[1:], pred[1:]))
    
    if return_pred:
        return rmse, pred
    return rmse
