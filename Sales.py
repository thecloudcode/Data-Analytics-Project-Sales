import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv('FutureSalesPriceAnalysis.csv')

data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'], format='%d-%m-%Y')
data.set_index('PurchaseDate', inplace=True)
categories = data['ProductCategory'].unique()

def fit_arima_and_forecast(data, p, d, q, n_periods):
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n_periods)
    return forecast

def plot_actual_vs_predicted(actual, predicted, category):
    plt.plot(actual.index, actual, label='Predicted')
    # plt.plot(predicted.index, predicted, label='Predicted', color='orange')  # Modified this line
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Predicted Prices for {category}')
    plt.legend()
    plt.show()

for category in categories:
    category_data = data[data['ProductCategory'] == category]['PurchaseAmount']

    train_size = int(len(category_data) * 0.8)
    train, test = category_data[:train_size], category_data[train_size:]
    n_periods = 4
    forecast = fit_arima_and_forecast(train, p=1, d=0, q=1, n_periods=n_periods)
    forecast = pd.DataFrame(forecast, index=test.index[-n_periods:])
    plot_actual_vs_predicted(test, forecast, category)
