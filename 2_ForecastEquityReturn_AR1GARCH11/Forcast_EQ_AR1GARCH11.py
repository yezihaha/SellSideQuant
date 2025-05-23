# garch_forecast.py

import os
import numpy as np
import pandas as pd
from arch import arch_model
from alpha_vantage.timeseries import TimeSeries


def load_data(api_key: str, cache_file='TSLA_AlphaVantage.csv') -> pd.Series:
    if os.path.exists(cache_file):
        raw_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        ts = TimeSeries(key=api_key, output_format='pandas')
        raw_data, _ = ts.get_daily(symbol='TSLA', outputsize='full')
        raw_data.index = pd.to_datetime(raw_data.index)
        raw_data.sort_index(inplace=True)
        raw_data.to_csv(cache_file)

    close_prices = raw_data['4. close']['2016-01-01':'2025-05-01']
    returns = np.log(close_prices).diff().dropna()
    returns = returns[returns.abs() <= 0.5]
    return returns


def rolling_garch_forecast(log_returns: pd.Series, initial_window='2016-01-01:2016-12-31',
                            sim_period='2017-01-01:2025-05-01', window_size=252, n_sim=1000):

    train_data = log_returns[initial_window.split(':')[0]:initial_window.split(':')[1]]
    forecast_dates = log_returns[sim_period.split(':')[0]:sim_period.split(':')[1]].index

    mean_forecasts, quantile_1, quantile_99, actuals = [], [], [], []
    num_outliers = 0

    for i, date in enumerate(forecast_dates[:-1]):
        window_data = train_data if i == 0 else log_returns.loc[:date].iloc[-window_size:]
        forecast_date = forecast_dates[i+1]

        model = arch_model(window_data, vol='Garch', p=1, q=1, mean='AR', lags=1, dist='normal')
        res = model.fit(disp='off')

        params = res.params
        mu = params['Const']
        phi = params[window_data.name + '[1]'] if f'{window_data.name}[1]' in params else params.iloc[1]

        last_return = window_data.iloc[-1]
        forecast = res.forecast(horizon=1, reindex=False)
        sigma = np.sqrt(forecast.variance.iloc[0, 0])

        shocks = np.random.normal(0, sigma, n_sim)
        simulated_returns = mu + phi * last_return + shocks

        mean_forecasts.append(np.mean(simulated_returns))
        quantile_1_sim = np.quantile(simulated_returns, 0.01)
        quantile_99_sim = np.quantile(simulated_returns, 0.99)
        quantile_1.append(quantile_1_sim)
        quantile_99.append(quantile_99_sim)

        realized_return = log_returns.loc[forecast_date]
        actuals.append(realized_return)
        if realized_return < quantile_1_sim or realized_return > quantile_99_sim:
            num_outliers += 1


    forecast_index = forecast_dates[:-1]
    return (
        pd.Series(mean_forecasts, index=forecast_index),
        pd.Series(quantile_1, index=forecast_index),
        pd.Series(quantile_99, index=forecast_index),
        pd.Series(actuals, index=forecast_index),
        num_outliers,
        num_outliers / len(forecast_dates)
    )
