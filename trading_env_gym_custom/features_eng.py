import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib
def create_features(df: pd.DataFrame, granularity: str = '1h'):
    # Map the window sizes based on the granularity
    window_sizes = {
        '1h': {'short': 24, 'long': 24*7, 'day': 24, 'week': 24*7, 'half_day': 12},
        '5min': {'short': 12, 'long': 12*24*7, 'day': 12*24, 'week': 12*24*7, 'half_day': 6}
    }
    windows = window_sizes.get(granularity, window_sizes['1h'])

    df['feature_hour'] = df.index.hour
    df['feature_dayofweek'] = df.index.dayofweek
    df['feature_month'] = df.index.month
    # df['feature_year'] = df.index.year
    # df['feature_dayofyear'] = df.index.dayofyear


    df['feature_volume'] = df['volume']
    df['feature_volume_week_mean'] = df['volume'].rolling(window=windows['week']).mean()
    df['feature_volume_day_mean'] = df['volume'].rolling(window=windows['day']).mean()
    df['feature_volume_day_std'] = df['volume'].rolling(window=windows['day']).std()
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]

    df['feature_return_1h'] = df['close'].pct_change(1)
    df['feature_return_24h'] = df['close'].pct_change(windows['day'])

    df['feature_volatility_24h'] = df['feature_close'].rolling(window=windows['day']).std()

    mean_close = df['close'].rolling(window=windows['day']).mean()
    std_close = df['close'].rolling(window=windows['day']).std()
    df['feature_large_swing'] = ((df['close'] - mean_close).abs() > 2.5 * std_close).astype(int)

    df['feature_short_MA'] = df['close'].rolling(window=windows['short']).mean()
    df['feature_long_MA'] = df['close'].rolling(window=windows['long']).mean()

    df['feature_market_trend'] = np.where(df['feature_short_MA'] > df['feature_long_MA'], 1,
                                          np.where(df['feature_short_MA'] < df['feature_long_MA'], -1, 0))

    df['feature_resistance'] = df['high'].rolling(window=windows['half_day']*2).max()
    df['feature_support'] = df['low'].rolling(window=windows['half_day']*2).min()

    df['feature_distance_from_resistance'] = df['close'] - df['feature_resistance']
    df['feature_distance_from_support'] = df['close'] - df['feature_support']

    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()

    df['feature_stochastic_oscillator'] = 100 * ((df['close'] - low_min) / (high_max - low_min))

    df.dropna(inplace=True)
    return df

def simple_reward(history):
    portfolio_valuation = history["portfolio_valuation"]
    if len(portfolio_valuation) < 2:
        return 0
    # Calculate the log return
    log_return = np.log(portfolio_valuation[-1] / portfolio_valuation[-2])
    return log_return
    # return (portfolio_valuation[-1] - portfolio_valuation[-2]) / portfolio_valuation[-2]

def custom_reward_function(history, window=252 * 24, risk_free_rate=0.03, non_trade_penalty=0.0002,
                           consecutive_non_trade_limit=24):
    # Get the portfolio valuations from the history
    portfolio_valuation = history["portfolio_valuation"]

    # Handle the first step (where we can't compute returns)
    if len(portfolio_valuation) < 2:
        return 0

    # Calculate hourly returns from the portfolio valuation
    returns = np.diff(portfolio_valuation) / portfolio_valuation[:-1]
    returns = returns[-window:]

    # Calculate Sortino ratio for hourly data
    hourly_risk_free_rate = risk_free_rate / (252 * 24)
    expected_return = np.mean(returns)
    downside_returns = returns[returns < hourly_risk_free_rate]

    if len(downside_returns) < 2:
        downside_std = 0.0001
    else:
        downside_std = np.std(downside_returns, ddof=1)
        downside_std = max(downside_std, 0.0001)

    sortino_ratio = (expected_return - hourly_risk_free_rate) / downside_std

    # Additional logic to encourage trading but allow non-trading in high-risk conditions
    recent_positions = history["position_index"][-consecutive_non_trade_limit:]
    if len(np.unique(recent_positions)) == 1 and np.unique(recent_positions)[
        0] == 0:  # If all recent positions are cash
        sortino_ratio -= non_trade_penalty

    return sortino_ratio

def split_data(df: pd.DataFrame, train_size=0.70, valid_size=0.15, test_size=0.15):
    """
    Split a dataframe into training, validation, and test sets.

    Parameters:
    - df: The input dataframe.
    - train_size: Proportion of the data for the training set.
    - valid_size: Proportion of the data for the validation set.
    - test_size: Proportion of the data for the test set.

    Returns:
    - train_df: Training dataframe.
    - valid_df: Validation dataframe.
    - test_df: Test dataframe.
    """

    # Check if proportions sum to 1
    assert train_size + valid_size + test_size == 1.0, "Proportions must sum to 1."

    # Calculate the index at which to split the data
    train_split = int(len(df) * train_size)
    valid_split = train_split + int(len(df) * valid_size)

    # Split the data
    train_df = df.iloc[:train_split]
    valid_df = df.iloc[train_split:valid_split]
    test_df = df.iloc[valid_split:]

    return train_df, valid_df, test_df

def split_data_by_dates(df: pd.DataFrame, train_end_date, valid_end_date):
    """
    Split a dataframe into training, validation, and test sets based on specified dates.

    Parameters:
    - df: The input dataframe with a datetime index.
    - train_end_date: The end date for the training set.
    - valid_end_date: The end date for the validation set. Also, the start date for the test set is assumed to be the next day.

    Returns:
    - train_df: Training dataframe.
    - valid_df: Validation dataframe.
    - test_df: Test dataframe.
    """

    # Ensure the DataFrame is sorted by date
    df = df.sort_index()

    # Split the data
    train_df = df.loc[:train_end_date]
    valid_df = df.loc[(train_end_date + pd.Timedelta(days=1)):valid_end_date]
    test_df = df.loc[(valid_end_date + pd.Timedelta(days=1)):]

    return train_df, valid_df, test_df

def rolling_zscore(df, window=20):
    mean = df.rolling(window=window).mean()
    std = df.rolling(window=window).std()
    zscore = (df - mean) / std
    return zscore

def get_tvt(data_path):
    # 1. Preparation
    df = pd.read_pickle(data_path)
    # add features
    df = create_features(df)

    # normalize data
    feature_columns = [col for col in df.columns if col.startswith("feature_")]
    for col in feature_columns:
        df[col] = rolling_zscore(df[col])

    df.dropna(inplace=True)

    # Split data (e.g., 90% training, 10% testing)
    train_df, eval_df, test_df = split_data(df, train_size=0.9, valid_size=0.05, test_size=0.05)

    return train_df, eval_df, test_df


class CustomScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def save(self, filename):
        joblib.dump(self.scaler, filename)

    def load(self, filename):
        self.scaler = joblib.load(filename)