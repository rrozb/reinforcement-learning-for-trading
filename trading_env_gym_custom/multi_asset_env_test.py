import pandas as pd

from trading_env_gym_custom.custom_env import TradingMultiAssetEnv

btc = pd.read_pickle("data/bitfinex2-BTCUSDT-1h_2015-2023.pkl")
eth = pd.read_pickle("data/bitfinex2-ETHUSDT-1h_2015-2023.pkl")
# Ensure the date columns are datetime objects
btc['date_close'] = pd.to_datetime(btc['date_close'])
eth['date_close'] = pd.to_datetime(eth['date_close'])

# Set the index to be the date
btc.set_index('date_close', inplace=True)
eth.set_index('date_close', inplace=True)

# Add the pct_change feature
btc['feature_pct_change'] = btc['close'].pct_change().fillna(0)
eth['feature_pct_change'] = eth['close'].pct_change().fillna(0)

# Concatenate the dataframes with a MultiIndex
multi_df = pd.concat({'BTC': btc, 'ETH': eth}, names=['Asset', 'Date'])

# Make sure to sort the index to avoid performance warnings
multi_df.sort_index(inplace=True)

env = TradingMultiAssetEnv(
    df=multi_df,
)
env._idx = 0
print(env._get_price("BTC", ))
print(env.df["close"].head())