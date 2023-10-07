import os
import pickle
from datetime import datetime

import gym_trading_env
import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

print(gym_trading_env)
def create_features(df: pd.DataFrame):
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]

    # Moving Averages
    df['feature_ma5'] = df['close'].rolling(window=5).mean()
    df['feature_ma20'] = df['close'].rolling(window=20).mean()

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(window=14).mean()
    RolDown = dDown.abs().rolling(window=14).mean()

    RS = RolUp / RolDown
    df['feature_RSI'] = 100.0 - (100.0 / (1.0 + RS))

    # Moving Average Convergence Divergence (MACD)
    df['feature_EMA12'] = df['close'].ewm(span=12).mean()
    df['feature_EMA26'] = df['close'].ewm(span=26).mean()
    df['feature_MACD'] = df['feature_EMA12'] - df['feature_EMA26']

    # Bollinger Bands
    df['feature_BB_Middle'] = df['close'].rolling(window=20).mean()
    df['feature_BB_Upper'] = df['feature_BB_Middle'] + (df['close'].rolling(window=20).std() * 2)
    df['feature_BB_Lower'] = df['feature_BB_Middle'] - (df['close'].rolling(window=20).std() * 2)

    df.dropna(inplace=True)
    return df

def rolling_zscore(df, window=20):
    mean = df.rolling(window=window).mean()
    std = df.rolling(window=window).std()
    zscore = (df - mean) / std
    return zscore


def eval(test_df, model, unique_save_dir):
    eval_env = gym.make("TradingEnv",
                        name="eval_BTCUSD",
                        df=test_df,
                        positions=[-1, 0, 1],
                        trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
                        borrow_interest_rate=0.0003 / 100, # 0.0003% per timestep (one timestep = 1h here)
                        )

    obs, _ = eval_env.reset()
    truncated = False

    while not truncated:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, done = eval_env.step(action)

    eval_env.get_wrapper_attr('save_for_render')(unique_save_dir)

    print(f"Training and evaluation completed. Artifacts saved to {unique_save_dir}")


def train_and_save_pipeline(data_path, save_dir):
    # Create a unique directory based on the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_save_dir = os.path.join(save_dir, f'run_{timestamp}')

    if not os.path.exists(unique_save_dir):
        os.makedirs(unique_save_dir)

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
    train_size = int(0.90 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]

    # 2. Training
    env = gym.make("TradingEnv",
                   name="BTCUSD",
                   df=train_df,
                   positions=[-1, 0, 1],
                   trading_fees=0,
                   borrow_interest_rate=0,
                   )

    vec_env = DummyVecEnv([lambda: env])
    policy_kwargs = dict(net_arch=[64, 64])
    model = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=2_000_000)

    # 3. Saving Artifacts
    # Save model
    model_path = os.path.join(unique_save_dir, "ppo_bitcoin_trader.zip")
    model.save(model_path)

    vec_env.envs[0].get_wrapper_attr('save_for_render')(unique_save_dir)

    # # Save historical info
    historical_info = vec_env.envs[0].get_wrapper_attr('historical_info')
    historical_info_path = os.path.join(unique_save_dir, "historical_info.pkl")
    with open(historical_info_path, 'wb') as f:
        pickle.dump(historical_info, f)

    # do evaluation
    eval(test_df, model, unique_save_dir)

    print(f"Training completed. Artifacts saved to {unique_save_dir}")


# Example usage
train_and_save_pipeline('data/binance-BTCUSDT-1h.pkl', '../training_results')
