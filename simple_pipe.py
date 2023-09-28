import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import gym_trading_env
import pickle
from datetime import datetime


def train_and_save_pipeline(data_path, save_dir):
    # Create a unique directory based on the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_save_dir = os.path.join(save_dir, f'run_{timestamp}')

    if not os.path.exists(unique_save_dir):
        os.makedirs(unique_save_dir)

    # 1. Preparation
    df = pd.read_pickle(data_path)
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df.dropna(inplace=True)

    # Split data (e.g., 80% training, 20% testing)
    train_size = int(0.8 * len(df))
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
    model.learn(total_timesteps=1_000_000)

    # 3. Saving Artifacts
    # Save model
    model_path = os.path.join(unique_save_dir, "ppo_bitcoin_trader.zip")
    model.save(model_path)

    # Save historical info
    historical_info = env.get_wrapper_attr('historical_info')
    historical_info_path = os.path.join(unique_save_dir, "historical_info.pkl")
    with open(historical_info_path, 'wb') as f:
        pickle.dump(historical_info, f)

    # Save training data for reproducibility
    train_data_path = os.path.join(unique_save_dir, "train_data.pkl")
    train_df.to_pickle(train_data_path)

    print(f"Training completed. Artifacts saved to {unique_save_dir}")


# Example usage
train_and_save_pipeline('data/binance-BTCUSDT-1h.pkl', 'training_results')
