import os
import pickle
from datetime import datetime
import random


import gymnasium as gym
import numpy as np
import optuna
import pandas as pd
import torch
from gymnasium import register
# from gym import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from custom_env import TradingEnv

print(TradingEnv)
register(
    id='TradingEnv',  # Unique ID for your environment
    entry_point='trading_env_gym_custom.custom_env:TradingEnv',  # Change 'path_to_your_module' to the module path of your environment
)


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

def predict_next_sb_log_dir(base_path, prefix="PPO_"):
    # Ensure the base directory exists.
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    existing_dirs = [d for d in os.listdir(base_path) if d.startswith(prefix)]
    existing_indices = sorted([int(d.replace(prefix, "")) for d in existing_dirs])
    next_index = existing_indices[-1] + 1 if existing_indices else 1
    return os.path.join(base_path, prefix + str(next_index))


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

def rolling_zscore(df, window=20):
    mean = df.rolling(window=window).mean()
    std = df.rolling(window=window).std()
    zscore = (df - mean) / std
    return zscore


def eval(test_df, model, unique_save_dir, test_log_path=None):
    eval_env = Monitor(gym.make("TradingEnv",
                        name="eval_BTCUSD",
                        df=test_df,
                        positions=[-1, 0, 1],
                        trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
                        borrow_interest_rate=0.0003 / 100, # 0.0003% per timestep (one timestep = 1h here),
                        reward_function=simple_reward,
                                tensorboard_log_path=test_log_path
                                ))

    obs, _ = eval_env.reset()
    truncated = False

    while not truncated:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = eval_env.step(action)

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
    train_df, eval_df, test_df = split_data(df, train_size=0.7, valid_size=0.15, test_size=0.15)

    base_log_path = './tensorboard_logs'
    predicted_path = predict_next_sb_log_dir(base_log_path)
    eval_log_path = os.path.join(predicted_path, 'eval')
    test_log_path = os.path.join(predicted_path, 'test')

    eval_env = Monitor(gym.make("TradingEnv",
                        name="eval_train",
                        df=eval_df,  # Your validation dataframe
                        positions=[-1, 0, 1],
                        trading_fees=0.01 / 100,
                        borrow_interest_rate=0.0003 / 100,
                        reward_function=simple_reward,tensorboard_log_path=eval_log_path,

                        ))
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path='./logs/best_model',
                                 log_path='./logs/results',
                                 eval_freq=len(train_df),
                                 deterministic=True, render=False)
    # 2. Training
    train_log_path = './tensorboard_logs/PPO_train'
    env = Monitor(gym.make("TradingEnv",
                   name="BTCUSD",
                   df=train_df,
                   positions=[-1, 0, 1],
                   trading_fees=0,
                   borrow_interest_rate=0,
                    reward_function=simple_reward,
                           tensorboard_log_path=predicted_path
                   ))

    vec_env = DummyVecEnv([lambda: env])
    policy_kwargs = dict(net_arch=[64, 64])
    model = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=base_log_path)
    model.learn(total_timesteps=300_000, callback=eval_callback)

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
    eval(test_df, model, unique_save_dir, test_log_path)

    print(f"Training completed. Artifacts saved to {unique_save_dir}")


def evaluate_model(model, env, num_episodes=1):
    """
    Evaluate the model's performance.

    Parameters:
    model (stable_baselines3 model): Trained model.
    env (gym.Env): Gym environment to test the model on.
    num_episodes (int): Number of episodes to test the model on.

    Returns:
    float: Mean rewards over the test episodes
    """
    total_returns = []

    for i in range(num_episodes):
        episode_reward = 0
        truncated = False
        obs, _ = env.reset()
        total_log_return = 0
        while not truncated:
            action, _states = model.predict(obs, deterministic=True)  # Use deterministic predictions for evaluation
            obs, reward, done, truncated, historical_info = env.step(action)
            total_log_return += reward

        total_return = np.exp(total_log_return) - 1  # Subtracting 1 to get the net return (profit/loss)
        total_returns.append(total_return)

    mean_total_return = sum(total_returns) / num_episodes
    return mean_total_return


def create_objective(env):
    def objective(trial):
        # Hyperparameters to tune
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        # Add other hyperparameters you want to tune here

        # Set up the RL model with suggested hyperparameters
        model = PPO('MlpPolicy', env, learning_rate=learning_rate, batch_size=batch_size, verbose=0)

        # Train your model
        model.learn(total_timesteps=16000)

        # Evaluate the trained model (you'll need to define evaluate_model yourself)
        rewards = evaluate_model(model, env)

        return rewards  # Optuna seeks to maximize this value

    return objective


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


def optimize_hyperparameters(data_path: str, n_trials=10):
    data, _, _ = get_tvt(data_path)
    env = Monitor(gym.make("TradingEnv",
                           name="BTCUSD",
                           df=data,
                           positions=[-1, 0, 1],
                           trading_fees=0,
                           borrow_interest_rate=0,
                           reward_function=simple_reward
                           ))

    objective = create_objective(env)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_hyperparams = study.best_params
    print(f"Optimal hyperparameters: {best_hyperparams}")


def train_and_evaluate(learning_rate, batch_size, data_path):
    data, _, _ = get_tvt(data_path)
    env = Monitor(gym.make("TradingEnv",
                           name="BTCUSD",
                           df=data,
                           positions=[-1, 0, 1],
                           trading_fees=0,
                           borrow_interest_rate=0,
                           reward_function=simple_reward
                           ))

    vec_env = DummyVecEnv([lambda: env])

    # Use the provided hyperparameters
    model = PPO('MlpPolicy', vec_env, learning_rate=learning_rate, batch_size=batch_size, verbose=0)

    # Get the total timesteps to train based on the environment
    total_timesteps = len(data)
    scaled_timesteps = total_timesteps * 1


    # Train the model
    model.learn(total_timesteps=scaled_timesteps)

    # Evaluate the trained model
    rewards = evaluate_model(model, env)  # Ensure this function is correctly aggregating rewards
    print(f"Mean rewards: {rewards}")
    return rewards

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if you are using CUDA
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)
# train_and_evaluate(0.0006, 64, 'data/binance-BTCUSDT-1h.pkl')
# Example usage
train_and_save_pipeline('data/binance-BTCUSDT-1h.pkl', '../training_results')
# optimize_hyperparameters('data/binance-BTCUSDT-1h.pkl', n_trials=10)
