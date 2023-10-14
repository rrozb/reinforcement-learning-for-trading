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
from trading_env_gym_custom.custom_policies import CustomPolicy
from trading_env_gym_custom.evaluation import evaluate_test, evaluate_model
from trading_env_gym_custom.features_eng import create_features, split_data, simple_reward, get_tvt, rolling_zscore
from trading_env_gym_custom.logging_process import predict_next_sb_log_dir


print(TradingEnv)
register(
    id='TradingEnv',  # Unique ID for your environment
    entry_point='trading_env_gym_custom.custom_env:TradingEnv',  # Change 'path_to_your_module' to the module path of your environment
)




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
    train_df, eval_df, test_df = split_data(df, train_size=0.9, valid_size=0.05, test_size=0.05)

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
    model, vec_env = train_model(env, train_log_path, eval_callback, total_timesteps=300_000)
    # vec_env = DummyVecEnv([lambda: env])
    # policy_kwargs = dict(net_arch=[64, 64])
    # model = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=base_log_path)
    # model.learn(total_timesteps=300_000, callback=eval_callback)

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
    evaluate_test(test_df, model, unique_save_dir, test_log_path)

    print(f"Training completed. Artifacts saved to {unique_save_dir}")


def train_model(env, base_log_path, eval_callback, total_timesteps=300_000):
    # Create the environment

    vec_env = DummyVecEnv([lambda: env])

    # Initialize the model with the custom policy
    model = PPO(CustomPolicy, vec_env, verbose=1, tensorboard_log=base_log_path)

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    return model, vec_env #FIXME: do i need it?

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
