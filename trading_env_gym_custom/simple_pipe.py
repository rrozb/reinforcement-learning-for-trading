import os
import random
from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import register
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from custom_env import TradingEnv
from trading_env_gym_custom.custom_policies import CustomPolicy
from trading_env_gym_custom.evaluation import load_model, evaluate_model
from trading_env_gym_custom.features_eng import create_features, split_data, simple_reward, rolling_zscore, CustomScaler
from trading_env_gym_custom.logging_process import predict_next_sb_log_dir

print(TradingEnv)
register(
    id='TradingEnv',  # Unique ID for your environment
    entry_point='trading_env_gym_custom.custom_env:TradingEnv',  # Change 'path_to_your_module' to the module path of your environment
)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if you are using CUDA
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_data(data_path, train_size=0.8, valid_size=0.15, test_size=0.05):
    df = pd.read_pickle(data_path)

    # add features
    df = create_features(df)

    # Split data first to avoid data leakage during normalization
    train_df, eval_df, test_df = split_data(df, train_size=train_size, valid_size=valid_size, test_size=test_size)

    # normalize data with CustomScaler
    feature_columns = [col for col in df.columns if col.startswith("feature_")]
    scaler = CustomScaler()
    train_df.loc[:, feature_columns] = scaler.fit_transform(train_df[feature_columns])
    eval_df.loc[:, feature_columns] = scaler.transform(eval_df[feature_columns])
    test_df.loc[:, feature_columns] = scaler.transform(test_df[feature_columns])

    # Extract base name and directory from data_path
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    directory = os.path.dirname(data_path)
    save_folder = os.path.join(directory, base_name)

    # Create folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save data splits
    train_df.to_pickle(os.path.join(save_folder, "train.pkl"))
    eval_df.to_pickle(os.path.join(save_folder, "eval.pkl"))
    test_df.to_pickle(os.path.join(save_folder, "test.pkl"))

    # Save the scaler for future use
    scaler.save(os.path.join(save_folder, "scaler.pkl"))

    return train_df, eval_df, test_df


def create_envs(train_df, eval_df, test_df):
    eval_env = Monitor(gym.make("TradingEnv",
                                name="eval_train",
                                df=eval_df,  # Your validation dataframe
                                positions=[-1, 0, 1],
                                trading_fees=0.01 / 100,
                                borrow_interest_rate=0.0003 / 100,
                                reward_function=simple_reward,

                                ))
    train_env = Monitor(gym.make("TradingEnv",
                           name="BTCUSD",
                           df=train_df,
                           positions=[-1, 0, 1],
                           trading_fees=0,
                           borrow_interest_rate=0,
                           reward_function=simple_reward
                           ))

    test_env = Monitor(gym.make("TradingEnv",
                                name="eval_BTCUSD",
                                df=test_df,
                                positions=[-1, 0, 1],
                                trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
                                borrow_interest_rate=0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here),
                                reward_function=simple_reward
                                ))

    return train_env, eval_env, test_env


def train_and_save_pipeline(data_dir, save_dir):
    # Create a unique directory based on the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_save_dir = os.path.join(save_dir, f'run_{timestamp}')
    model_class = PPO
    #PPO Spefific hyperparams
    hyperparams = {
        'learning_rate': 2.5e-5,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        # 'use_sde': True,
        # 'sde_sample_freq': 5
    }
    if not os.path.exists(unique_save_dir):
        os.makedirs(unique_save_dir)

    # Load data
    train_df = pd.read_pickle(os.path.join(data_dir, "train.pkl"))
    eval_df = pd.read_pickle(os.path.join(data_dir, "eval.pkl"))
    test_df = pd.read_pickle(os.path.join(data_dir, "test.pkl"))

    # Create environments
    train_env, eval_env, _ = create_envs(train_df, eval_df, test_df)

    base_log_path = f'./tensorboard_logs/{model_class.__name__}/{timestamp}/'

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=os.path.join(unique_save_dir, "best_model"),
                                 log_path=os.path.join(unique_save_dir, "results"),
                                 eval_freq=20_000,
                                 deterministic=True, render=False)
    # 2. Training
    model = model_class('MlpPolicy', train_env, verbose=1, **hyperparams, tensorboard_log=base_log_path)
    model.learn(total_timesteps=1_000_000, callback=eval_callback)


# def train_model(env, base_log_path, eval_callback, total_timesteps=300_000):
#     # Create the environment
#
#     vec_env = DummyVecEnv([lambda: env])
#     CLIP_EPSILON = 0.2  # Adjust this as needed
#     ENTROPY_COEFF = 0.01  # Adjust this as needed
#     # Initialize the model with the custom policy
#     model = PPO(CustomPolicy, vec_env, verbose=1, tensorboard_log=base_log_path, clip_range=CLIP_EPSILON,
#                 ent_coef=ENTROPY_COEFF)
#
#     # Train the model
#     model.learn(total_timesteps=total_timesteps, callback=eval_callback)
#
#     return model, vec_env #FIXME: do i need it?

# def evaluate_with_retrain(model_path, data_path, unique_save_dir=None):
#     model = load_model(model_path=model_path)
#     train_df, eval_df, test_df = get_data(data_path)
#     _, env_eval, test_env = create_envs(train_df, eval_df, test_df, None, None, None)
#     evaluate_model(model, env_eval, unique_save_dir)
#     evaluate_model(model, test_env, unique_save_dir)
#
#     E = 2
#     len_df_eval = len(eval_df)
#     vec_env = DummyVecEnv([lambda: env_eval])
#     model.env = vec_env
#     model.learn(total_timesteps=len_df_eval*E)
#     print("Evaluating EVAL env after retrain")
#     evaluate_model(model, env_eval, unique_save_dir)
#     print("Evaluating TEST env after retrain")
#     evaluate_model(model, test_env, unique_save_dir)





set_seeds(42)

# Example usage
# train_and_save_pipeline('data/binance-BTCUSDT-1h.pkl', '../training_results')
# evaluate_with_retrain('/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/logs/26/best_model.zip', 'data/binance-BTCUSDT-1h.pkl', 'training_results/testtest')

process_data("/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/bitfinex2-BTCUSD-1h_2015-2023.pkl",
             train_size=0.97, valid_size=0.02, test_size=0.01)

train_and_save_pipeline('/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/bitfinex2-BTCUSD-1h_2015-2023',
'/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/training_results')
