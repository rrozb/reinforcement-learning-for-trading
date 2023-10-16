import os
import random
from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import register
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from custom_env import TradingEnv
from trading_env_gym_custom.custom_policies import CustomPolicy, CustomLSTMPolicy
from trading_env_gym_custom.evaluation import load_model, evaluate_model
from trading_env_gym_custom.features_eng import create_features, split_data, simple_reward, scale_features

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

def get_data(data_path, train_size=0.8, valid_size=0.15, test_size=0.05):
    df = pd.read_pickle(data_path)
    # add features
    df = create_features(df)

    # normalize data

    df.dropna(inplace=True)

    # Split data (e.g., 90% training, 10% testing)
    return  split_data(df, train_size=train_size, valid_size=valid_size, test_size=test_size)


def create_envs(train_df, eval_df, test_df, eval_log_path, test_log_path, train_log_path, window_size=120):
    eval_env = Monitor(gym.make("TradingEnv",
                                name="eval_train",
                                df=eval_df,  # Your validation dataframe
                                positions=[-1, 0, 1],
                                trading_fees=0.01 / 100,
                                borrow_interest_rate=0.0003 / 100,
                                windows=window_size,
                                reward_function=simple_reward, tensorboard_log_path=eval_log_path,

                                ))
    train_env = Monitor(gym.make("TradingEnv",
                           name="BTCUSD",
                           df=train_df,
                           positions=[-1, 0, 1],
                           trading_fees=0,
                           borrow_interest_rate=0,
                                 windows=window_size,
                           reward_function=simple_reward,
                           tensorboard_log_path=train_log_path
                           ))

    test_env = Monitor(gym.make("TradingEnv",
                                name="eval_BTCUSD",
                                df=test_df,
                                positions=[-1, 0, 1],
                                trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
                                borrow_interest_rate=0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here),
                                reward_function=simple_reward,
                                windows=window_size,
                                tensorboard_log_path=test_log_path
                                ))

    return train_env, eval_env, test_env
def train_and_save_pipeline(data_path, save_dir):
    # Create a unique directory based on the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_save_dir = os.path.join(save_dir, f'run_{timestamp}')

    if not os.path.exists(unique_save_dir):
        os.makedirs(unique_save_dir)

    # # Split data (e.g., 90% training, 10% testing)
    train_df, eval_df, test_df = get_data(data_path)

    train_df, train_scaler = scale_features(train_df)
    eval_df, _ = scale_features(eval_df, train_scaler)
    test_df, _ = scale_features(test_df, train_scaler)

    # base_log_path = './tensorboard_logs'
    predicted_path, next_index = None, 0  # predict_next_sb_log_dir(base_log_path)
    eval_log_path = None  # os.path.join(predicted_path, 'eval')
    test_log_path = None  # os.path.join(predicted_path, 'test')
    #
    env, eval_env, _ = create_envs(train_df, eval_df, test_df, eval_log_path, test_log_path, predicted_path)


    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=f'./logs/{next_index}',
                                 log_path=f'./logs/results/{next_index}',
                                 eval_freq=len(train_df),
                                 deterministic=True, render=False)
    # # 2. Training
    train_log_path = './tensorboard_logs/'
    #
    model, vec_env = train_model(env, train_log_path, eval_callback, total_timesteps=100_000)
    #
    # # 3. Saving Artifacts
    # # Save model
    # model_path = os.path.join(unique_save_dir, "ppo_bitcoin_trader.zip")
    # model.save(model_path)
    #
    # vec_env.envs[0].get_wrapper_attr('save_for_render')(unique_save_dir)
    #
    # # # Save historical info
    # # historical_info = vec_env.envs[0].get_wrapper_attr('historical_info')
    # # historical_info_path = os.path.join(unique_save_dir, "historical_info.pkl")
    # # with open(historical_info_path, 'wb') as f:
    # #     pickle.dump(historical_info, f)
    # #
    # # # do evaluation
    # # evaluate_test(test_df, model, unique_save_dir, test_log_path)
    #
    # print(f"Training completed. Artifacts saved to {unique_save_dir}")


def train_model(env, base_log_path, eval_callback, total_timesteps=300_000):
    # Create the environment

    vec_env = DummyVecEnv([lambda: env])
    CLIP_EPSILON = 0.2  # Adjust this as needed
    ENTROPY_COEFF = 0.01  # Adjust this as needed
    # Initialize the model with the custom policy
    model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=1, tensorboard_log=base_log_path,
                         clip_range=CLIP_EPSILON, ent_coef=ENTROPY_COEFF)

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    return model, vec_env #FIXME: do i need it?

def evaluate_with_retrain(model_path, data_path, unique_save_dir=None):
    model = load_model(model_path=model_path)
    train_df, eval_df, test_df = get_data(data_path)
    _, env_eval, test_env = create_envs(train_df, eval_df, test_df, None, None, None)
    evaluate_model(model, env_eval, unique_save_dir)
    evaluate_model(model, test_env, unique_save_dir)

    E = 1
    len_df_eval = len(eval_df)
    vec_env = DummyVecEnv([lambda: env_eval])
    model.env = vec_env
    model.learn(total_timesteps=len_df_eval*E)
    print("Evaluating EVAL env after retrain")
    evaluate_model(model, env_eval, unique_save_dir)
    print("Evaluating TEST env after retrain")
    evaluate_model(model, test_env, unique_save_dir)





set_seeds(42)



# Example usage
train_and_save_pipeline('data/binance-BTCUSDT-1h.pkl', '../training_results')
# evaluate_with_retrain('/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/logs/26/best_model.zip', 'data/binance-BTCUSDT-1h.pkl', 'training_results/testtest')
