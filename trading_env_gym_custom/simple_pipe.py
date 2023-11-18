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

from custom_env import TradingEnv
from trading_env_gym_custom.evaluation import evaluate_model
from trading_env_gym_custom.features_eng import create_features, split_data, simple_reward, CustomScaler, \
    split_data_by_dates

print(TradingEnv)
register(
    id='TradingEnv',  # Unique ID for your environment
    entry_point='trading_env_gym_custom.custom_env:TradingEnv',
    # Change 'path_to_your_module' to the module path of your environment
)
register(
    id='MultiAssetTradingEnv',  # Unique ID for your environment
    entry_point='trading_env_gym_custom.custom_env:MultiAssetPortfolio',
    # Change 'path_to_your_module' to the module path of your environment
)


def get_warmup_data(test_df, current_period, warmup_length=1):
    """
    Fetches warmup data for LSTM.

    Parameters:
    test_df (DataFrame): The dataframe containing test data.
    current_period (Period): The current period for which the model is being evaluated.
    warmup_length (int): Number of periods to include in the warmup.

    Returns:
    DataFrame: Warmup data for the LSTM model.
    """
    warmup_periods = [current_period - i for i in range(1, warmup_length + 1)]
    warmup_data = test_df[test_df['period'].isin(warmup_periods)]
    return warmup_data

def linear_schedule(initial_value, final_value):
    def schedule(progress_remaining):
        return initial_value + (final_value - initial_value) * (1 - progress_remaining)

    return schedule


def exponential_schedule(initial_value, decay_rate):
    def schedule(progress_remaining):
        return initial_value * (decay_rate ** (1 - progress_remaining))

    return schedule


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if you are using CUDA
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_data(data_path, sizes=None, date_indexes=None, granularity='1h'):
    if sizes is not None and date_indexes is not None:
        raise ValueError("Only one of sizes and date_indexes can be specified")

    df = pd.read_pickle(data_path)

    # add features
    df = create_features(df, granularity=granularity)

    if sizes is not None:
        # Split data (e.g., 90% training, 10% testing)
        train_df, eval_df, test_df = split_data(df, train_size=sizes[0], valid_size=sizes[1], test_size=sizes[2])

    elif date_indexes is not None:
        # Split data (e.g., 90% training, 10% testing)
        train_df, eval_df, test_df = split_data_by_dates(df, train_end_date=date_indexes[0],
                                                         valid_end_date=date_indexes[1])

    else:
        raise ValueError("Either sizes or date_indexes must be specified")

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
                                windows=24 * 7,

                                ))
    train_env = Monitor(gym.make("TradingEnv",
                                 name="BTCUSD",
                                 df=train_df,
                                 positions=[-1, 0, 1],
                                 trading_fees=0,
                                 borrow_interest_rate=0,
                                 reward_function=simple_reward,
                                 windows=24 * 7,

                                 ))

    test_env = Monitor(gym.make("TradingEnv",
                                name="eval_BTCUSD",
                                df=test_df,
                                positions=[-1, 0, 1],
                                trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
                                borrow_interest_rate=0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here),
                                reward_function=simple_reward,
                                windows=24 * 7,
                                ))

    return train_env, eval_env, test_env

def create_multi_asset_envs(train_df, eval_df, test_df):
    eval_env = Monitor(gym.make("MultiAssetTradingEnv",
                                name="eval_train",
                                df=eval_df, 
                                trading_fees=0.01 / 100,
                                borrow_interest_rate=0,
                                reward_function=simple_reward,

                                ))
    train_env = Monitor(gym.make("MultiAssetTradingEnv",
                                 name="BTCUSD",
                                 df=train_df,

                                 trading_fees=0.01 / 100,
                                 borrow_interest_rate=0,
                                 reward_function=simple_reward
                                 ))

    test_env = Monitor(gym.make("MultiAssetTradingEnv",
                                name="eval_BTCUSD",
                                df=test_df,
                                trading_fees=0.01 / 100,
                                borrow_interest_rate=0,
                                reward_function=simple_reward
                                ))

    return train_env, eval_env, test_env

def load_best_model(model_class, save_dir):
    # Load the best model based on validation before final training
    best_model_dir = os.path.join(save_dir, "best_model")
    model = model_class.load(best_model_dir + "/best_model.zip")
    return model

def validate_and_roll_train(model_class, data_dir, save_dir, period='W'):
    # Load data
    train_df = pd.read_pickle(os.path.join(data_dir, "train.pkl"))
    eval_df = pd.read_pickle(os.path.join(data_dir, "eval.pkl"))
    test_df = pd.read_pickle(os.path.join(data_dir, "test.pkl"))

    # Split test data into periods (e.g., months)
    test_df['period'] = test_df.index.to_period(period)
    periods = test_df['period'].unique()

    # Load the best model
    best_model = load_best_model(model_class, save_dir)
    _, eval_env, test_env = create_envs(train_df, eval_df, test_df)

    best_model.set_env(eval_env)
    eval_model = evaluate_model(best_model, eval_env, save_dir)
    print(f"Eval model: {eval_model}")
    eval_env.reset()
    best_model.learn(total_timesteps=len(eval_df) * 4)

    best_model.set_env(test_env)
    test_model = evaluate_model(best_model, test_env, save_dir)
    print(f"TEST not training model: {test_model}")

    portfolio_value = 1000  # Starting portfolio value
    total_returns = []

    for i in range(1, len(periods)):
        p = periods[i]
        prev_p = periods[i - 1]
        print(f"Training on period {p}. Portfolio value so far: {portfolio_value}")

        # Create a subset of test data for the previous and current period
        last_week_prev_period = test_df[test_df['period'] == prev_p].last('W')
        period_df = test_df[test_df['period'] == p]
        combined_df = pd.concat([last_week_prev_period, period_df])

        # Create environment for the combined data
        _, _, test_env_period = create_envs(train_df, eval_df, combined_df)

        # Set the environment to the current period
        best_model.set_env(test_env_period)

        # Evaluate model on the current period without training
        period_return = evaluate_model(best_model, test_env_period, save_dir)
        total_returns.append(period_return)

        # Update portfolio value
        portfolio_value *= (1 + period_return)

        # Reset the environment state if necessary before training
        test_env_period.reset()

        # Train model on the current period
        best_model.learn(total_timesteps=len(combined_df) * 4)

    print(f"Final Portfolio Value: {portfolio_value}")
    print(f"Total Compounded Return: {portfolio_value / 1000 - 1}")



def train_and_save_pipeline(data_dir, save_dir):
    # Create a unique directory based on the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_save_dir = os.path.join(save_dir, f'run_{timestamp}')
    print(f"Saving results to {unique_save_dir}")
    model_class = PPO
    if not os.path.exists(unique_save_dir):
        os.makedirs(unique_save_dir)

    # Load data
    train_df = pd.read_pickle(os.path.join(data_dir, "train.pkl"))
    train_df = train_df.tail(500_000)
    eval_df = pd.read_pickle(os.path.join(data_dir, "eval.pkl"))
    test_df = pd.read_pickle(os.path.join(data_dir, "test.pkl"))

    # Create environments
    train_env, eval_env, test_env = create_envs(train_df, eval_df, test_df)

    base_log_path = f'./tensorboard_logs/{model_class.__name__}/{timestamp}/'

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=os.path.join(unique_save_dir, "best_model"),
                                 log_path=os.path.join(unique_save_dir, "results"),
                                 eval_freq=50_000,
                                 n_eval_episodes=2,
                                 deterministic=True, render=False)
    total_timesteps = 300_000
    learning_rate_schedule = linear_schedule(3e-4, 2.5e-5)
    clip_range_schedule = linear_schedule(0.3, 0.1)
    # ent_coef_schedule = linear_schedule(0.1, 0.01)

    hyperparams = {
        'learning_rate': 2.5e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.3,
        'ent_coef': 0.1,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }
    # 2. Training
    # model = model_class('MlpPolicy', train_env, verbose=1, **hyperparams, tensorboard_log=base_log_path)
    model = RecurrentPPO("MlpLstmPolicy", train_env, **hyperparams, verbose=1, tensorboard_log=base_log_path)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # validate_and_roll_train(model_class, data_dir, unique_save_dir)

    # best_model = load_best_model(model_class, unique_save_dir)
    # best_model.set_env(eval_env)
    # best_model.learn(total_timesteps=len(eval_df))
    # # first evaluate on eval test env no training
    # evaluate_model(best_model, eval_env, unique_save_dir)
    # # then evaluate on test env no training
    # evaluate_model(best_model, test_env, unique_save_dir)
    # # finally do a final training on the test env
    # best_model.learn(total_timesteps=len(test_df))



# def test_with_retraining(data_dir, unique_save_dir, model_class=PPO):
#     # Load data
#     train_df = pd.read_pickle(os.path.join(data_dir, "train.pkl"))
#     eval_df = pd.read_pickle(os.path.join(data_dir, "eval.pkl"))
#     test_df = pd.read_pickle(os.path.join(data_dir, "test.pkl"))
#
#     # split test_df
#
#     # Create environments
#     test_env = Monitor(gym.make("TradingEnv",
#                                 name="eval_BTCUSD",
#                                 df=test_df,
#                                 positions=[-1, 0, 1],
#                                 trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
#                                 borrow_interest_rate=0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here),
#                                 reward_function=simple_reward
#                                 ))

# def sequential_training(data_dirs, final_data_dir, save_dir, model_class=PPO):
#     # Initialize model
#     model = None
#
#     hyperparams = {
#         'learning_rate': 2.5e-5,
#         'n_steps': 2048,
#         'batch_size': 64,
#         'n_epochs': 10,
#         'gamma': 0.99,
#         'gae_lambda': 0.95,
#         'clip_range': 0.3,
#         'ent_coef': 0.1,
#         'vf_coef': 0.5,
#         'max_grad_norm': 0.5,
#     }
#     base_log_path = f'./tensorboard_logs/{model_class.__name__}'
#     # Train on each dataset sequentially
#     for data_dir in data_dirs:
#         print(f"Training on {data_dir.split('/')[-1]}")
#         # Update save_dir for each run
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         unique_save_dir = os.path.join(save_dir, f'run_{timestamp}')
#         os.makedirs(unique_save_dir, exist_ok=True)
#
#         train_df = pd.read_pickle(os.path.join(data_dir, "train.pkl"))
#         eval_df = pd.read_pickle(os.path.join(data_dir, "eval.pkl"))
#         test_df = pd.read_pickle(os.path.join(data_dir, "test.pkl"))
#
#         train_env, eval_env, test_env = create_envs(train_df, eval_df, test_df)
#
#         if model is None:
#             # First iteration: create a new model
#             model = model_class('MlpPolicy', train_env, tensorboard_log=base_log_path, verbose=1, **hyperparams)
#         else:
#             # Subsequent iterations: continue training the existing model
#             model.set_env(train_env)
#
#         model.learn(total_timesteps=len(train_df) * 2, tb_log_name=f"run_{model_class}_{timestamp}")
#
#     # After sequential training, fine-tune on the final dataset
#     final_train_df = pd.read_pickle(os.path.join(final_data_dir, "train.pkl"))
#     final_eval_df = pd.read_pickle(os.path.join(final_data_dir, "eval.pkl"))
#     final_test_df = pd.read_pickle(os.path.join(final_data_dir, "test.pkl"))
#
#     final_train_env, final_eval_env, _ = create_envs(final_train_df, final_eval_df, final_test_df)
#
#     model.set_env(final_train_env)
#
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     unique_save_dir = os.path.join(save_dir, f'run_pre_{timestamp}')
#     model_class = PPO
#     if not os.path.exists(unique_save_dir):
#         os.makedirs(unique_save_dir)
#
#     # Load the best model based on validation before final training
#
#     eval_callback = EvalCallback(final_eval_env,
#                                  best_model_save_path=os.path.join(unique_save_dir, "best_model"),
#                                  log_path=os.path.join(unique_save_dir, "results"),
#                                  eval_freq=5_000,
#                                  deterministic=True, render=False)
#
#     # Final training
#     model.learn(total_timesteps=500_0000, callback=eval_callback, tb_log_name=f"run_final_{model_class}_{timestamp}")
#
#     # load best model
#     model = model_class.load(os.path.join(unique_save_dir, "best_model"))
#     # train it on eval data
#     model.set_env(final_eval_env)
#     model.learn(total_timesteps=len(final_eval_df), tb_log_name=f"run_eval_{model_class}_{timestamp}")


set_seeds(42)
#
# data_dirs = [
#     "/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/bitfinex2-ETHUSDT-1h_2015-2023",
#     "/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/huobi-ETHUSDT-1h_2015-2023"
#     ,
#     "/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/bitfinex2-BTCUSDT-1h_2015-2023",
#     "/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/huobi-BTCUSDT-1h_2015-2023", ]
#
# final_data_dir = "/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/bitfinex2-BTCUSD-1h_2015-2023"
# save_dir = '/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/training_results'
# sequential_training(data_dirs, final_data_dir, save_dir)
# process_data("/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/bitfinex2-BTCUSD-5m.pkl",
#              date_indexes=[pd.to_datetime("2023-08-01"), pd.to_datetime("2023-09-01")], granularity='5m')

# train_and_save_pipeline('/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/bitfinex2-BTCUSD-1h_2015-2023',
# '/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/training_results')

validate_and_roll_train(RecurrentPPO,
                        '/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/bitfinex2-BTCUSD-1h_2015-2023',
                        '/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/training_results/run_20231118_101636')
