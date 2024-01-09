import os
import random
import time
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
from trading_env_gym_custom.eval import evaluate_policy
from trading_env_gym_custom.features_eng import create_features, split_data, simple_reward, CustomScaler, \
    split_data_by_dates, risk_adjusted

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


def segment_and_shuffle(df, segment_size='1M'):
    # Assuming df has a datetime index
    segments = [group for _, group in df.groupby(pd.Grouper(freq=segment_size))]
    np.random.shuffle(segments)
    return pd.concat(segments)


def add_noise(df, noise_level=0.01):
    noisy_df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'float64':  # Add noise only to numerical columns
            noise = np.random.normal(0, noise_level, size=df[col].shape)
            noisy_df[col] += noise
    return noisy_df


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


def create_envs(train_df, eval_df, test_df, reward_function=simple_reward):
    def create_env(df):
        return Monitor(gym.make("TradingEnv",
                                name="BTCUSD",
                                df=df,
                                positions=[-1, 0, 1],
                                trading_fees=0.01 / 100,
                                borrow_interest_rate=0,
                                reward_function=reward_function,
                                windows=24 * 7,
                                ))

    eval_env = create_env(eval_df)
    train_env = create_env(train_df)
    test_env = create_env(test_df)

    return train_env, eval_env, test_env



def load_best_model(model_class, save_dir):
    # Load the best model based on validation before final training
    best_model_dir = os.path.join(save_dir, "best_model")
    model = model_class.load(best_model_dir + "/best_model.zip")
    return model


def validate_and_roll_train(model_class, data_dir, save_dir, period='W'):
    repeat_dataset = 1
    # Load data
    train_df = pd.read_pickle(os.path.join(data_dir, "train.pkl"))
    eval_df = pd.read_pickle(os.path.join(data_dir, "eval.pkl"))
    test_df = pd.read_pickle(os.path.join(data_dir, "test.pkl"))

    # Split test data into periods (e.g., months)
    test_df['period'] = test_df.index.to_period(period)
    periods = test_df['period'].unique()

    # Load the best model
    best_model = load_best_model(model_class, save_dir)
    best_model.tensorboard_log = None
    _, eval_env, test_env = create_envs(train_df, eval_df, test_df, simple_reward)

    best_model.set_env(eval_env)
    mean_reward, std_reward, all_infos = evaluate_policy(best_model,
                                              eval_env,
                                              n_eval_episodes=1,
                                              deterministic=True,
                                              return_episode_rewards=True)

    print(f"Mean reward before fine tuning: {mean_reward} +/- {std_reward}.")
    eval_env.reset()
    best_model.learn(total_timesteps=len(eval_df) * repeat_dataset, )

    best_model.set_env(test_env)
    mean_reward, std_reward, all_infos = evaluate_policy(best_model,
                                              eval_env,
                                              n_eval_episodes=1,
                                              deterministic=True,
                                              return_episode_rewards=True)

    print(f"Mean reward after fine tuning: {mean_reward} +/- {std_reward}.")
    mean_reward, std_reward, all_infos = evaluate_policy(best_model,
                                              test_env,
                                              n_eval_episodes=1,
                                              deterministic=True,
                                              return_episode_rewards=True)

    print(f"Test Mean reward after fine tuning: {mean_reward} +/- {std_reward}.")
    portfolio_value = 1000  # Starting portfolio value
    infos_combined = []
    for i in range(1, len(periods)):
        p = periods[i]
        prev_p = periods[i - 1]
        print(f"Training on period {p}. Portfolio value so far: {portfolio_value}")

        # Create a subset of test data for the previous and current period
        last_week_prev_period = test_df[test_df['period'] == prev_p].last('W')
        period_df = test_df[test_df['period'] == p]
        combined_df = pd.concat([last_week_prev_period, period_df])

        # Create environment for the combined data
        _, _, test_env_period = create_envs(train_df, eval_df, combined_df, simple_reward)

        # Set the environment to the current period
        best_model.set_env(test_env_period)

        # Evaluate model on the current period without training
        # period_return = evaluate_model(best_model, test_env_period, save_dir)
        _, _, all_infos = evaluate_policy(best_model,
                                                  test_env_period,
                                                  n_eval_episodes=1,
                                                  deterministic=True,
                                                  return_episode_rewards=True)
        flattened_infos = [item for sublist in all_infos for item in sublist]

        infos_combined.extend(flattened_infos)
        mean_reward =  flattened_infos[-1]['portfolio_valuation'] - flattened_infos[0]['portfolio_valuation']

        # Update portfolio value
        portfolio_value += mean_reward

        # Reset the environment state if necessary before training
        test_env_period.reset()

        # Train model on the current period
        best_model.learn(total_timesteps=len(period_df)*5)

    print(f"Final Portfolio Value: {portfolio_value}")
    pd.DataFrame(infos_combined).to_csv(f"/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/evaluations/{model_class.__name__}{time.time()}_infos.csv")
    # print(f"Total Compounded Return: {portfolio_value / 1000 - 1}")


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
    eval_df = pd.read_pickle(os.path.join(data_dir, "eval.pkl"))
    test_df = pd.read_pickle(os.path.join(data_dir, "test.pkl"))


    # Create environments
    train_env, eval_env, test_env = create_envs(train_df, eval_df, test_df, risk_adjusted)

    base_log_path = f'./tensorboard_logs/{model_class.__name__}/{timestamp}/'

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=os.path.join(unique_save_dir, "best_model"),
                                 log_path=os.path.join(unique_save_dir, "results"),
                                 eval_freq=len(train_df),
                                 n_eval_episodes=1,
                                 deterministic=True, render=False)
    total_timesteps = len(train_df) * 10
    learning_rate_schedule = linear_schedule(3e-4, 2.5e-5)
    clip_range_schedule = linear_schedule(0.3, 0.1)
    # ent_coef_schedule = linear_schedule(0.1, 0.01)

    hyperparams = {
        'learning_rate': 7.5e-5,
        'n_steps': 2048,
        'batch_size': 128,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.3,
        'ent_coef': 0.1,
        'vf_coef': 0.5,
        'max_grad_norm': 0.8,
    }
    # 2. Training

    # model = model_class('MlpPolicy', train_env, verbose=1, **hyperparams, tensorboard_log=base_log_path)
    model = RecurrentPPO("MlpLstmPolicy", train_env, **hyperparams, verbose=1, tensorboard_log=base_log_path)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)


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
# process_data("/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/bitfinex2-BTCUSD-1h.pkl",
#              date_indexes=[pd.to_datetime("2022-06-01"), pd.to_datetime("2023-01-01")], granularity='1h')

# train_and_save_pipeline(
#     '/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/bitfinex2-BTCUSD-1h',
#     '/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/training_results/BTCUSD_1h')

validate_and_roll_train(RecurrentPPO,
                        '/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/data/bitfinex2-BTCUSD-1h',
                        '/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/training_results/BTCUSD_1h/run_20240107_143406')
