import gymnasium as gym
import numpy as np
from gymnasium import register
from stable_baselines3.common.monitor import Monitor

from custom_env import TradingEnv
from trading_env_gym_custom.features_eng import simple_reward

print(TradingEnv)



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


def evaluate_test(test_df, model, unique_save_dir, test_log_path=None, reward_function=simple_reward):
    eval_env = Monitor(gym.make("TradingEnv",
                                name="eval_BTCUSD",
                                df=test_df,
                                positions=[-1, 0, 1],
                                trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
                                borrow_interest_rate=0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here),
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
