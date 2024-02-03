import gymnasium as gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from custom_env import TradingEnv
from trading_env_gym_custom.features_eng import simple_reward

print(TradingEnv)


def evaluate_model(model, env, unique_save_dir=None):
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

    truncated = False
    obs, _ = env.reset()
    total_log_return = 0
    while not truncated:
        action, _states = model.predict(obs, deterministic=True)  # Use deterministic predictions for evaluation
        obs, reward, done, truncated, historical_info = env.step(action)
        total_log_return += reward

    total_return = np.exp(total_log_return) - 1  # Subtracting 1 to get the net return (profit/loss)
    total_returns.append(total_return)

    mean_total_return = sum(total_returns)

    if unique_save_dir:
        env.get_wrapper_attr('save_for_render')(unique_save_dir)

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


def load_model(model_path):
    return PPO.load(model_path)


# Load CSV Data
def load_csv(file_path):
    return pd.read_csv(file_path)


# Calculate Sharpe Ratio (annualized for hourly data)
def sharpe_ratio(return_series, risk_free_rate=0, periods_per_year=8760):
    excess_ret = return_series - risk_free_rate
    return np.mean(excess_ret) / np.std(excess_ret) * np.sqrt(periods_per_year)


# Calculate Sortino Ratio
def sortino_ratio(series, risk_free_rate=0, periods_per_year=8760):
    negative_returns = series[series < risk_free_rate]
    return (np.mean(series) - risk_free_rate) / np.std(negative_returns) * np.sqrt(periods_per_year)


# Maximum Drawdown
def max_drawdown(return_series):
    comp_ret = (1 + return_series).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret / peak) - 1
    return dd.min()


# Total Return
def total_return(series):
    return (series.iloc[-1] / series.iloc[0] - 1) * 100


# Standard Deviation (Volatility)
def volatility(series):
    return np.std(series)


# Normalize and Plot Data
# def plot_data(portfolio, benchmark):
#     portfolio = (portfolio / portfolio.iloc[0]) * 100
#     benchmark = (benchmark / benchmark.iloc[0]) * 100
#     plt.figure(figsize=(12, 6))
#     plt.plot(portfolio, label="Portfolio")
#     plt.plot(benchmark, label="Benchmark (BTC)")
#     plt.xlabel("Date")
#     plt.ylabel("Normalized Value")
#     plt.title("Portfolio vs Benchmark")
#     plt.legend()
#     plt.show()


# Main function to run analysis
def analyze_data(file_path):
    data = load_csv(file_path)

    portfolio_valuation = data['portfolio_valuation']
    btc_price = data['data_close']

    portfolio_daily_returns = portfolio_valuation.pct_change().dropna()
    btc_daily_returns = btc_price.pct_change().dropna()

    # Portfolio KPIs
    print("Portfolio Sharpe Ratio:", sharpe_ratio(portfolio_daily_returns))
    print("Portfolio Sortino Ratio:", sortino_ratio(portfolio_daily_returns))
    print("Portfolio Max Drawdown:", max_drawdown(portfolio_daily_returns))
    print("Portfolio Total Return:", total_return(portfolio_valuation))
    print("Portfolio Volatility:", volatility(portfolio_daily_returns))

    # BTC (Benchmark) KPIs
    print("BTC Sharpe Ratio:", sharpe_ratio(btc_daily_returns))
    print("BTC Sortino Ratio:", sortino_ratio(btc_daily_returns))
    print("BTC Max Drawdown:", max_drawdown(btc_daily_returns))
    print("BTC Total Return:", total_return(btc_price))
    print("BTC Volatility:", volatility(btc_daily_returns))

    # plot_data(portfolio_valuation, btc_price)


# Example usage
analyze_data("/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/trading_env_gym_custom/evaluations/RecurrentPPO1706034756.1288917_infos.csv")