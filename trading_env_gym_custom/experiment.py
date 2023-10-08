import pandas as pd

import gymnasium as gym
import gym_trading_env
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

df = pd.read_pickle('data/binance-BTCUSDT-1h.pkl')
df["feature_close"] = df["close"].pct_change()

# Create the feature : open[t] / close[t]
df["feature_open"] = df["open"]/df["close"]

# Create the feature : high[t] / close[t]
df["feature_high"] = df["high"]/df["close"]

# Create the feature : low[t] / close[t]
df["feature_low"] = df["low"]/df["close"]
# more features


df.dropna(inplace=True)
env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees =0, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0, # 0.0003% per timestep (one timestep = 1h here)
    )

vec_env = DummyVecEnv([lambda: env])

# Initialize the PPO agent
policy_kwargs = dict(
    net_arch=[64, 64]  # Simple neural network architecture with two layers of 64 units each
)
model = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs, verbose=1)

model.learn(total_timesteps=1_000_000)

historical_info = env.get_wrapper_attr('historical_info')
# pickle it
rewards = [info['reward'] for info in historical_info]
plt.plot(rewards)
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Rewards Over Time')
plt.show()
