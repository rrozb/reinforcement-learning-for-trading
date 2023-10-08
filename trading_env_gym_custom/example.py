import gymnasium as gym
import gym_trading_env
import pandas as pd
from stable_baselines3 import A2C
df = pd.read_pickle("data/binance-BTCUSDT-1h.pkl")
df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
df["macd_signal"] = df["macd"].ewm(span=9).mean()
df["macd_diff"] = df["macd"] - df["macd_signal"]

env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
    )

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=len(df))

# done, truncated = False, False
# observation, info = env.reset()
# while not done and not truncated:
#     # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
#     position_index = env.action_space.sample() # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
#     observation, reward, done, truncated, info = env.step(position_index)